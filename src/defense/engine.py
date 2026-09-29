import copy
import torch
from typing import Dict, List, Optional

from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine
from src.core.interfaces import ClientState
from src.core.model import vector_to_model, model_to_vector
from src.data.dataset import ClientDataset

from src.defense.config import DefenseConfig
from src.defense.aggregator import SoftRejectionAggregator
from src.defense.trust_tracker import TrustTracker


def check_inf_nan(weights: torch.Tensor, initial_weights: torch.Tensor) -> bool:
    """True = CLEAN, False = MALICIOUS (Inf/NaN detected or shape mismatch)."""
    if weights.shape != initial_weights.shape:
        return False
    delta = weights - initial_weights
    return not (torch.isnan(delta).any() or torch.isinf(delta).any())



class DefendedEnsembleEngine(HierarchicalEnsembleEngine):
    """
    Inherit HierarchicalEnsembleEngine, override run_round() to add defense.

    Phase 1: Defense at Cluster Head aggregation (aggregation ①).
    Phase 2: Add defense at Global Server aggregation (aggregation ②).
    """

    def __init__(self, config, topology, aggregator, device="cpu",
                 defense_config: Optional[DefenseConfig] = None):
        super().__init__(config, topology, aggregator, device)
        if defense_config is None:
            params = getattr(config.topology, "params", {})
            defense_config = DefenseConfig(
                defense_mode=params.get("defense_mode", "soft_cosine"),
                temperature=params.get("temperature", 1.0),
                temperature_decay=params.get("temperature_decay", 0.95),
                temperature_min=params.get("temperature_min", 0.1),
                defense_scope=params.get("defense_scope", "cluster"),
                norm_threshold=params.get("norm_threshold", 2.0),
            )
        self.defense_config = defense_config
        self.defense_aggregator = SoftRejectionAggregator(self.defense_config)
        self.trust_tracker = TrustTracker()

    def run_round(self, round_num: int):
        all_clients = list(range(self.config.clients.num_clients))

        # Lưu reference weights BEFORE round (để tính delta sau)

        pre_round_cluster_weights: Dict[int, torch.Tensor] = {}
        for hid in self.cluster_heads_state:
            pre_round_cluster_weights[hid] = self.cluster_heads_state[hid].weights.clone()

        pre_round_server_weights = self.server_weights.clone()

        # 1. Prepare for distribution
        
        cluster_updates_parent = {hid: [] for hid in self.cluster_heads_state.keys()}
        cluster_updates_root = {hid: [] for hid in self.cluster_heads_state.keys()}

        # Pre-extract cluster head fc2_parent states for shared-backbone multi-head synchronization
        use_shared_backbone = (
            getattr(self.config.clients, "compute_optimization_mode", None) == "shared_backbone"
            and hasattr(self.updater, "multihead_model")
            and hasattr(self.updater.multihead_model, "fc2_parent")
        )
        head_parent_states = {}
        if use_shared_backbone:
            temp_model = copy.deepcopy(self.updater.multihead_model)
            for hid, h_state in self.cluster_heads_state.items():
                if h_state.weights is not None and len(h_state.weights) == len(model_to_vector(temp_model)):
                    vector_to_model(h_state.weights.to(self.device), temp_model)
                    head_parent_states[hid] = {k: v.clone() for k, v in temp_model.fc2_parent.state_dict().items()}

        for client_id in all_clients:
            self.clients_state[client_id].weights = self.server_weights.clone()
            head_id = self.topology.get_neighbors(client_id)[0]
            self.clients_state[client_id].parent_weights = (
                self.cluster_heads_state[head_id].weights.clone()
            )
            if head_id in head_parent_states:
                self.clients_state[client_id].parent_head_state = {
                    k: v.clone() for k, v in head_parent_states[head_id].items()
                }

        # 2. Local Updates
        current_lr = self.get_current_lr(round_num)
        for client_id in all_clients:
            state = self.clients_state[client_id].copy()
            client_ds = self.client_train_datasets[client_id]

            updated_state = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng,
                current_lr=current_lr,
            )
            self.clients_state[client_id] = updated_state

            head_id = self.topology.get_neighbors(client_id)[0]

            s_parent = updated_state.copy()
            if updated_state.parent_weights is not None:
                s_parent.weights = updated_state.parent_weights
            cluster_updates_parent[head_id].append(s_parent)

            cluster_updates_root[head_id].append(updated_state)

        # === INF/NAN SENTINEL ===
        for client_id in all_clients:
            state = self.clients_state[client_id]
            head_id = self.topology.get_neighbors(client_id)[0]
            has_explosion = (
                not check_inf_nan(state.weights, pre_round_server_weights)
                or (state.parent_weights is not None and not check_inf_nan(state.parent_weights, pre_round_cluster_weights[head_id]))
            )
            if has_explosion:
                state.is_confirmed_malicious = True  # dynamic attr, no interfaces.py edit
                print(f"[Round {round_num}] ⚠️ SENTINEL: Client {client_id} — "
                      f"gradient explosion → excluded from aggregation.")

        # Filter out confirmed malicious clients before aggregation
        for hid in self.cluster_heads_state.keys():
            cluster_updates_parent[hid] = [
                s for s in cluster_updates_parent[hid]
                if not getattr(self.clients_state[s.client_id], "is_confirmed_malicious", False)
                and check_inf_nan(s.weights, pre_round_cluster_weights[hid])
            ]
            cluster_updates_root[hid] = [
                s for s in cluster_updates_root[hid]
                if not getattr(self.clients_state[s.client_id], "is_confirmed_malicious", False)
                and check_inf_nan(s.weights, pre_round_server_weights)
            ]

        # 3. Intra-cluster Aggregation - DEFENSE
        defense_scope = self.defense_config.defense_scope

        for hid, states in cluster_updates_parent.items():
            ref = pre_round_cluster_weights[hid]
            clean_states = [
                s for s in states
                if not getattr(s, "is_confirmed_malicious", False)
                and check_inf_nan(s.weights, ref)
            ]
            if not clean_states:
                # Fallback to reference_weights if all states are corrupt
                self.cluster_heads_state[hid].weights = ref.clone()
                continue

            if defense_scope in ("cluster", "both"):
                # Dùng SoftRejectionAggregator thay FedAvg
                agg_weights_parent = self.defense_aggregator.aggregate(
                    clean_states,
                    reference_weights=ref,
                )
                # Ghi trust scores cho round này
                self.trust_tracker.log(
                    round_num, hid, self.defense_aggregator.get_last_trust_scores()
                )
            else:
                # Không defense ở cluster → FedAvg bình thường
                agg_weights_parent = self.aggregator.aggregate(clean_states)

            if not check_inf_nan(agg_weights_parent, ref):
                agg_weights_parent = ref.clone()

            self.cluster_heads_state[hid].weights = agg_weights_parent

        # 4. Global Aggregation
        all_root_contributions: List[ClientState] = []
        for hid, states in cluster_updates_root.items():
            clean = [
                s for s in states
                if not getattr(s, "is_confirmed_malicious", False)
                and check_inf_nan(s.weights, pre_round_server_weights)
            ]
            all_root_contributions.extend(clean)

        if all_root_contributions:
            if defense_scope in ("global", "both"):
                # Phase 2: Defense ở Global Server
                self.server_weights = self.defense_aggregator.aggregate(
                    all_root_contributions,
                    reference_weights=pre_round_server_weights,
                )
                # Log trust scores cho global level
                self.trust_tracker.log(
                    round_num, -1, self.defense_aggregator.get_last_trust_scores()
                )
            else:
                # Phase 1: FedAvg bình thường ở global
                self.server_weights = self.aggregator.aggregate(all_root_contributions)

            if not check_inf_nan(self.server_weights, pre_round_server_weights):
                self.server_weights = pre_round_server_weights.clone()
        else:
            self.server_weights = pre_round_server_weights.clone()

        # 5. Temperature decay
        
        self.defense_aggregator.decay_temperature()

        # 6. Metrics
        
        acc, test_loss = self.evaluate_model(self.server_weights)

        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": len(all_clients),
            "total_clients_targeted": len(all_clients),
            "defense_temperature": self.defense_aggregator.current_temperature,
        }

        if self.config.clients.use_ensemble or self.config.clients.hierarchical_ensemble:
            ens_acc, ens_loss = self.evaluate_ensemble()
            round_data["ensemble_test_accuracy"] = ens_acc
            round_data["ensemble_test_loss"] = ens_loss

        self.metrics.log_round(round_data)
