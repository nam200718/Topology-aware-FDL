"""
Unified Multi-Tier Defended Engine for Heterogeneous Federated Learning.
Unites Hierarchical Ensemble Personalization (FedHEP) and Multi-Tier Byzantine Defense.

Features:
1. Multi-Scale Parameter Coordination (Global, Cluster, and On-Device Personalization Heads).
2. Skew-Calibrated Soft Cosine Rejection: Eliminates false-positive penalties on specialized clients.
3. Sentinel Pre-Filter: Disqualifies NaN / Inf gradient explosions immediately.
4. Temporal Client Trust Tracking: Tracks dynamic client trust evolution across rounds.
5. Tail Fairness Evaluation: Computes worst-client min utility and bottom-10% fairness metrics.
"""

import copy
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine
from src.core.interfaces import ClientState
from src.core.model import vector_to_model, model_to_vector
from src.core.aggregator import DeltaSpaceRobustAggregator, compute_buffer_slices, compute_classifier_slices
from src.defense.config import DefenseConfig
from src.defense.trust_tracker import TrustTracker


class UnifiedDefendedEngine(HierarchicalEnsembleEngine):
    """
    Unified Multi-Tier Defended Engine.
    Coordinates hierarchical ensemble personalization (FedHEP) with multi-tier Byzantine defense.
    """
    def __init__(self, config, topology, aggregator, device="cpu",
                 defense_config: Optional[DefenseConfig] = None):
        super().__init__(config, topology, aggregator, device)

        if defense_config is None:
            params = getattr(config.topology, "params", {})
            defense_config = DefenseConfig(
                defense_mode=params.get("defense_mode", "soft_cosine"),
                temperature=params.get("temperature", 0.5),
                temperature_decay=params.get("temperature_decay", 0.95),
                temperature_min=params.get("temperature_min", 0.1),
                defense_scope=params.get("defense_scope", "both"),
                norm_threshold=params.get("norm_threshold", 2.0),
            )
        self.defense_config = defense_config
        self.trust_tracker = TrustTracker()

        # Multi-Tier Skew-Calibrated Robust Aggregators
        # Tier 1: Cluster / Coalition Head Aggregator
        self.cluster_defense_aggregator = DeltaSpaceRobustAggregator(
            mode="soft_cosine",
            temperature=self.defense_config.temperature,
            norm_bound_k=getattr(self.defense_config, "norm_bounding_multiplier", 3.0),
            adaptive_temperature=getattr(self.defense_config, "adaptive_temperature", True),
            hard_rejection=getattr(self.defense_config, "hard_rejection_enabled", True),
            hard_threshold=getattr(self.defense_config, "hard_rejection_threshold", -0.1),
            buffer_slices=compute_buffer_slices(self.updater.multihead_model),
            classifier_slice=compute_classifier_slices(self.updater.multihead_model),
        )

        # Tier 2: Global Central Server Aggregator
        self.global_defense_aggregator = DeltaSpaceRobustAggregator(
            mode="soft_cosine",
            temperature=self.defense_config.temperature,
            norm_bound_k=3.0,
            adaptive_temperature=True,
            hard_rejection=True,
            hard_threshold=-0.1,
            buffer_slices=compute_buffer_slices(self.updater.multihead_model),
            classifier_slice=compute_classifier_slices(self.updater.multihead_model),
        )

        # Cumulative Client Trust Store: {client_id: cumulative_trust}
        self.client_trust_scores: Dict[int, float] = {
            cid: 1.0 for cid in range(self.config.clients.num_clients)
        }
        self.reputation_momentum = 0.85
        self.malicious_reputation_threshold = getattr(self.defense_config, "malicious_reputation_threshold", 0.25)

    @property
    def agent_reputations(self) -> Dict[int, float]:
        """Backward-compatible alias for client_trust_scores."""
        return self.client_trust_scores

    @agent_reputations.setter
    def agent_reputations(self, val: Dict[int, float]):
        self.client_trust_scores = val

    def run_round(self, round_num: int):
        num_total = self.config.clients.num_clients
        all_clients = list(range(num_total))

        # References before local updates for delta computation
        root_reference = self.server_weights.clone()
        parent_references = {
            hid: self.cluster_heads_state[hid].weights.clone()
            for hid in self.cluster_heads_state.keys()
        }

        # 1. Distribute Models to Clients
        cluster_updates_parent = {hid: [] for hid in self.cluster_heads_state.keys()}
        cluster_updates_root = {hid: [] for hid in self.cluster_heads_state.keys()}
        client_active_masks = {}

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

        # 2. Local Client Updates
        current_lr = self.get_current_lr(round_num)
        current_deltas = {}
        for client_id in all_clients:
            state = self.clients_state[client_id].copy()
            client_ds = self.client_train_datasets[client_id]

            updated_state = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng,
                current_lr=current_lr
            )
            updated_state.participation_count = (
                getattr(self.clients_state[client_id], "participation_count", 0) + 1
            )
            self.clients_state[client_id] = updated_state
            head_id = self.topology.get_neighbors(client_id)[0]

            # Track active class mask if available
            if hasattr(updated_state, "active_mask") and updated_state.active_mask is not None:
                client_active_masks[client_id] = updated_state.active_mask

            # Delta vector calculation
            delta = (updated_state.weights - root_reference).detach()
            current_deltas[client_id] = delta

            # === SENTINEL PRE-FILTER: Disqualify NaN / Inf gradient explosions ===
            has_nan_inf = (
                torch.isnan(delta).any()
                or torch.isinf(delta).any()
                or (
                    updated_state.parent_weights is not None
                    and (torch.isnan(updated_state.parent_weights).any() or torch.isinf(updated_state.parent_weights).any())
                )
            )
            if has_nan_inf:
                self.clients_state[client_id].is_confirmed_malicious = True
                print(f"[Round {round_num}] ⚠️ SENTINEL GUARD: Client {client_id} gradient explosion → zeroed out.")
                continue

            if getattr(self.clients_state[client_id], "is_confirmed_malicious", False):
                print(f"[Round {round_num}] ⚠️ ISOLATION: Confirmed malicious client {client_id} excluded from aggregation.")
                continue

            s_parent = updated_state.copy()
            if updated_state.parent_weights is not None:
                s_parent.weights = updated_state.parent_weights
            cluster_updates_parent[head_id].append(s_parent)
            cluster_updates_root[head_id].append(updated_state)

        # 3. Tier 1: Intra-Coalition / Cluster Defense
        defense_scope = self.defense_config.defense_scope
        for hid, states in cluster_updates_parent.items():
            if not states:
                continue
            ref = parent_references[hid]
            clean_states = [s for s in states if not getattr(s, "is_confirmed_malicious", False)]
            if not clean_states:
                continue

            if defense_scope in ("cluster", "both"):
                # Compute deltas relative to parent reference
                p_deltas = [s.weights - ref for s in clean_states]
                masks = [client_active_masks.get(s.client_id) for s in clean_states]
                stacked_masks = torch.stack(masks).to(ref.device) if all(m is not None for m in masks) else None

                agg_delta = self.cluster_defense_aggregator.aggregate_deltas(
                    p_deltas, reference=ref, active_masks=stacked_masks
                )
                self.cluster_heads_state[hid].weights = ref + agg_delta

                # Update Trust Tracker & Client Trust Scores for cluster members
                last_trust = self.cluster_defense_aggregator.get_last_trust_scores()
                if last_trust is not None:
                    valid_states = [s for s, d in zip(clean_states, p_deltas) if not (torch.isnan(d).any() or torch.isinf(d).any())]
                    trust_dict = {
                        valid_states[i].client_id: float(last_trust[i].item())
                        for i in range(min(len(valid_states), len(last_trust)))
                    }
                    self.trust_tracker.log(round_num, hid, trust_dict)
                    n_members = len(clean_states)
                    for cid, t_score in trust_dict.items():
                        norm_t = min(1.0, t_score * n_members)
                        self.client_trust_scores[cid] = (
                            self.reputation_momentum * self.client_trust_scores[cid]
                            + (1.0 - self.reputation_momentum) * norm_t
                        )
                        if self.client_trust_scores[cid] < self.malicious_reputation_threshold:
                            self.clients_state[cid].is_confirmed_malicious = True
                            print(f"[Round {round_num}] ⚠️ TRUST ISOLATION: Client {cid} trust score decayed to {self.client_trust_scores[cid]:.3f} < {self.malicious_reputation_threshold} → permanently isolated.")
            else:
                self.cluster_heads_state[hid].weights = self.aggregator.aggregate(clean_states)

        # 4. Tier 2: Global Central Server Defense
        all_root_contributions: List[ClientState] = []
        for hid, states in cluster_updates_root.items():
            all_root_contributions.extend(
                [s for s in states if not getattr(s, "is_confirmed_malicious", False)]
            )

        if all_root_contributions:
            if defense_scope in ("global", "both"):
                r_deltas = [s.weights - root_reference for s in all_root_contributions]
                masks = [client_active_masks.get(s.client_id) for s in all_root_contributions]
                stacked_masks = torch.stack(masks).to(root_reference.device) if all(m is not None for m in masks) else None

                agg_delta = self.global_defense_aggregator.aggregate_deltas(
                    r_deltas, reference=root_reference, active_masks=stacked_masks
                )
                self.server_weights = root_reference + agg_delta

                # Log global trust scores & update cumulative trust scores
                last_trust_global = self.global_defense_aggregator.get_last_trust_scores()
                if last_trust_global is not None:
                    valid_global_states = [s for s, d in zip(all_root_contributions, r_deltas) if not (torch.isnan(d).any() or torch.isinf(d).any())]
                    global_dict = {
                        valid_global_states[i].client_id: float(last_trust_global[i].item())
                        for i in range(min(len(valid_global_states), len(last_trust_global)))
                    }
                    self.trust_tracker.log(round_num, -1, global_dict)
                    n_global = len(all_root_contributions)
                    for cid, t_score in global_dict.items():
                        norm_t = min(1.0, t_score * n_global)
                        self.client_trust_scores[cid] = (
                            self.reputation_momentum * self.client_trust_scores[cid]
                            + (1.0 - self.reputation_momentum) * norm_t
                        )
                        if self.client_trust_scores[cid] < self.malicious_reputation_threshold:
                            self.clients_state[cid].is_confirmed_malicious = True
                            print(f"[Round {round_num}] ⚠️ GLOBAL TRUST ISOLATION: Client {cid} trust score decayed to {self.client_trust_scores[cid]:.3f} < {self.malicious_reputation_threshold} → permanently isolated.")
            else:
                self.server_weights = self.aggregator.aggregate(all_root_contributions)

        # 5. Metrics & Tail Fairness Tracking
        if not self.should_evaluate(round_num):
            self.metrics.log_round({
                "round": round_num,
                "participating_clients": num_total,
                "evaluated": False,
            })
            return

        acc, test_loss = self.evaluate_model(self.server_weights)
        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": num_total,
            "evaluated": True,
        }

        if self.config.clients.use_ensemble or self.config.clients.hierarchical_ensemble:
            ens_acc, ens_loss = self.evaluate_ensemble(round_num)
            round_data["ensemble_test_accuracy"] = ens_acc
            round_data["ensemble_test_loss"] = ens_loss

            # Tail Fairness & Worst-Client Utility
            if self._last_per_client_accuracy:
                accs = sorted(self._last_per_client_accuracy.values())
                k = max(1, int(np.ceil(0.1 * len(accs))))
                round_data["bottom10_fairness"] = float(np.mean(accs[:k]))
                round_data["worst_client_min_accuracy"] = float(accs[0])
                round_data["rawlsian_min_accuracy"] = float(accs[0])
                round_data["accuracy_variance"] = float(np.var(accs))

        self.metrics.log_round(round_data)
