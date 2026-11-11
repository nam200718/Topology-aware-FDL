"""
Oreh Engines: Multi-Tier and Centralized Byzantine Resilient Federated Learning Engines
supporting targeted tier attacks (root_only vs parent_only vs all), ALIE model poisoning,
defense scope ablations (root/global vs cluster vs both), and baselines (FedAvg, Ditto, Multi-Krum).
Inherits from UnifiedDefendedEngine and CentralizedEngine without modifying existing codebase files.
"""

import copy
import time
from typing import Dict, List, Optional, Literal, Any
import numpy as np
import torch
import torch.nn.functional as F

from src.core.unified_defended_engine import UnifiedDefendedEngine
from src.core.centralized_engine import CentralizedEngine
from src.core.interfaces import ClientState
from src.core.model import vector_to_model, model_to_vector
from src.core.aggregator import (
    DeltaSpaceRobustAggregator,
    compute_buffer_slices,
    compute_classifier_slices,
)
from src.defense.config import DefenseConfig

from oreh.updater import OrehLocalUpdater
from oreh.attacks import (
    generate_alie_delta,
    compute_alie_z_max,
    generate_ipm_delta,
    generate_adaptive_cosine_delta,
)


class OrehDefendedEngine(UnifiedDefendedEngine):
    """
    Subclasses UnifiedDefendedEngine to provide:
    1. Targeted Tier Attack simulation ('all', 'root_only', 'parent_only').
    2. Coordinated ALIE attack execution (calculating empirical mean and std from benign deltas).
    3. Flexible Defense Scopes ('cluster', 'global', 'both', 'none').
    4. Granular per-tier latency tracking (ms) for cost-benefit evaluation.
    5. Detailed attacker vs honest trust score recording.
    6. Multi-vector Byzantine attacks (ALIE, IPM, Adaptive Cosine, Delayed Onset).
    """

    def __init__(
        self,
        config,
        topology,
        aggregator,
        device="cpu",
        defense_config: Optional[DefenseConfig] = None,
        attack_target_tier: Literal["all", "root_only", "parent_only"] = "all",
        alie_z_max: Optional[float] = None,
        ipm_epsilon: float = 0.1,
        attack_onset_round: int = 15,
    ):
        super().__init__(
            config=config,
            topology=topology,
            aggregator=aggregator,
            device=device,
            defense_config=defense_config,
        )
        self.attack_target_tier = attack_target_tier
        self.alie_z_max = alie_z_max
        self.ipm_epsilon = ipm_epsilon
        self.attack_onset_round = attack_onset_round

        # Replace default updater with OrehLocalUpdater
        self.updater = OrehLocalUpdater(
            device=device,
            in_channels=self.in_channels,
            model_name=config.clients.model_name,
            num_classes=self.num_classes,
            attack_target_tier=attack_target_tier,
            alie_z_max=alie_z_max,
            ipm_epsilon=ipm_epsilon,
            attack_onset_round=attack_onset_round,
        )

        # Performance profiling logs
        self.cluster_latency_history: List[float] = []
        self.global_latency_history: List[float] = []
        self.total_defense_latency_history: List[float] = []

        # Attacker tracking logs
        self.byzantine_trust_history: List[float] = []
        self.honest_trust_history: List[float] = []

    def run_round(self, round_num: int):
        num_total = self.config.clients.num_clients
        all_clients = list(range(num_total))

        # References before local updates
        root_reference = self.server_weights.clone()
        parent_references = {
            hid: self.cluster_heads_state[hid].weights.clone()
            for hid in self.cluster_heads_state.keys()
        }

        # 1. Distribute parameters to clients
        head_parent_states = {}
        for hid in self.cluster_heads_state.keys():
            head_m = copy.deepcopy(self.updater.multihead_model)
            vector_to_model(parent_references[hid].to(self.device), head_m)
            if hasattr(head_m, "fc2_parent"):
                head_parent_states[hid] = {k: v.clone() for k, v in head_m.fc2_parent.state_dict().items()}
            elif hasattr(head_m, "parent_head"):
                head_parent_states[hid] = {k: v.clone() for k, v in head_m.parent_head.state_dict().items()}

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

        current_lr = self.get_current_lr(round_num)
        client_active_masks = {}

        # Identify honest and Byzantine clients
        honest_clients = [
            cid for cid in all_clients
            if not getattr(self.clients_state[cid], "is_byzantine", False)
        ]
        byzantine_clients = [
            cid for cid in all_clients
            if getattr(self.clients_state[cid], "is_byzantine", False)
        ]

        # 2A. First pass: compute local updates for HONEST clients
        honest_root_deltas: List[torch.Tensor] = []
        honest_parent_deltas: Dict[int, List[torch.Tensor]] = {
            hid: [] for hid in self.cluster_heads_state.keys()
        }

        for client_id in honest_clients:
            state = self.clients_state[client_id].copy()
            client_ds = self.client_train_datasets[client_id]

            updated_state = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng,
                current_lr=current_lr,
            )
            updated_state.participation_count = (
                getattr(self.clients_state[client_id], "participation_count", 0) + 1
            )
            self.clients_state[client_id] = updated_state
            head_id = self.topology.get_neighbors(client_id)[0]

            r_delta = (updated_state.weights - root_reference).detach()
            honest_root_deltas.append(r_delta)

            if updated_state.parent_weights is not None:
                p_ref = parent_references[head_id]
                p_delta = (updated_state.parent_weights - p_ref).detach()
                honest_parent_deltas[head_id].append(p_delta)

            if hasattr(updated_state, "active_mask") and updated_state.active_mask is not None:
                client_active_masks[client_id] = updated_state.active_mask

        # 2B. Second pass: compute updates for BYZANTINE clients (supporting ALIE, IPM, Adaptive Cosine, Delayed Onset)
        alie_root_delta = None
        alie_parent_deltas = {}
        ipm_root_delta = None
        ipm_parent_deltas = {}
        adaptive_root_delta = None
        adaptive_parent_deltas = {}

        is_alie = any(getattr(self.clients_state[cid], "byzantine_type", "") == "alie" for cid in byzantine_clients)
        is_ipm = any(getattr(self.clients_state[cid], "byzantine_type", "") == "ipm" for cid in byzantine_clients)
        is_adaptive = any(getattr(self.clients_state[cid], "byzantine_type", "") == "adaptive_cosine" for cid in byzantine_clients)

        if is_alie and honest_root_deltas:
            alie_root_delta = generate_alie_delta(
                honest_root_deltas,
                z_max=self.alie_z_max,
                num_total=num_total,
                num_attackers=len(byzantine_clients),
            )
            for hid in self.cluster_heads_state.keys():
                if honest_parent_deltas[hid]:
                    alie_parent_deltas[hid] = generate_alie_delta(
                        honest_parent_deltas[hid],
                        z_max=self.alie_z_max,
                        num_total=len(self.topology.get_cluster_members(hid)) if hasattr(self.topology, "get_cluster_members") else num_total,
                        num_attackers=sum(1 for cid in byzantine_clients if self.topology.get_neighbors(cid)[0] == hid),
                    )
                else:
                    alie_parent_deltas[hid] = alie_root_delta

        if is_ipm and honest_root_deltas:
            ipm_root_delta = generate_ipm_delta(honest_root_deltas, epsilon=self.ipm_epsilon)
            for hid in self.cluster_heads_state.keys():
                if honest_parent_deltas[hid]:
                    ipm_parent_deltas[hid] = generate_ipm_delta(honest_parent_deltas[hid], epsilon=self.ipm_epsilon)
                else:
                    ipm_parent_deltas[hid] = ipm_root_delta

        if is_adaptive and honest_root_deltas:
            prev_byz_trust = float(np.mean([self.client_trust_scores.get(c, 1.0) for c in byzantine_clients])) if byzantine_clients else 1.0
            adaptive_root_delta = generate_adaptive_cosine_delta(honest_root_deltas, prev_trust_score=prev_byz_trust)
            for hid in self.cluster_heads_state.keys():
                if honest_parent_deltas[hid]:
                    adaptive_parent_deltas[hid] = generate_adaptive_cosine_delta(honest_parent_deltas[hid], prev_trust_score=prev_byz_trust)
                else:
                    adaptive_parent_deltas[hid] = adaptive_root_delta

        for client_id in byzantine_clients:
            state = self.clients_state[client_id].copy()
            state.attack_target_tier = self.attack_target_tier
            head_id = self.topology.get_neighbors(client_id)[0]
            state.parent_reference_weights = parent_references[head_id]
            state.current_round = round_num
            state.attack_onset_round = self.attack_onset_round
            state.prev_trust_score = self.client_trust_scores.get(client_id, 1.0)

            if alie_root_delta is not None:
                state.alie_root_delta = alie_root_delta
                state.alie_parent_delta = alie_parent_deltas.get(head_id, alie_root_delta)

            if ipm_root_delta is not None:
                state.ipm_root_delta = ipm_root_delta
                state.ipm_parent_delta = ipm_parent_deltas.get(head_id, ipm_root_delta)

            if adaptive_root_delta is not None:
                state.adaptive_root_delta = adaptive_root_delta
                state.adaptive_parent_delta = adaptive_parent_deltas.get(head_id, adaptive_root_delta)

            client_ds = self.client_train_datasets[client_id]
            updated_state = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng,
                current_lr=current_lr,
            )
            updated_state.participation_count = (
                getattr(self.clients_state[client_id], "participation_count", 0) + 1
            )
            self.clients_state[client_id] = updated_state

            if hasattr(updated_state, "active_mask") and updated_state.active_mask is not None:
                client_active_masks[client_id] = updated_state.active_mask

        # Group updates for Tier 1 and Tier 2
        cluster_updates_parent: Dict[int, List[ClientState]] = {
            hid: [] for hid in self.cluster_heads_state.keys()
        }
        cluster_updates_root: Dict[int, List[ClientState]] = {
            hid: [] for hid in self.cluster_heads_state.keys()
        }

        for client_id in all_clients:
            updated_state = self.clients_state[client_id]
            head_id = self.topology.get_neighbors(client_id)[0]

            delta = (updated_state.weights - root_reference).detach()
            has_nan_inf = (
                torch.isnan(delta).any()
                or torch.isinf(delta).any()
                or (
                    updated_state.parent_weights is not None
                    and (
                        torch.isnan(updated_state.parent_weights).any()
                        or torch.isinf(updated_state.parent_weights).any()
                    )
                )
            )
            if has_nan_inf:
                self.clients_state[client_id].is_confirmed_malicious = True
                continue

            if getattr(self.clients_state[client_id], "is_confirmed_malicious", False):
                continue

            s_parent = updated_state.copy()
            if updated_state.parent_weights is not None:
                s_parent.weights = updated_state.parent_weights
            cluster_updates_parent[head_id].append(s_parent)
            cluster_updates_root[head_id].append(updated_state)

        # 3. Tier 1: Intra-Coalition / Cluster Head Defense
        defense_scope = getattr(self.defense_config, "defense_scope", "both")
        t_cluster_start = time.perf_counter()

        for hid, states in cluster_updates_parent.items():
            if not states:
                continue
            ref = parent_references[hid]
            clean_states = [s for s in states if not getattr(s, "is_confirmed_malicious", False)]
            if not clean_states:
                continue

            if defense_scope in ("cluster", "both"):
                p_deltas = [s.weights - ref for s in clean_states]
                masks = [client_active_masks.get(s.client_id) for s in clean_states]
                stacked_masks = (
                    torch.stack(masks).to(ref.device)
                    if all(m is not None for m in masks)
                    else None
                )
                agg_delta = self.cluster_defense_aggregator.aggregate_deltas(
                    p_deltas, reference=ref, active_masks=stacked_masks
                )
                self.cluster_heads_state[hid].weights = ref + agg_delta

                last_trust = self.cluster_defense_aggregator.get_last_trust_scores()
                if last_trust is not None:
                    valid_states = [
                        s for s, d in zip(clean_states, p_deltas)
                        if not (torch.isnan(d).any() or torch.isinf(d).any())
                    ]
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
            else:
                self.cluster_heads_state[hid].weights = self.aggregator.aggregate(clean_states)

        t_cluster_end = time.perf_counter()
        cluster_latency_ms = (t_cluster_end - t_cluster_start) * 1000.0
        self.cluster_latency_history.append(cluster_latency_ms)

        # 4. Tier 2: Global Central Server Defense
        t_global_start = time.perf_counter()
        all_root_contributions: List[ClientState] = []
        for hid, states in cluster_updates_root.items():
            all_root_contributions.extend(
                [s for s in states if not getattr(s, "is_confirmed_malicious", False)]
            )

        if all_root_contributions:
            if defense_scope in ("global", "both"):
                r_deltas = [s.weights - root_reference for s in all_root_contributions]
                masks = [client_active_masks.get(s.client_id) for s in all_root_contributions]
                stacked_masks = (
                    torch.stack(masks).to(root_reference.device)
                    if all(m is not None for m in masks)
                    else None
                )
                agg_delta = self.global_defense_aggregator.aggregate_deltas(
                    r_deltas, reference=root_reference, active_masks=stacked_masks
                )
                self.server_weights = root_reference + agg_delta

                last_trust_global = self.global_defense_aggregator.get_last_trust_scores()
                if last_trust_global is not None:
                    valid_global_states = [
                        s for s, d in zip(all_root_contributions, r_deltas)
                        if not (torch.isnan(d).any() or torch.isinf(d).any())
                    ]
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
            else:
                self.server_weights = self.aggregator.aggregate(all_root_contributions)

        t_global_end = time.perf_counter()
        global_latency_ms = (t_global_end - t_global_start) * 1000.0
        self.global_latency_history.append(global_latency_ms)
        self.total_defense_latency_history.append(cluster_latency_ms + global_latency_ms)

        # Track average trust scores of honest vs byzantine clients
        if byzantine_clients:
            byz_trust = [self.client_trust_scores.get(c, 1.0) for c in byzantine_clients]
            self.byzantine_trust_history.append(float(np.mean(byz_trust)))
        if honest_clients:
            hon_trust = [self.client_trust_scores.get(c, 1.0) for c in honest_clients]
            self.honest_trust_history.append(float(np.mean(hon_trust)))

        # 5. Decay defense temperature
        if hasattr(self.cluster_defense_aggregator, "decay_temperature"):
            self.cluster_defense_aggregator.decay_temperature()
        if hasattr(self.global_defense_aggregator, "decay_temperature"):
            self.global_defense_aggregator.decay_temperature()

        # 6. Evaluation metrics
        if not self.should_evaluate(round_num):
            self.metrics.log_round({
                "round": round_num,
                "participating_clients": len(all_clients),
                "total_clients_targeted": num_total,
                "evaluated": False,
            })
            return

        acc, test_loss = self.evaluate_model(self.server_weights)
        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": len(all_clients),
            "total_clients_targeted": num_total,
            "evaluated": True,
        }
        if self.byzantine_trust_history:
            round_data["byzantine_trust_score"] = float(self.byzantine_trust_history[-1])
        if self.honest_trust_history:
            round_data["honest_trust_score"] = float(self.honest_trust_history[-1])

        if getattr(self.config.clients, "use_ensemble", False) or getattr(self.config.clients, "hierarchical_ensemble", False):
            ens_acc, ens_loss = self.evaluate_ensemble(round_num)
            round_data["ensemble_test_accuracy"] = ens_acc
            round_data["ensemble_test_loss"] = ens_loss
            round_data["personalized_test_acc"] = ens_acc
            if round_num == self.config.num_rounds and getattr(self, "_last_per_client_accuracy", None):
                accs = sorted(self._last_per_client_accuracy.values())
                k = max(1, int(np.ceil(0.1 * len(accs))))
                fairness_val = float(np.mean(accs[:k]))
                round_data["bottom10_fairness"] = fairness_val
                round_data["bottom_10_acc"] = fairness_val
                round_data["per_client_accuracy"] = {
                    str(cid): round(a, 4) for cid, a in sorted(self._last_per_client_accuracy.items())
                }

        self.metrics.log_round(round_data)


class OrehCentralizedEngine(CentralizedEngine):
    """
    Subclasses CentralizedEngine to support ALIE Byzantine attacks
    across Star-topology baselines (FedAvg, Ditto, Multi-Krum).
    """

    def __init__(
        self,
        config,
        topology,
        aggregator,
        device="cpu",
        alie_z_max: Optional[float] = None,
        ipm_epsilon: float = 0.1,
        attack_onset_round: int = 15,
    ):
        super().__init__(config, topology, aggregator, device)
        self.alie_z_max = alie_z_max
        self.ipm_epsilon = ipm_epsilon
        self.attack_onset_round = attack_onset_round
        self.updater = OrehLocalUpdater(
            device=device,
            in_channels=self.in_channels,
            model_name=config.clients.model_name,
            num_classes=self.num_classes,
            attack_target_tier="all",
            alie_z_max=alie_z_max,
            ipm_epsilon=ipm_epsilon,
            attack_onset_round=attack_onset_round,
        )

    def run_round(self, round_num: int):
        target_clients = self.topology.get_server_connected_clients()

        # Download global model
        for client_id in target_clients:
            self.clients_state[client_id].weights = self.server_weights.clone()

        current_lr = self.get_current_lr(round_num)

        honest_clients = [
            cid for cid in target_clients
            if not getattr(self.clients_state[cid], "is_byzantine", False)
        ]
        byzantine_clients = [
            cid for cid in target_clients
            if getattr(self.clients_state[cid], "is_byzantine", False)
        ]

        # 1. Update honest clients first
        honest_deltas = []
        updated_states = {}
        for client_id in honest_clients:
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
            updated_states[client_id] = updated_state
            delta = (updated_state.weights - self.server_weights).detach()
            honest_deltas.append(delta)

        # 2. Attack delta generation for centralized attackers
        alie_delta = None
        ipm_delta = None
        adaptive_delta = None

        is_alie = any(getattr(self.clients_state[cid], "byzantine_type", "") == "alie" for cid in byzantine_clients)
        is_ipm = any(getattr(self.clients_state[cid], "byzantine_type", "") == "ipm" for cid in byzantine_clients)
        is_adaptive = any(getattr(self.clients_state[cid], "byzantine_type", "") == "adaptive_cosine" for cid in byzantine_clients)

        if is_alie and honest_deltas:
            alie_delta = generate_alie_delta(
                honest_deltas,
                z_max=self.alie_z_max,
                num_total=len(target_clients),
                num_attackers=len(byzantine_clients),
            )

        if is_ipm and honest_deltas:
            ipm_delta = generate_ipm_delta(honest_deltas, epsilon=self.ipm_epsilon)

        if is_adaptive and honest_deltas:
            adaptive_delta = generate_adaptive_cosine_delta(honest_deltas, prev_trust_score=1.0)

        # 3. Update byzantine clients
        for client_id in byzantine_clients:
            state = self.clients_state[client_id].copy()
            state.current_round = round_num
            state.attack_onset_round = self.attack_onset_round
            if alie_delta is not None:
                state.alie_root_delta = alie_delta
            if ipm_delta is not None:
                state.ipm_root_delta = ipm_delta
            if adaptive_delta is not None:
                state.adaptive_root_delta = adaptive_delta
            client_ds = self.client_train_datasets[client_id]

            updated_state = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng,
                current_lr=current_lr,
            )
            self.clients_state[client_id] = updated_state
            updated_states[client_id] = updated_state

        all_updated_states = [
            updated_states[cid] for cid in target_clients if cid in updated_states
        ]

        # 4. Aggregate
        if all_updated_states:
            expected = len(self.server_weights)
            bad = [s.client_id for s in all_updated_states if len(s.weights) != expected]
            if bad:
                all_updated_states = [s for s in all_updated_states if len(s.weights) == expected]
            if all_updated_states:
                self.server_weights = self.aggregator.aggregate(all_updated_states)

        # 5. Metrics evaluation
        if not self.should_evaluate(round_num):
            self.metrics.log_round({
                "round": round_num,
                "participating_clients": len(target_clients),
                "total_clients_targeted": len(target_clients),
                "evaluated": False,
            })
            return

        acc, test_loss = self.evaluate_model(self.server_weights)

        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": len(target_clients),
            "total_clients_targeted": len(target_clients),
            "evaluated": True,
        }

        has_pers = getattr(self.config.clients, "use_ensemble", False) or (
            getattr(self.config.clients, "personalization_method", "none")
            in ("ditto", "apfl", "fedala", "fedrep", "fedper", "fedbabu")
        )
        if has_pers:
            ens_acc, ens_loss = self.evaluate_ensemble()
        else:
            ens_acc, ens_loss = self._evaluate_global_on_client_partitions()

        round_data["ensemble_test_accuracy"] = ens_acc
        round_data["ensemble_test_loss"] = ens_loss

        if round_num == self.config.num_rounds and getattr(self, "_last_per_client_accuracy", None):
            accs = sorted(self._last_per_client_accuracy.values())
            k = max(1, int(np.ceil(0.1 * len(accs))))
            round_data["bottom10_fairness"] = float(np.mean(accs[:k]))
            round_data["per_client_accuracy"] = {
                str(cid): round(a, 4) for cid, a in sorted(self._last_per_client_accuracy.items())
            }

        self.metrics.log_round(round_data)
