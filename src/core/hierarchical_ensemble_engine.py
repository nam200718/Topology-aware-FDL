import numpy as np
import torch
import torch.nn.functional as F

from src.core.aggregator import DeltaSpaceRobustAggregator, compute_buffer_slices
from src.core.base_engine import BaseEngine
from src.core.interfaces import ClientState
from src.core.model import vector_to_model
from src.data.dataset import get_fast_dataloader


def _prediction_entropy(logits: torch.Tensor) -> float:
    """Mean Shannon entropy of softmax predictions, in nats."""
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-8)).sum(dim=1).mean().item()


def compute_head_weights(weighting_mode, logits_root, logits_parent, logits_local,
                         head_losses, ensemble_alpha, loss_weight_beta, temperature,
                         static_weights):
    """Loss-calibrated dynamic 3-tier ensemble weighting (single source of truth).

    Replaces the four previously duplicated weighting blocks in evaluate_ensemble.
    Returns (w_local, w_parent, w_root).
    """
    if weighting_mode == "static":
        return static_weights

    l_local = head_losses.get("local", 0.0)
    l_parent = head_losses.get("parent", 0.0)
    l_root = head_losses.get("root", 0.0)

    if weighting_mode == "dynamic_confidence":
        h_root = _prediction_entropy(logits_root)
        h_parent = _prediction_entropy(logits_parent)
        h_local = _prediction_entropy(logits_local)
        score_local = -h_local - loss_weight_beta * l_local
        score_parent = -h_parent - loss_weight_beta * l_parent
        score_root = -h_root - loss_weight_beta * l_root
    elif weighting_mode == "dynamic_loss":
        score_local = -l_local
        score_parent = -l_parent
        score_root = -l_root
    else:
        raise ValueError(f"Unknown ensemble_weighting_mode '{weighting_mode}'")

    if ensemble_alpha is not None and len(ensemble_alpha) == 3:
        score_local += np.log(max(1e-4, ensemble_alpha[0]))
        score_parent += np.log(max(1e-4, ensemble_alpha[1]))
        score_root += np.log(max(1e-4, ensemble_alpha[2]))

    weights = F.softmax(torch.tensor([score_local, score_parent, score_root],
                                     dtype=torch.float32) / temperature, dim=0)
    return weights[0].item(), weights[1].item(), weights[2].item()


class HierarchicalEnsembleEngine(BaseEngine):
    """
    Engine for Hierarchical Ensemble Federated Learning.
    Supports Dual Aggregation, Dynamic Weighting (entropy & loss-calibrated),
    Shared Backbone Multi-Head compute optimization, and Privacy-Preserving Adaptive Update-Similarity Clustering.
    """
    def __init__(self, config, topology, aggregator, device="cpu"):
        super().__init__(config, topology, aggregator, device)

        # Initialize cluster heads
        self.cluster_heads_state = {}
        for hid in self.topology.get_server_connected_clients():
            head_state = ClientState(client_id=hid, initial_weights=self.server_weights.clone())
            self.cluster_heads_state[hid] = head_state

        # Adaptive Update-Similarity Clustering Params
        topo_params = config.topology.params
        self.cluster_method = topo_params.get("cluster_method", None)
        if self.cluster_method is None:
            self.cluster_method = "label_aware" if topo_params.get("cluster_by_label_dist", False) else "random"

        self.warmup_min_rounds = topo_params.get("warmup_min_rounds", 2)
        self.warmup_max_rounds = topo_params.get("warmup_max_rounds", 10)
        self.stability_threshold = topo_params.get("stability_threshold", 0.30)
        self.misalignment_threshold = topo_params.get("misalignment_threshold", 0.15)
        # Fraction of misaligned clients that triggers re-clustering (was inline literal).
        self.misaligned_ratio_trigger = topo_params.get("misaligned_ratio_trigger", 0.15)
        # Intra-cluster cosine similarity above which cluster heads sync to the
        # global model instead of aggregating (prevents IID fragmentation).
        self.cluster_sync_sim_threshold = topo_params.get("cluster_sync_sim_threshold", 0.5)

        # Staleness tracking for S-AFR (partial-participation fairness routing)
        self.last_sampled_round = {}
        self.cluster_head_velocity = {}
        # Per-client directional affinity to its assigned vs. nearest-alternate
        # cluster centroid: cid -> (own_similarity, best_other_similarity)
        self.client_cluster_affinities = {}

        # Server-side robust aggregation over update deltas (zero client cost).
        # "fedavg" preserves legacy behavior exactly.
        agg_mode = getattr(self.config.clients, "robust_aggregation_mode", "fedavg")
        self.robust_aggregator = None
        if agg_mode != "fedavg":
            self.robust_aggregator = DeltaSpaceRobustAggregator(
                mode=agg_mode,
                beta=getattr(self.config.clients, "trimmed_mean_beta", 0.20),
                temperature=getattr(self.config.clients, "soft_cosine_temperature", 0.5),
                norm_bound_k=getattr(self.config.clients, "norm_bound_k", 3.0),
                buffer_slices=compute_buffer_slices(self.updater.multihead_model),
            )

        self.has_initial_clustered = False
        self.prev_update_deltas = {}
        self._last_per_client_accuracy = {}

    def _aggregate_states_robustly(self, states, reference):
        """Aggregate client states; robust modes operate on deltas vs. the
        round-start reference so directional filtering is well-defined."""
        if self.robust_aggregator is None or len(states) <= 1:
            return self.aggregator.aggregate(states)
        deltas = [s.weights - reference for s in states]
        return reference + self.robust_aggregator.aggregate_deltas(deltas)

    def run_round(self, round_num: int):
        num_total = self.config.clients.num_clients
        all_clients = list(range(num_total))

        # Optional partial participation (legacy default: everyone participates)
        fraction = getattr(self.config.clients, "participation_fraction", 1.0)
        if fraction < 1.0:
            k = max(1, int(round(fraction * num_total)))
            participants = sorted(self.rng.choice(num_total, size=k, replace=False).tolist())
        else:
            participants = all_clients

        # 1. Prepare for distribution
        cluster_updates_parent = {hid: [] for hid in self.cluster_heads_state.keys()}
        cluster_updates_root = {hid: [] for hid in self.cluster_heads_state.keys()}

        # Round-start references for delta-space robust aggregation
        root_reference = self.server_weights.clone()
        parent_references = {
            hid: self.cluster_heads_state[hid].weights.clone()
            for hid in self.cluster_heads_state.keys()
        }

        weights_before = {}
        for client_id in participants:
            # Client receives Root model (global server)
            self.clients_state[client_id].weights = self.server_weights.clone()

            # Client receives Parent model (cluster head)
            head_id = self.topology.get_neighbors(client_id)[0]
            self.clients_state[client_id].parent_weights = self.cluster_heads_state[head_id].weights.clone()

            weights_before[client_id] = self.server_weights.clone()

        # 2. Local Updates
        current_lr = self.get_current_lr(round_num)
        current_deltas = {}
        for client_id in participants:
            state = self.clients_state[client_id].copy()
            client_ds = self.client_train_datasets[client_id]

            # Calls PyTorchLocalUpdater.update which handles 3 models / multi-head
            updated_state = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng,
                current_lr=current_lr
            )
            self.clients_state[client_id] = updated_state

            # Compute client model update vector (delta)
            delta = (updated_state.weights - weights_before[client_id]).detach()
            current_deltas[client_id] = delta
            self.last_sampled_round[client_id] = round_num

        # Track directional cluster affinity for Top-2 head routing at eval time
        if self.cluster_method == "update_similarity":
            self.client_cluster_affinities.update(
                self._client_cluster_affinities(participants, current_deltas))

        # --- Adaptive Update-Similarity Clustering Controller ---
        if self.cluster_method == "update_similarity" and hasattr(self.topology, "build_update_similarity"):
            should_recluster = False
            is_initial_trigger = False

            if not self.has_initial_clustered:
                # 1. Adaptive Warm-Up Trigger (Directional Stability)
                if round_num >= self.warmup_min_rounds:
                    if self.prev_update_deltas:
                        stabilities = []
                        for cid in participants:
                            if cid in self.prev_update_deltas:
                                d_prev = self.prev_update_deltas[cid]
                                d_curr = current_deltas[cid]
                                n_p = torch.norm(d_prev)
                                n_c = torch.norm(d_curr)
                                if n_p > 1e-8 and n_c > 1e-8:
                                    sim = (torch.dot(d_prev, d_curr) / (n_p * n_c)).item()
                                    stabilities.append(sim)
                        avg_stability = float(np.mean(stabilities)) if stabilities else 0.0
                    else:
                        avg_stability = 0.0

                    if avg_stability >= self.stability_threshold or round_num >= self.warmup_max_rounds:
                        should_recluster = True
                        is_initial_trigger = True
            else:
                # 2. Adaptive Re-Clustering Trigger (Client Misalignment Detection)
                misaligned_count = self._count_misaligned_clients(participants, current_deltas)
                misaligned_ratio = misaligned_count / len(participants)
                if misaligned_ratio > self.misaligned_ratio_trigger:
                    should_recluster = True

            if should_recluster:
                self.topology.build_update_similarity(
                    num_clients=num_total,
                    client_update_vectors=current_deltas,
                    seed=self.config.env.seed + round_num
                )

                if is_initial_trigger:
                    print(f"[Round {round_num}] Triggered Initial Update-Similarity Clustering. Resetting cluster heads to global model.")
                    for hid in self.cluster_heads_state.keys():
                        self.cluster_heads_state[hid].weights = self.server_weights.clone()
                    for cid in participants:
                        self.clients_state[cid].parent_head_state = None
                        self.clients_state[cid].local_head_state = None
                    self.has_initial_clustered = True
                else:
                    print(f"[Round {round_num}] Triggered Adaptive Re-Clustering due to client misalignment (>{100*self.misaligned_ratio_trigger:.0f}%). Carrying over head weights.")

            self.prev_update_deltas.update(current_deltas)

        # Re-pack contributions for parent and root aggregation
        for client_id in participants:
            updated_state = self.clients_state[client_id]
            head_id = self.topology.get_neighbors(client_id)[0]

            s_parent = updated_state.copy()
            if updated_state.parent_weights is not None:
                s_parent.weights = updated_state.parent_weights
            cluster_updates_parent[head_id].append(s_parent)
            cluster_updates_root[head_id].append(updated_state)

        # Check inter-client update similarity for topology cluster gating (O(N*K) centroid-based similarity)
        avg_pairwise_sim = self._avg_intra_cluster_similarity(participants, current_deltas)

        # 3. Intra-cluster Aggregation (Heads aggregate parent models)
        for hid, states in cluster_updates_parent.items():
            if states:
                if avg_pairwise_sim > self.cluster_sync_sim_threshold:
                    # Low non-IID / IID regime: Cluster heads synchronize with global server
                    # to prevent artificial sub-cluster fragmentation
                    new_weights = self.server_weights.clone()
                else:
                    new_weights = self._aggregate_states_robustly(states, parent_references[hid])
                self._apply_cluster_momentum(hid, new_weights)

        # 4. Global Aggregation (Server aggregates root models)
        all_root_contributions = []
        for hid, states in cluster_updates_root.items():
            all_root_contributions.extend(states)

        if all_root_contributions:
            expected = len(root_reference)
            bad = [s.client_id for s in all_root_contributions if len(s.weights) != expected]
            if bad:
                print(f"[warn] dropping client updates with unexpected sizes: {bad}")
                all_root_contributions = [s for s in all_root_contributions if len(s.weights) == expected]
            if all_root_contributions:
                self.server_weights = self._aggregate_states_robustly(all_root_contributions, root_reference)

        # Metrics (skipped rounds log a lightweight row; final round always evaluated)
        if not self.should_evaluate(round_num):
            self.metrics.log_round({
                "round": round_num,
                "participating_clients": len(participants),
                "total_clients_targeted": num_total,
                "evaluated": False,
            })
            return

        acc, test_loss = self.evaluate_model(self.server_weights)

        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": len(participants),
            "total_clients_targeted": num_total,
            "evaluated": True,
        }

        if self.config.clients.use_ensemble or self.config.clients.hierarchical_ensemble:
            ens_acc, ens_loss = self.evaluate_ensemble(round_num)
            round_data["ensemble_test_accuracy"] = ens_acc
            round_data["ensemble_test_loss"] = ens_loss
            # Final-round fairness artifact: worst-decile + raw per-client dump
            if round_num == self.config.num_rounds and self._last_per_client_accuracy:
                accs = sorted(self._last_per_client_accuracy.values())
                k = max(1, int(np.ceil(0.1 * len(accs))))
                round_data["bottom10_fairness"] = float(np.mean(accs[:k]))
                round_data["per_client_accuracy"] = {
                    str(cid): round(acc, 4)
                    for cid, acc in sorted(self._last_per_client_accuracy.items())
                }

        self.metrics.log_round(round_data)

    # ------------------------------------------------------------------
    # Clustering controller helpers
    # ------------------------------------------------------------------

    def _client_cluster_affinities(self, clients, current_deltas):
        """Directional affinity of each client to its own cluster centroid vs.
        the nearest alternate centroid: cid -> (own_sim, best_other_sim)."""
        cluster_centroids = {}
        for hid in self.cluster_heads_state.keys():
            member_cids = [cid for cid in clients if self.topology.get_neighbors(cid)[0] == hid]
            if member_cids:
                member_deltas = torch.stack([current_deltas[cid] for cid in member_cids])
                cluster_centroids[hid] = torch.mean(member_deltas, dim=0)

        affinities = {}
        if len(cluster_centroids) <= 1:
            return affinities

        for cid in clients:
            curr_head = self.topology.get_neighbors(cid)[0]
            d_cid = current_deltas[cid]
            norm_cid = torch.norm(d_cid)
            if norm_cid < 1e-8:
                continue

            sims = {}
            for hid, cent in cluster_centroids.items():
                norm_cent = torch.norm(cent)
                if norm_cent > 1e-8:
                    sims[hid] = (torch.dot(d_cid, cent) / (norm_cid * norm_cent)).item()

            if curr_head in sims:
                best_other_sim = max([s for h, s in sims.items() if h != curr_head], default=-1.0)
                affinities[cid] = (sims[curr_head], best_other_sim)
        return affinities

    def _count_misaligned_clients(self, all_clients, current_deltas) -> int:
        """Count clients whose update direction matches another cluster centroid
        better than their own by more than misalignment_threshold."""
        affinities = self._client_cluster_affinities(all_clients, current_deltas)
        return sum(
            1 for own_sim, best_other_sim in affinities.values()
            if best_other_sim - own_sim > self.misalignment_threshold
        )

    def _avg_intra_cluster_similarity(self, all_clients, current_deltas) -> float:
        """O(N*K) mean cosine similarity of members to their cluster centroid."""
        cluster_sims = []
        for hid in self.cluster_heads_state.keys():
            member_cids = [cid for cid in all_clients if self.topology.get_neighbors(cid)[0] == hid and cid in current_deltas]
            if len(member_cids) > 1:
                member_deltas = torch.stack([current_deltas[cid] for cid in member_cids])
                centroid = torch.mean(member_deltas, dim=0)
                norm_cent = torch.norm(centroid)
                if norm_cent > 1e-8:
                    for cid in member_cids:
                        d_c = current_deltas[cid]
                        n_c = torch.norm(d_c)
                        if n_c > 1e-8:
                            cluster_sims.append((torch.dot(d_c, centroid) / (n_c * norm_cent)).item())
        return float(np.mean(cluster_sims)) if cluster_sims else 0.0

    # ------------------------------------------------------------------
    # Cluster-head server-side momentum
    # ------------------------------------------------------------------

    def _apply_cluster_momentum(self, hid, new_weights):
        """w_new = beta * w_old + (1 - beta) * candidate. beta=0 -> plain replacement."""
        beta = getattr(self.config.clients, "cluster_momentum_beta", 0.0)
        old = self.cluster_heads_state[hid].weights
        if beta > 0.0 and old is not None and old.shape == new_weights.shape:
            self.cluster_heads_state[hid].weights = beta * old + (1.0 - beta) * new_weights
        else:
            self.cluster_heads_state[hid].weights = new_weights

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _apply_top2_routing(self, weights, client_id):
        """Gated Top-2 head routing (learnings doc §3 Rec. 2).

        Suppresses the drag-inducing third head when cluster affinity is clear:
        strong own-centroid similarity -> trust Root+Parent (drop Local noise);
        weak similarity -> fall back to Root+Local (drop cluster regularization).
        """
        cfg = self.config.clients
        if not getattr(cfg, "top2_routing", False):
            return weights
        aff = self.client_cluster_affinities.get(client_id)
        if aff is None:
            return weights

        own_sim, _ = aff
        w_local, w_parent, w_root = weights
        if own_sim >= self.cluster_sync_sim_threshold:
            w_local = 0.0
        else:
            w_parent = 0.0
        total = w_local + w_parent + w_root
        if total <= 1e-12:
            return (0.0, 0.0, 1.0)
        return (w_local / total, w_parent / total, w_root / total)

    def evaluate_ensemble(self, round_num: int = None):
        """
        Evaluate an ensemble of Root, Parent, and Local models.
        Supports Loss-Calibrated Dynamic Weighting incorporating prediction entropy and training loss.
        """
        clients_cfg = self.config.clients
        weighting_mode = clients_cfg.ensemble_weighting_mode
        compute_mode = clients_cfg.compute_optimization_mode
        loss_beta = clients_cfg.loss_weight_beta
        tau = clients_cfg.ensemble_temperature
        static_weights = (
            clients_cfg.ensemble_alpha,
            clients_cfg.ensemble_beta,
            max(0.0, 1.0 - clients_cfg.ensemble_alpha - clients_cfg.ensemble_beta),
        )
        temps = (clients_cfg.eval_temp_local, clients_cfg.eval_temp_parent, clients_cfg.eval_temp_root)
        iid_threshold = clients_cfg.iid_route_threshold

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        per_client_stats = {}
        self._last_per_client_accuracy = {}

        if compute_mode == "shared_backbone":
            multi_model = self.updater.multihead_model
            multi_model.eval()

            for client_id in range(self.config.clients.num_clients):
                state = self.clients_state[client_id]
                if getattr(state, "is_byzantine", False):
                    continue

                # Load current multihead parameters
                vector_to_model(state.weights.to(self.device), multi_model)
                if state.parent_head_state is not None:
                    multi_model.fc2_parent.load_state_dict(state.parent_head_state)
                if state.local_head_state is not None:
                    multi_model.fc2_local.load_state_dict(state.local_head_state)

                client_test_ds = self.client_test_datasets[client_id]
                if len(client_test_ds) == 0:
                    continue

                blend_weights = self._safr_blend_weights(state, client_id, round_num, iid_threshold)

                c_correct, c_total = 0, 0
                test_loader = get_fast_dataloader(client_test_ds, batch_size=min(len(client_test_ds), 1024), shuffle=False)
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        logits_root, logits_parent, logits_local = multi_model(images, head="all")

                        if blend_weights is None:
                            dynamic = compute_head_weights(
                                weighting_mode, logits_root, logits_parent, logits_local,
                                state.head_losses, getattr(state, "ensemble_alpha", None),
                                loss_beta, tau, static_weights)
                            blend_weights = self._apply_top2_routing(dynamic, client_id)
                        w_local, w_parent, w_root = blend_weights

                        logits_ensemble = (
                            w_local * (logits_local / temps[0])
                            + w_parent * (logits_parent / temps[1])
                            + w_root * (logits_root / temps[2])
                        )
                        loss = criterion(logits_ensemble, labels)
                        total_loss += loss.item()
                        predicted = logits_ensemble.argmax(dim=1)
                        total_samples += labels.size(0)
                        hits = (predicted == labels).sum().item()
                        total_correct += hits
                        c_correct += hits
                        c_total += labels.size(0)
                if c_total > 0:
                    per_client_stats[client_id] = (c_correct / c_total) * 100.0
        else:
            root_model = self.updater.global_model
            parent_model = self.updater.parent_model
            local_model = self.updater.local_model

            root_model.eval()
            parent_model.eval()
            local_model.eval()

            for client_id in range(self.config.clients.num_clients):
                state = self.clients_state[client_id]
                if getattr(state, "is_byzantine", False):
                    continue

                vector_to_model(self.server_weights.to(self.device), root_model)
                head_id = self.topology.get_neighbors(client_id)[0]
                agg_parent_weights = self.cluster_heads_state[head_id].weights

                if state.parent_weights is not None:
                    vector_to_model(agg_parent_weights.to(self.device), parent_model)
                else:
                    vector_to_model(self.server_weights.to(self.device), parent_model)

                if state.local_weights is not None:
                    vector_to_model(state.local_weights.to(self.device), local_model)
                else:
                    vector_to_model(state.weights.to(self.device), local_model)

                client_test_ds = self.client_test_datasets[client_id]
                if len(client_test_ds) == 0:
                    continue

                c_correct, c_total = 0, 0
                test_loader = get_fast_dataloader(client_test_ds, batch_size=min(len(client_test_ds), 1024), shuffle=False)
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        logits_root = root_model(images)
                        logits_parent = parent_model(images)
                        logits_local = local_model(images)

                        dynamic = compute_head_weights(
                            weighting_mode, logits_root, logits_parent, logits_local,
                            state.head_losses, getattr(state, "ensemble_alpha", None),
                            loss_beta, tau, static_weights)
                        w_local, w_parent, w_root = self._apply_top2_routing(dynamic, client_id)

                        logits_ensemble = w_local * logits_local + w_parent * logits_parent + w_root * logits_root
                        loss = criterion(logits_ensemble, labels)
                        total_loss += loss.item()
                        predicted = logits_ensemble.argmax(dim=1)
                        total_samples += labels.size(0)
                        hits = (predicted == labels).sum().item()
                        total_correct += hits
                        c_correct += hits
                        c_total += labels.size(0)
                if c_total > 0:
                    per_client_stats[client_id] = (c_correct / c_total) * 100.0

        if total_samples == 0:
            return 0.0, 0.0

        self._last_per_client_accuracy = per_client_stats
        avg_acc = (total_correct / total_samples) * 100.0
        avg_loss = total_loss / total_samples
        return avg_acc, avg_loss

    # ------------------------------------------------------------------
    # Staleness-Aware Fallback Routing (S-AFR)
    # ------------------------------------------------------------------

    def _safr_blend_weights(self, state, client_id, round_num, iid_threshold):
        """Return fixed blend weights when S-AFR or hard-IID routing applies,
        else None (per-batch dynamic weighting)."""
        clients_cfg = self.config.clients
        client_r_skew = getattr(state, "r_skew", 0.5)

        # Uniform/IID regime: complete Root head routing (closes IID generalization gap)
        if client_r_skew >= iid_threshold:
            return (0.0, 0.0, 1.0)

        if not clients_cfg.s_afr_enabled or round_num is None:
            return None

        last_sampled = self.last_sampled_round.get(client_id, None)
        if last_sampled is None:
            return None

        staleness = round_num - last_sampled
        if staleness <= clients_cfg.s_afr_staleness_window:
            return None

        # Fade the personalized heads toward the root head as staleness grows.
        fade = float(np.exp(-staleness / clients_cfg.s_afr_fade_tau))
        prior = getattr(state, "ensemble_alpha", None)
        if prior is not None and len(prior) == 3:
            w_local, w_parent = prior[0] * fade, prior[1] * fade
        else:
            w_local = clients_cfg.ensemble_alpha * fade
            w_parent = clients_cfg.ensemble_beta * fade
        w_root = max(0.0, 1.0 - w_local - w_parent)
        return (w_local, w_parent, w_root)
