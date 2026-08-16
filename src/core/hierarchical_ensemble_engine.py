import numpy as np
import torch
import torch.nn.functional as F
from src.core.base_engine import BaseEngine
from src.core.interfaces import ClientState
from src.core.model import SimpleCNN, vector_to_model
from src.data.dataset import ClientDataset
from torch.utils.data import DataLoader

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
        
        self.has_initial_clustered = False
        self.prev_update_deltas = {}

    def run_round(self, round_num: int):
        all_clients = list(range(self.config.clients.num_clients))
        
        # 1. Prepare for distribution
        cluster_updates_parent = {hid: [] for hid in self.cluster_heads_state.keys()}
        cluster_updates_root = {hid: [] for hid in self.cluster_heads_state.keys()}
        
        weights_before = {}
        for client_id in all_clients:
            # Client receives Root model (global server)
            self.clients_state[client_id].weights = self.server_weights.clone()
            
            # Client receives Parent model (cluster head)
            head_id = self.topology.get_neighbors(client_id)[0]
            self.clients_state[client_id].parent_weights = self.cluster_heads_state[head_id].weights.clone()
            
            weights_before[client_id] = self.server_weights.clone()
            
        # 2. Local Updates
        current_lr = self.get_current_lr(round_num)
        current_deltas = {}
        for client_id in all_clients:
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

        # --- Adaptive Update-Similarity Clustering Controller ---
        if self.cluster_method == "update_similarity" and hasattr(self.topology, "build_update_similarity"):
            should_recluster = False
            is_initial_trigger = False

            if not self.has_initial_clustered:
                # 1. Adaptive Warm-Up Trigger (Directional Stability)
                if round_num >= self.warmup_min_rounds:
                    if self.prev_update_deltas:
                        stabilities = []
                        for cid in all_clients:
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
                cluster_centroids = {}
                for hid in self.cluster_heads_state.keys():
                    member_cids = [cid for cid in all_clients if self.topology.get_neighbors(cid)[0] == hid]
                    if member_cids:
                        member_deltas = torch.stack([current_deltas[cid] for cid in member_cids])
                        cluster_centroids[hid] = torch.mean(member_deltas, dim=0)

                if len(cluster_centroids) > 1:
                    misaligned_count = 0
                    for cid in all_clients:
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
                            assigned_sim = sims[curr_head]
                            best_other_sim = max([s for h, s in sims.items() if h != curr_head], default=-1.0)
                            if best_other_sim - assigned_sim > self.misalignment_threshold:
                                misaligned_count += 1

                    misaligned_ratio = misaligned_count / len(all_clients)
                    if misaligned_ratio > 0.15:
                        should_recluster = True

            if should_recluster:
                self.topology.build_update_similarity(
                    num_clients=len(all_clients),
                    client_update_vectors=current_deltas,
                    seed=self.config.env.seed + round_num
                )
                
                if is_initial_trigger:
                    print(f"[Round {round_num}] Triggered Initial Update-Similarity Clustering. Resetting cluster heads to global model.")
                    for hid in self.cluster_heads_state.keys():
                        self.cluster_heads_state[hid].weights = self.server_weights.clone()
                    for cid in all_clients:
                        self.clients_state[cid].parent_head_state = None
                        self.clients_state[cid].local_head_state = None
                    self.has_initial_clustered = True
                else:
                    print(f"[Round {round_num}] Triggered Adaptive Re-Clustering due to client misalignment (>15%). Carrying over head weights.")

            self.prev_update_deltas = current_deltas

        # Re-pack contributions for parent and root aggregation
        for client_id in all_clients:
            updated_state = self.clients_state[client_id]
            head_id = self.topology.get_neighbors(client_id)[0]
            
            s_parent = updated_state.copy()
            if updated_state.parent_weights is not None:
                s_parent.weights = updated_state.parent_weights
            cluster_updates_parent[head_id].append(s_parent)
            cluster_updates_root[head_id].append(updated_state)
            
        # Check inter-client update similarity for topology cluster gating (O(N*K) centroid-based similarity)
        cluster_sims = []
        for hid in cluster_updates_parent.keys():
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
        avg_pairwise_sim = float(np.mean(cluster_sims)) if cluster_sims else 0.0

        # 3. Intra-cluster Aggregation (Heads aggregate parent models)
        for hid, states in cluster_updates_parent.items():
            if states:
                if avg_pairwise_sim > 0.5:
                    # Low non-IID / IID regime: Cluster heads synchronize with global server
                    # to prevent artificial sub-cluster fragmentation
                    self.cluster_heads_state[hid].weights = self.server_weights.clone()
                else:
                    agg_weights_parent = self.aggregator.aggregate(states)
                    self.cluster_heads_state[hid].weights = agg_weights_parent
            
        # 4. Global Aggregation (Server aggregates root models)
        all_root_contributions = []
        for hid, states in cluster_updates_root.items():
            all_root_contributions.extend(states)
            
        if all_root_contributions:
            self.server_weights = self.aggregator.aggregate(all_root_contributions)
            
        # Metrics
        acc, test_loss = self.evaluate_model(self.server_weights)
        
        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": len(all_clients),
            "total_clients_targeted": len(all_clients)
        }
        
        if self.config.clients.use_ensemble or self.config.clients.hierarchical_ensemble:
            ens_acc, ens_loss = self.evaluate_ensemble()
            round_data["ensemble_test_accuracy"] = ens_acc
            round_data["ensemble_test_loss"] = ens_loss
            
        self.metrics.log_round(round_data)

    def evaluate_ensemble(self):
        """
        Evaluate an ensemble of Root, Parent, and Local models.
        Supports Loss-Calibrated Dynamic Weighting incorporating prediction entropy and training loss.
        """
        weighting_mode = getattr(self.config.clients, "ensemble_weighting_mode", "dynamic_confidence")
        compute_mode = getattr(self.config.clients, "compute_optimization_mode", "shared_backbone")
        beta = getattr(self.config.clients, "loss_weight_beta", 1.0)
        alpha_static = getattr(self.config.clients, "ensemble_alpha", 0.33)
        beta_static = getattr(self.config.clients, "ensemble_beta", 0.33)
        gamma_static = max(0.0, 1.0 - alpha_static - beta_static)

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

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

                from src.data.dataset import get_fast_dataloader
                test_loader = get_fast_dataloader(client_test_ds, batch_size=len(client_test_ds), shuffle=False)
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        logits_root, logits_parent, logits_local = multi_model(images, head="all")

                        tau = getattr(self.config.clients, "ensemble_temperature", 1.0)
                        if weighting_mode == "dynamic_confidence":
                            # Compute prediction entropy H(p)
                            p_root = F.softmax(logits_root, dim=1)
                            p_parent = F.softmax(logits_parent, dim=1)
                            p_local = F.softmax(logits_local, dim=1)

                            h_root = -(p_root * torch.log(p_root + 1e-8)).sum(dim=1).mean().item()
                            h_parent = -(p_parent * torch.log(p_parent + 1e-8)).sum(dim=1).mean().item()
                            h_local = -(p_local * torch.log(p_local + 1e-8)).sum(dim=1).mean().item()

                            # Loss-calibrated weighting with learned 3-tier simplex prior
                            head_losses = getattr(state, "head_losses", {})
                            l_root = head_losses.get("root", 0.0)
                            l_parent = head_losses.get("parent", 0.0)
                            l_local = head_losses.get("local", 0.0)

                            score_local = -h_local - beta * l_local
                            score_parent = -h_parent - beta * l_parent
                            score_root = -h_root - beta * l_root

                            alpha_vec = getattr(state, "ensemble_alpha", None)
                            if alpha_vec is not None and len(alpha_vec) == 3:
                                score_local += np.log(max(1e-4, alpha_vec[0]))
                                score_parent += np.log(max(1e-4, alpha_vec[1]))
                                score_root += np.log(max(1e-4, alpha_vec[2]))

                            weights = F.softmax(torch.tensor([score_local, score_parent, score_root], dtype=torch.float32) / tau, dim=0)
                            w_local, w_parent, w_root = weights[0].item(), weights[1].item(), weights[2].item()
                        elif weighting_mode == "dynamic_loss":
                            head_losses = getattr(state, "head_losses", {})
                            l_root = head_losses.get("root", 0.0)
                            l_parent = head_losses.get("parent", 0.0)
                            l_local = head_losses.get("local", 0.0)

                            score_local = -l_local
                            score_parent = -l_parent
                            score_root = -l_root

                            alpha_vec = getattr(state, "ensemble_alpha", None)
                            if alpha_vec is not None and len(alpha_vec) == 3:
                                score_local += np.log(max(1e-4, alpha_vec[0]))
                                score_parent += np.log(max(1e-4, alpha_vec[1]))
                                score_root += np.log(max(1e-4, alpha_vec[2]))

                            weights = F.softmax(torch.tensor([score_local, score_parent, score_root], dtype=torch.float32) / tau, dim=0)
                            w_local, w_parent, w_root = weights[0].item(), weights[1].item(), weights[2].item()
                        else: # static
                            w_local, w_parent, w_root = alpha_static, beta_static, gamma_static

                        logits_ensemble = w_local * logits_local + w_parent * logits_parent + w_root * logits_root
                        loss = criterion(logits_ensemble, labels)
                        total_loss += loss.item()
                        predicted = logits_ensemble.argmax(dim=1)
                        total_samples += labels.size(0)
                        total_correct += (predicted == labels).sum().item()
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

                from src.data.dataset import get_fast_dataloader
                test_loader = get_fast_dataloader(client_test_ds, batch_size=len(client_test_ds), shuffle=False)
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        logits_root = root_model(images)
                        logits_parent = parent_model(images)
                        logits_local = local_model(images)

                        tau = getattr(self.config.clients, "ensemble_temperature", 1.0)
                        if weighting_mode == "dynamic_confidence":
                            p_root = F.softmax(logits_root, dim=1)
                            p_parent = F.softmax(logits_parent, dim=1)
                            p_local = F.softmax(logits_local, dim=1)

                            h_root = -(p_root * torch.log(p_root + 1e-8)).sum(dim=1).mean().item()
                            h_parent = -(p_parent * torch.log(p_parent + 1e-8)).sum(dim=1).mean().item()
                            h_local = -(p_local * torch.log(p_local + 1e-8)).sum(dim=1).mean().item()

                            head_losses = getattr(state, "head_losses", {})
                            l_root = head_losses.get("root", 0.0)
                            l_parent = head_losses.get("parent", 0.0)
                            l_local = head_losses.get("local", 0.0)

                            score_local = -h_local - beta * l_local
                            score_parent = -h_parent - beta * l_parent
                            score_root = -h_root - beta * l_root

                            alpha_vec = getattr(state, "ensemble_alpha", None)
                            if alpha_vec is not None and len(alpha_vec) == 3:
                                score_local += np.log(max(1e-4, alpha_vec[0]))
                                score_parent += np.log(max(1e-4, alpha_vec[1]))
                                score_root += np.log(max(1e-4, alpha_vec[2]))

                            weights = F.softmax(torch.tensor([score_local, score_parent, score_root], dtype=torch.float32) / tau, dim=0)
                            w_local, w_parent, w_root = weights[0].item(), weights[1].item(), weights[2].item()
                        elif weighting_mode == "dynamic_loss":
                            head_losses = getattr(state, "head_losses", {})
                            l_root = head_losses.get("root", 0.0)
                            l_parent = head_losses.get("parent", 0.0)
                            l_local = head_losses.get("local", 0.0)

                            score_local = -l_local
                            score_parent = -l_parent
                            score_root = -l_root

                            alpha_vec = getattr(state, "ensemble_alpha", None)
                            if alpha_vec is not None and len(alpha_vec) == 3:
                                score_local += np.log(max(1e-4, alpha_vec[0]))
                                score_parent += np.log(max(1e-4, alpha_vec[1]))
                                score_root += np.log(max(1e-4, alpha_vec[2]))

                            weights = F.softmax(torch.tensor([score_local, score_parent, score_root], dtype=torch.float32) / tau, dim=0)
                            w_local, w_parent, w_root = weights[0].item(), weights[1].item(), weights[2].item()
                        else:
                            w_local, w_parent, w_root = alpha_static, beta_static, gamma_static

                        logits_ensemble = w_local * logits_local + w_parent * logits_parent + w_root * logits_root
                        loss = criterion(logits_ensemble, labels)
                        total_loss += loss.item()
                        predicted = logits_ensemble.argmax(dim=1)
                        total_samples += labels.size(0)
                        total_correct += (predicted == labels).sum().item()

        if total_samples == 0:
            return 0.0, 0.0

        avg_acc = (total_correct / total_samples) * 100.0
        avg_loss = total_loss / total_samples
        return avg_acc, avg_loss
