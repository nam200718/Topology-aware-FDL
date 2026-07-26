import numpy as np
import torch
import torch.nn.functional as F
from src.core.hierarchical_engine import HierarchicalEngine
from src.core.interfaces import ClientState
from src.core.model import SimpleCNN
from src.data.dataset import ClientDataset
from torch.utils.data import DataLoader

class HierarchicalEnsembleEngine(HierarchicalEngine):
    """
    Engine for Hierarchical Ensemble Federated Learning.
    Supports Dual Aggregation, Dynamic Weighting (entropy & loss-calibrated),
    Shared Backbone Multi-Head compute optimization, and Label-Distribution Aware Clustering.
    """
    def __init__(self, config, topology, aggregator, device="cpu"):
        cluster_by_label = config.topology.params.get("cluster_by_label_dist", True)
        if cluster_by_label and hasattr(topology, "build_distribution_aware"):
            pass
        super().__init__(config, topology, aggregator, device)

    def run_round(self, round_num: int):
        all_clients = list(range(self.config.clients.num_clients))
        
        # 1. Prepare for distribution
        cluster_updates_parent = {hid: [] for hid in self.cluster_heads_state.keys()}
        cluster_updates_root = {hid: [] for hid in self.cluster_heads_state.keys()}
        
        for client_id in all_clients:
            # Client receives Root model (global server)
            self.clients_state[client_id].weights = self.server_weights.clone()
            
            # Client receives Parent model (cluster head)
            head_id = self.topology.get_neighbors(client_id)[0]
            self.clients_state[client_id].parent_weights = self.cluster_heads_state[head_id].weights.clone()
            
        # 2. Local Updates
        for client_id in all_clients:
            state = self.clients_state[client_id].copy()
            client_ds = ClientDataset(self.train_dataset, self.client_indices[client_id])
            
            # Calls PyTorchLocalUpdater.update which handles 3 models / multi-head
            updated_state = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng
            )
            self.clients_state[client_id] = updated_state
            
            head_id = self.topology.get_neighbors(client_id)[0]
            
            # Prepare state for parent aggregation (head level)
            s_parent = updated_state.copy()
            if updated_state.parent_weights is not None:
                s_parent.weights = updated_state.parent_weights
            cluster_updates_parent[head_id].append(s_parent)
            
            # Prepare state for root aggregation (server level)
            cluster_updates_root[head_id].append(updated_state)
            
        # 3. Intra-cluster Aggregation (Heads aggregate parent models)
        for hid, states in cluster_updates_parent.items():
            if states:
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
                torch.nn.utils.vector_to_parameters(state.weights.to(self.device), multi_model.parameters())

                client_test_ds = ClientDataset(self.test_dataset, self.client_test_indices[client_id])
                if len(client_test_ds) == 0:
                    continue

                test_loader = DataLoader(client_test_ds, batch_size=len(client_test_ds), shuffle=False)
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        logits_root, logits_parent, logits_local = multi_model(images, head="all")

                        if weighting_mode == "dynamic_confidence":
                            # Compute prediction entropy H(p)
                            p_root = F.softmax(logits_root, dim=1)
                            p_parent = F.softmax(logits_parent, dim=1)
                            p_local = F.softmax(logits_local, dim=1)

                            h_root = -(p_root * torch.log(p_root + 1e-8)).sum(dim=1).mean().item()
                            h_parent = -(p_parent * torch.log(p_parent + 1e-8)).sum(dim=1).mean().item()
                            h_local = -(p_local * torch.log(p_local + 1e-8)).sum(dim=1).mean().item()

                            # Loss-calibrated weighting: score = -entropy - beta * training_loss
                            head_losses = getattr(state, "head_losses", {})
                            l_root = head_losses.get("root", 0.0)
                            l_parent = head_losses.get("parent", 0.0)
                            l_local = head_losses.get("local", 0.0)

                            score_local = -h_local - beta * l_local
                            score_parent = -h_parent - beta * l_parent
                            score_root = -h_root - beta * l_root

                            weights = F.softmax(torch.tensor([score_local, score_parent, score_root], device=self.device), dim=0)
                            w_local, w_parent, w_root = weights[0].item(), weights[1].item(), weights[2].item()
                        elif weighting_mode == "dynamic_loss":
                            l_root = F.cross_entropy(logits_root, labels).item()
                            l_parent = F.cross_entropy(logits_parent, labels).item()
                            l_local = F.cross_entropy(logits_local, labels).item()

                            weights = F.softmax(torch.tensor([-l_local, -l_parent, -l_root], device=self.device), dim=0)
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

                torch.nn.utils.vector_to_parameters(self.server_weights.to(self.device), root_model.parameters())
                head_id = self.topology.get_neighbors(client_id)[0]
                agg_parent_weights = self.cluster_heads_state[head_id].weights

                if state.parent_weights is not None:
                    torch.nn.utils.vector_to_parameters(agg_parent_weights.to(self.device), parent_model.parameters())
                else:
                    torch.nn.utils.vector_to_parameters(self.server_weights.to(self.device), parent_model.parameters())

                if state.local_weights is not None:
                    torch.nn.utils.vector_to_parameters(state.local_weights.to(self.device), local_model.parameters())
                else:
                    torch.nn.utils.vector_to_parameters(state.weights.to(self.device), local_model.parameters())

                client_test_ds = ClientDataset(self.test_dataset, self.client_test_indices[client_id])
                if len(client_test_ds) == 0:
                    continue

                test_loader = DataLoader(client_test_ds, batch_size=len(client_test_ds), shuffle=False)
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        logits_root = root_model(images)
                        logits_parent = parent_model(images)
                        logits_local = local_model(images)

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

                            weights = F.softmax(torch.tensor([score_local, score_parent, score_root], device=self.device), dim=0)
                            w_local, w_parent, w_root = weights[0].item(), weights[1].item(), weights[2].item()
                        elif weighting_mode == "dynamic_loss":
                            l_root = F.cross_entropy(logits_root, labels).item()
                            l_parent = F.cross_entropy(logits_parent, labels).item()
                            l_local = F.cross_entropy(logits_local, labels).item()

                            weights = F.softmax(torch.tensor([-l_local, -l_parent, -l_root], device=self.device), dim=0)
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
