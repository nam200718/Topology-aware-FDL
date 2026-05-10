import numpy as np
import torch
from src.core.hierarchical_engine import HierarchicalEngine
from src.core.interfaces import ClientState
from src.core.model import SimpleCNN
from src.data.dataset import ClientDataset
from torch.utils.data import DataLoader

class HierarchicalEnsembleEngine(HierarchicalEngine):
    """
    Engine for Hierarchical Ensemble Federated Learning.
    Each client trains three models:
    1. Root model (from global server)
    2. Parent model (from cluster head)
    3. Local model (persistent local model)
    
    During inference, it uses an ensemble of these three models.
    """
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
            
            # This calls PyTorchLocalUpdater.update which now handles 3 models
            updated_state = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng
            )
            self.clients_state[client_id] = updated_state
            
            head_id = self.topology.get_neighbors(client_id)[0]
            
            # Prepare state for parent aggregation (head level)
            # We need a ClientState where .weights is the trained parent model
            s_parent = updated_state.copy()
            s_parent.weights = updated_state.parent_weights
            cluster_updates_parent[head_id].append(s_parent)
            
            # Prepare state for root aggregation (server level)
            # updated_state.weights is already the trained root model
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
        Logits are combined as: alpha*Local + beta*Parent + (1-alpha-beta)*Root
        """
        alpha = getattr(self.config.clients, "ensemble_alpha", 0.33)
        beta = getattr(self.config.clients, "ensemble_beta", 0.33)
        gamma = max(0, 1.0 - alpha - beta)
        
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        # Reuse model objects from the engine's updater if possible, or create once
        root_model = self.updater.global_model
        parent_model = self.updater.parent_model
        local_model = self.updater.local_model
        
        root_model.eval()
        parent_model.eval()
        local_model.eval()
        
        for client_id in range(self.config.clients.num_clients):
            state = self.clients_state[client_id]
            
            # Load weights
            torch.nn.utils.vector_to_parameters(state.weights.to(self.device), root_model.parameters())
            
            if state.parent_weights is not None:
                torch.nn.utils.vector_to_parameters(state.parent_weights.to(self.device), parent_model.parameters())
            else:
                torch.nn.utils.vector_to_parameters(state.weights.to(self.device), parent_model.parameters())
                
            if state.local_weights is not None:
                torch.nn.utils.vector_to_parameters(state.local_weights.to(self.device), local_model.parameters())
            else:
                torch.nn.utils.vector_to_parameters(state.weights.to(self.device), local_model.parameters())
            
            client_test_ds = ClientDataset(self.test_dataset, self.client_test_indices[client_id])
            if len(client_test_ds) == 0:
                continue
                
            # If dataset is FastDataset, we don't really need a full DataLoader for small batches,
            # but it's consistent. For FastDataset, DataLoader is very fast.
            test_loader = DataLoader(client_test_ds, batch_size=len(client_test_ds), shuffle=False)
            
            with torch.no_grad():
                for images, labels in test_loader:
                    # images, labels are already on device if using FastDataset
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    logits_root = root_model(images)
                    logits_parent = parent_model(images)
                    logits_local = local_model(images)
                    
                    # Ensemble logits
                    logits_ensemble = alpha * logits_local + beta * logits_parent + gamma * logits_root
                    
                    loss = criterion(logits_ensemble, labels)
                    total_loss += loss.item()
                    
                    _, predicted = torch.max(logits_ensemble.data, 1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
                    
        if total_samples == 0:
            return 0.0, 0.0
            
        acc = 100 * total_correct / total_samples
        avg_loss = total_loss / total_samples
        return acc, avg_loss
