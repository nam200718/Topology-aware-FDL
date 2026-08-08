import numpy as np
from src.core.base_engine import BaseEngine
from src.core.interfaces import ClientState

class CentralizedEngine(BaseEngine):
    def __init__(self, config, topology, aggregator, device="cpu"):
        super().__init__(config, topology, aggregator, device)
        # Server specific state - start from client 0's weights
        self.server_weights = self.clients_state[0].weights.clone()
        
    def run_round(self, round_num: int):
        target_clients = self.topology.get_server_connected_clients()
        
        # Download
        for client_id in target_clients:
            self.clients_state[client_id].weights = self.server_weights.clone()
            
        # Update
        updated_states = []
        for client_id in target_clients:
            state = self.clients_state[client_id].copy()
            client_ds = self.client_train_datasets[client_id]
            
            updated_state = self.updater.update(
                state=state, 
                client_dataset=client_ds, 
                config=self.config.clients, 
                rng=self.local_rng
            )
            self.clients_state[client_id] = updated_state
            updated_states.append(updated_state)
            
        # Aggregate
        if updated_states:
            self.server_weights = self.aggregator.aggregate(updated_states)
            
        # Metrics
        acc, test_loss = self.evaluate_model(self.server_weights)
        
        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": len(target_clients),
            "total_clients_targeted": len(target_clients)
        }
        
        # Always log per-client personalized accuracy for fair comparison
        has_pers = getattr(self.config.clients, "use_ensemble", False) or (getattr(self.config.clients, "personalization_method", "none") in ("ditto", "apfl"))
        if has_pers:
            ens_acc, ens_loss = self.evaluate_ensemble()
        else:
            ens_acc, ens_loss = self._evaluate_global_on_client_partitions()

        round_data["ensemble_test_accuracy"] = ens_acc
        round_data["ensemble_test_loss"] = ens_loss
            
        self.metrics.log_round(round_data)

    def _evaluate_global_on_client_partitions(self):
        """
        Evaluate the global server model on each client's local test partition.
        Reports mean per-client accuracy, matching the metric format of Ditto/APFL/Ensemble.
        """
        import torch
        from src.data.dataset import ClientDataset, get_fast_dataloader
        from src.core.model import vector_to_model

        model = self.updater.global_model
        vector_to_model(self.server_weights.to(self.device), model)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        for client_id in range(self.config.clients.num_clients):
            client_test_ds = ClientDataset(self.test_dataset, self.client_test_indices[client_id])
            if len(client_test_ds) == 0:
                continue
            test_loader = get_fast_dataloader(client_test_ds, batch_size=len(client_test_ds), shuffle=False)

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    total_loss += loss.item()

                    predicted = logits.argmax(dim=1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()

        if total_samples == 0:
            return 0.0, 0.0

        acc = 100 * total_correct / total_samples
        avg_loss = total_loss / total_samples
        return acc, avg_loss

