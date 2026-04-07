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
            from src.data.dataset import ClientDataset
            client_ds = ClientDataset(self.train_dataset, self.client_indices[client_id])
            
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
        self.metrics.log_round(round_data)
