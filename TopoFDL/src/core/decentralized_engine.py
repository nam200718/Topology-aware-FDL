import numpy as np
from src.core.base_engine import BaseEngine

class DecentralizedEngine(BaseEngine):
    def __init__(self, config, topology, aggregator, device="cpu"):
        super().__init__(config, topology, aggregator, device)

    def run_round(self, round_num: int):
        all_clients = list(range(self.config.clients.num_clients))
        
        # 1. Download/Exchange phase
        # Active clients fetch weights from their neighbors
        # To simulate synchronous step, we read from current memory and write to a buffer
        buffers = {}
        for client_id in all_clients:
            neighbors = self.topology.get_neighbors(client_id)
            neighbor_states = [self.clients_state[n] for n in neighbors]
            # Including own state for aggregation
            neighbor_states.append(self.clients_state[client_id])
            
            # Aggregate incoming weights + own weight
            agg_weights = self.aggregator.aggregate(neighbor_states)
            buffers[client_id] = agg_weights
            
        # 2. Local Update phase
        for client_id in all_clients:
            # apply aggregated neighbor weights first
            state = self.clients_state[client_id].copy()
            state.weights = buffers[client_id]
            
            from src.data.dataset import ClientDataset
            client_ds = ClientDataset(self.train_dataset, self.client_indices[client_id])
            
            # perform local update
            updated_state = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng
            )
            self.clients_state[client_id] = updated_state
            
        # 3. Metrics calculation
        import torch
        avg_weights = torch.mean(torch.stack([self.clients_state[c].weights for c in range(self.config.clients.num_clients)]), dim=0)
        acc, test_loss = self.evaluate_model(avg_weights)
        
        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": len(all_clients),
            "total_clients_targeted": len(all_clients)
        }
        self.metrics.log_round(round_data)
