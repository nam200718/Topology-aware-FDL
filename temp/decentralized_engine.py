import torch
from src.core.base_engine import BaseEngine
from src.data.dataset import ClientDataset

class DecentralizedEngine(BaseEngine):
    def __init__(self, config, topology, aggregator, device="cpu"):
        super().__init__(config, topology, aggregator, device)

    def run_round(self, round_num: int):
        all_clients = list(range(self.config.clients.num_clients))
        
        # 1. Exchange: client aggregate neighbors' weights
        buffers = {}
        for cid in all_clients:
            neighbors = self.topology.get_neighbors(cid)
            neighbor_states = [self.clients_state[n] for n in neighbors]
            neighbor_states.append(self.clients_state[cid])  # include self
            buffers[cid] = self.aggregator.aggregate(neighbor_states)
            
        # 2. Local Update on aggregated weights
        for cid in all_clients:
            state = self.clients_state[cid].copy()
            state.weights = buffers[cid]
            
            client_ds = ClientDataset(self.train_dataset, self.client_indices[cid])
            
            updated = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng
            )
            self.clients_state[cid] = updated
            
        # 3. Global metrics = avg all clients
        avg_weights = torch.mean(torch.stack([self.clients_state[c].weights for c in all_clients]), dim=0)
        acc, test_loss = self.evaluate_model(avg_weights)
        
        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": len(all_clients),
            "total_clients_targeted": len(all_clients)
        }
        
        self.metrics.log_round(round_data)
