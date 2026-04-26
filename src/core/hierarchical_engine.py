import numpy as np
from src.core.base_engine import BaseEngine
from src.core.interfaces import ClientState

class HierarchicalEngine(BaseEngine):
    def __init__(self, config, topology, aggregator, device="cpu"):
        super().__init__(config, topology, aggregator, device)
        # Global server - start from client 0's weights
        self.server_weights = self.clients_state[0].weights.clone()
        
        # Cluster heads
        self.num_clusters = config.topology.params.get("num_clusters", 5)
        self.cluster_heads_state = {}
        for cluster_id in range(self.num_clusters):
            # cluster IDs from -2 to -num_clusters-1 (just to keep them negative like server_id=-1)
            hid = -2 - cluster_id
            self.cluster_heads_state[hid] = ClientState(hid, self.server_weights.clone())
            
    def run_round(self, round_num: int):
        all_clients = list(range(self.config.clients.num_clients))
        
        # 1. Global Server to Cluster Heads
        # In a real system, Server sends to Heads. We'll simply sync them perfectly here
        for hid in self.cluster_heads_state.keys():
            self.cluster_heads_state[hid].weights = self.server_weights.clone()
            
        # 2. Cluster Heads to Clients (Download)
        cluster_updates = {hid: [] for hid in self.cluster_heads_state.keys()}
        
        for client_id in all_clients:
            # We expect get_neighbors to return the cluster head ID for a client
            neighbors = self.topology.get_neighbors(client_id)
            head_id = neighbors[0] # assuming exactly 1 head per client
            self.clients_state[client_id].weights = self.cluster_heads_state[head_id].weights.clone()
            
        # 3. Local Updates
        for client_id in all_clients:
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
            
            # send to cluster head mapping
            head_id = self.topology.get_neighbors(client_id)[0]
            cluster_updates[head_id].append(updated_state)
            
        # 4. Intra-cluster Aggregation (Heads aggregate client weights)
        head_states = []
        for hid, client_states in cluster_updates.items():
            if client_states:
                agg_weights = self.aggregator.aggregate(client_states)
                self.cluster_heads_state[hid].weights = agg_weights
            head_states.append(self.cluster_heads_state[hid])
            
        # 5. Global Aggregation (Server aggregates cluster head weights)
        if head_states:
            self.server_weights = self.aggregator.aggregate(head_states)
            
        # Metrics
        acc, test_loss = self.evaluate_model(self.server_weights)
        
        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": len(all_clients),
            "total_clients_targeted": len(all_clients)
        }
        
        if getattr(self.config.clients, "use_ensemble", False):
            ens_acc, ens_loss = self.evaluate_ensemble()
            round_data["ensemble_test_accuracy"] = ens_acc
            round_data["ensemble_test_loss"] = ens_loss
            
        self.metrics.log_round(round_data)
