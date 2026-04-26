import json
import os
import numpy as np
from typing import Dict, Any

from src.config import SimulationConfig
from src.core.interfaces import Topology, Aggregator, ClientState, MetricsCollector

from src.core.updater import VectorLocalUpdater
from src.utils.random import get_random_state

class SimulationEngine:
    def __init__(self, config: SimulationConfig, topology: Topology, aggregator: Aggregator):
        self.config = config
        self.topology = topology
        self.aggregator = aggregator
        
        # Build components
        self.rng = get_random_state()
        self.local_rng = get_random_state()
        
        self.topology.build(self.config.clients.num_clients, seed=self.config.env.seed)
        
        self.metrics = MetricsCollector()

        self.updater = VectorLocalUpdater()
        
        import torch
        
        # Initialize target optimum (e.g. zeros)
        self.global_target = np.zeros(self.config.clients.model_dim)
        
        # Initialize simulation state with torch tensor
        self.server_weights = torch.tensor(self.rng.normal(scale=1.0, size=self.config.clients.model_dim))
        
        # Initialize client states
        self.clients_state = {}
        for client_id in range(self.config.clients.num_clients):
            # All clients start with the server weights (torch tensor)
            self.clients_state[client_id] = ClientState(client_id, self.server_weights.clone())
            
    def run(self):
        print(f"Starting simulation: {self.config.experiment_name}")
        for round_num in range(1, self.config.num_rounds + 1):
            self.run_round(round_num)
            
        self.save_metrics()
        
    def run_round(self, round_num: int):
        # 1. Server selects clients (for Star topology: all connected clients)
        # Note: In real FedAvg we might sample a fraction, but let's assume full participation 
        # before failures for MVP.
        target_clients = self.topology.get_server_connected_clients()
        
        # 2. Apply failures (Skipped)
        
        # 3. Download weights (virtual) - server to client 
        # For a star topology, clients get the latest server weights
        for client_id in target_clients:
            self.clients_state[client_id].weights = self.server_weights.copy()
            
        # 4. Local Update
        updated_states = []
        for client_id in target_clients:
            state = self.clients_state[client_id].copy()
            # Perform approximate update
            updated_state = self.updater.update(
                state=state, 
                global_target=self.global_target, 
                config=self.config.clients, 
                rng=self.local_rng
            )
            # Update local memory
            self.clients_state[client_id] = updated_state
            updated_states.append(updated_state)
            
        # 5. Aggregate
        if updated_states:
            self.server_weights = self.aggregator.aggregate(updated_states)
            
        # 6. Compute metrics
        l2_distance = float(np.linalg.norm(self.server_weights - self.global_target))
        
        round_data = {
            "round": round_num,
            "convergence_l2_distance": l2_distance,
            "participating_clients": len(target_clients),
            "total_clients_targeted": len(target_clients)
        }
        self.metrics.log_round(round_data)
        
    def save_metrics(self):
        out_dir = self.config.ensure_output_dir()
        file_path = os.path.join(out_dir, "metrics.json")
        with open(file_path, "w") as f:
            json.dump(self.metrics.get_history(), f, indent=4)
        print(f"Metrics saved to {file_path}")
