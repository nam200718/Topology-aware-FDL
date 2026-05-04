import json
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

from src.config import SimulationConfig
from src.core.interfaces import Topology, Aggregator, ClientState, MetricsCollector

from src.utils.random import get_random_state
from src.data.dataset import get_mnist, partition_data_non_iid, ClientDataset
from src.core.model import SimpleCNN
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class BaseEngine(ABC):
    def __init__(self, config: SimulationConfig, topology: Topology, aggregator: Aggregator, device="cpu"):
        self.config = config
        self.topology = topology
        self.aggregator = aggregator
        self.device = torch.device(device)
        
        # Build components
        self.rng = get_random_state()
        self.local_rng = get_random_state()
        
        self.topology.build(self.config.clients.num_clients, seed=self.config.env.seed)
        
        self.metrics = MetricsCollector()
        
        # Fetch datasets
        print("Downloading and dividing MNIST dataset...")
        self.train_dataset, self.test_dataset = get_mnist()
        
        non_iid_enabled = getattr(self.config.non_iid, "enabled", True)
        num_shards = getattr(self.config.non_iid, "num_shards", 200)
        
        from src.data.dataset import partition_data
        self.client_indices = partition_data(
            self.train_dataset, 
            self.config.clients.num_clients, 
            non_iid=non_iid_enabled, 
            num_shards=num_shards, 
            seed=self.config.env.seed
        )
        self.client_test_indices = partition_data(
            self.test_dataset, 
            self.config.clients.num_clients, 
            non_iid=non_iid_enabled, 
            num_shards=num_shards, 
            seed=self.config.env.seed
        )
        
        # Fetch initial model vector
        dummy_model = SimpleCNN()
        initial_w = torch.nn.utils.parameters_to_vector(dummy_model.parameters()).detach()
        num_params = initial_w.numel()
        print(f"Model instantiated with {num_params} parameters.")
        
        from src.core.updater import PyTorchLocalUpdater
        self.updater = PyTorchLocalUpdater(device=device)
        
        # Track clients
        self.clients_state = {}
        for client_id in range(self.config.clients.num_clients):
            state = ClientState(client_id, initial_w.clone())
            if self.rng.rand() < getattr(self.config.robustness, "byzantine_rate", 0.0):
                state.is_byzantine = True
                state.byzantine_type = self.config.robustness.byzantine_type
            self.clients_state[client_id] = state
            
    def run(self):
        print(f"Starting simulation: {self.config.experiment_name} | Topo: {self.config.topology.type}")
        for round_num in range(1, self.config.num_rounds + 1):
            self.run_round(round_num)
            
        self.save_metrics()
        
    @abstractmethod
    def run_round(self, round_num: int):
        pass
        
    def evaluate_model(self, weights: torch.Tensor):
        model = SimpleCNN().to(self.device)
        torch.nn.utils.vector_to_parameters(weights, model.parameters())
        model.eval()
        
        test_loader = DataLoader(self.test_dataset, batch_size=1024, shuffle=False)
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        acc = 100 * correct / total
        avg_loss = total_loss / total
        return acc, avg_loss

    def evaluate_ensemble(self):
        use_ensemble = getattr(self.config.clients, "use_ensemble", False)
        if not use_ensemble:
            return 0.0, 0.0
            
        alpha = getattr(self.config.clients, "ensemble_alpha", 0.5)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        # Instantiate two models
        global_model = SimpleCNN().to(self.device)
        local_model = SimpleCNN().to(self.device)
        global_model.eval()
        local_model.eval()
        
        for client_id in range(self.config.clients.num_clients):
            state = self.clients_state[client_id]
            if getattr(state, "local_weights", None) is None:
                continue
                
            # Load weights
            torch.nn.utils.vector_to_parameters(state.weights, global_model.parameters())
            torch.nn.utils.vector_to_parameters(state.local_weights, local_model.parameters())
            
            # Client's local test data
            client_test_ds = ClientDataset(self.test_dataset, self.client_test_indices[client_id])
            if len(client_test_ds) == 0:
                continue
                
            test_loader = DataLoader(client_test_ds, batch_size=256, shuffle=False)
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    logits_global = global_model(images)
                    logits_local = local_model(images)
                    
                    # Ensemble logits
                    logits_ensemble = alpha * logits_local + (1.0 - alpha) * logits_global
                    
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

    def save_metrics(self):
        out_dir = self.config.ensure_output_dir()
        
        # Save JSON
        json_path = os.path.join(out_dir, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(self.metrics.get_history(), f, indent=4)
            
        # Save CSV
        try:
            import pandas as pd
            csv_path = os.path.join(out_dir, "metrics.csv")
            df = pd.DataFrame(self.metrics.get_history())
            df.to_csv(csv_path, index=False)
            print(f"Metrics saved to {json_path} and {csv_path}")
        except ImportError:
            print(f"Metrics saved to {json_path} (Skipped CSV, pandas not installed)")
