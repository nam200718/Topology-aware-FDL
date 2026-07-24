import json
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List

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
        dataset_name = getattr(self.config.env, "dataset", "mnist").lower()
        if dataset_name == "cifar10":
            print("Downloading and dividing CIFAR-10 dataset...")
            from src.data.dataset import get_cifar10
            train_ds, test_ds = get_cifar10(
                train_subset=getattr(self.config.env, "train_subset", None),
                test_subset=getattr(self.config.env, "test_subset", None)
            )
            self.in_channels = 3
        else:
            print("Downloading and dividing MNIST dataset...")
            train_ds, test_ds = get_mnist(
                train_subset=getattr(self.config.env, "train_subset", None),
                test_subset=getattr(self.config.env, "test_subset", None)
            )
            self.in_channels = 1

        self.train_dataset: torch.utils.data.Dataset = train_ds
        self.test_dataset: torch.utils.data.Dataset = test_ds

        from src.data.dataset import FastDataset
        self.train_dataset = FastDataset(self.train_dataset, self.device)
        self.test_dataset = FastDataset(self.test_dataset, self.device)
        
        non_iid_enabled = getattr(self.config.non_iid, "enabled", True)
        alpha = getattr(self.config.non_iid, "alpha", 0.5)
        
        from src.data.dataset import partition_data
        self.client_indices: Dict[int, List[int]] = partition_data(
            self.train_dataset, 
            self.config.clients.num_clients, 
            non_iid=non_iid_enabled, 
            alpha=alpha, 
            seed=self.config.env.seed
        )
        self.client_test_indices: Dict[int, List[int]] = partition_data(
            self.test_dataset, 
            self.config.clients.num_clients, 
            non_iid=non_iid_enabled, 
            alpha=alpha, 
            seed=self.config.env.seed
        )

        # Check if topology requests label-distribution aware clustering
        if self.config.topology.params.get("cluster_by_label_dist", False) and hasattr(self.topology, "build_distribution_aware"):
            from src.data.dataset import _get_labels
            all_train_labels = _get_labels(self.train_dataset)
            if all_train_labels is not None:
                if isinstance(all_train_labels, torch.Tensor):
                    all_train_labels = all_train_labels.cpu().numpy()
                client_label_counts = {}
                for cid, idxs in self.client_indices.items():
                    c_labels = all_train_labels[idxs]
                    unique, counts = np.unique(c_labels, return_counts=True)
                    client_label_counts[cid] = dict(zip(unique.tolist(), counts.tolist()))
                self.topology.build_distribution_aware(self.config.clients.num_clients, client_label_counts, seed=self.config.env.seed)
        
        # Fetch initial model vector
        is_ensemble = getattr(self.config.clients, "use_ensemble", False) or getattr(self.config.clients, "hierarchical_ensemble", False)
        compute_mode = getattr(self.config.clients, "compute_optimization_mode", "shared_backbone")
        
        if is_ensemble and compute_mode == "shared_backbone":
            from src.core.model import MultiHeadSimpleCNN
            dummy_model = MultiHeadSimpleCNN(in_channels=self.in_channels)
        else:
            dummy_model = SimpleCNN(in_channels=self.in_channels)

        initial_w = torch.nn.utils.parameters_to_vector(dummy_model.parameters()).detach()
        num_params = initial_w.numel()
        print(f"Model instantiated with {num_params} parameters.")
        
        from src.core.updater import PyTorchLocalUpdater
        self.updater = PyTorchLocalUpdater(device=device, in_channels=self.in_channels)
        self.server_weights: torch.Tensor = initial_w.clone()
        
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
            print(f"Round {round_num}/{self.config.num_rounds} completed.")

        self.save_metrics()
    @abstractmethod
    def run_round(self, round_num: int):
        pass
        
    def evaluate_model(self, weights: torch.Tensor):
        is_ensemble = getattr(self.config.clients, "use_ensemble", False) or getattr(self.config.clients, "hierarchical_ensemble", False)
        compute_mode = getattr(self.config.clients, "compute_optimization_mode", "shared_backbone")
        if is_ensemble and compute_mode == "shared_backbone":
            model = self.updater.multihead_model
        else:
            model = self.updater.global_model

        torch.nn.utils.vector_to_parameters(weights.to(self.device), model.parameters())
        model.eval()
        
        # If test_dataset is FastDataset, this loader is very efficient.
        test_loader = DataLoader(self.test_dataset, batch_size=1024, shuffle=False)
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for images, labels in test_loader:
                # images, labels are already on device if using FastDataset
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                predicted = outputs.argmax(dim=1)
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
        
        # Instantiate two models (reuse from updater)
        global_model = self.updater.global_model
        local_model = self.updater.local_model
        global_model.eval()
        local_model.eval()
        
        for client_id in range(self.config.clients.num_clients):
            state = self.clients_state[client_id]
            if getattr(state, "is_byzantine", False):
                continue
            if getattr(state, "local_weights", None) is None:
                continue
                
            # Load weights
            torch.nn.utils.vector_to_parameters(self.server_weights.to(self.device), global_model.parameters())
            torch.nn.utils.vector_to_parameters(state.local_weights.to(self.device), local_model.parameters())
            
            # Client's local test data
            client_test_ds = ClientDataset(self.test_dataset, self.client_test_indices[client_id])
            if len(client_test_ds) == 0:
                continue
                
            test_loader = DataLoader(client_test_ds, batch_size=len(client_test_ds), shuffle=False)
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    logits_global = global_model(images)
                    logits_local = local_model(images)
                    
                    # Ensemble logits
                    logits_ensemble = alpha * logits_local + (1.0 - alpha) * logits_global
                    
                    loss = criterion(logits_ensemble, labels)
                    total_loss += loss.item()
                    
                    predicted = logits_ensemble.argmax(dim=1)
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
        with open(json_path, "w", encoding='utf-8') as f:
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
