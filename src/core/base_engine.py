import json
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from src.config import SimulationConfig
from src.core.interfaces import Topology, Aggregator, ClientState, MetricsCollector

from src.utils.random import get_random_state
from src.data.dataset import get_mnist, partition_data_non_iid, ClientDataset
from src.core.model import SimpleCNN, model_to_vector, vector_to_model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class BaseEngine(ABC):
    def __init__(self, config: SimulationConfig, topology: Topology, aggregator: Aggregator, device="cpu"):
        self.config = config
        self.topology = topology
        self.aggregator = aggregator
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # Build components
        self.rng = get_random_state()
        self.local_rng = get_random_state()
        
        self.topology.build(self.config.clients.num_clients, seed=self.config.env.seed)
        
        self.metrics = MetricsCollector()
        
        # Fetch datasets
        dataset_name = getattr(self.config.env, "dataset", "mnist").lower()
        if dataset_name in ("cifar10", "cifar100"):
            if dataset_name == "cifar100":
                from src.data.dataset import get_cifar100
                print("Downloading and dividing CIFAR-100 dataset...")
                train_ds, test_ds = get_cifar100(
                    train_subset=getattr(self.config.env, "train_subset", None),
                    test_subset=getattr(self.config.env, "test_subset", None),
                    seed=getattr(self.config.env, "seed", 42)
                )
            else:
                print("Downloading and dividing CIFAR-10 dataset...")
                from src.data.dataset import get_cifar10
                train_ds, test_ds = get_cifar10(
                    train_subset=getattr(self.config.env, "train_subset", None),
                    test_subset=getattr(self.config.env, "test_subset", None),
                    seed=getattr(self.config.env, "seed", 42)
                )
            self.in_channels = 3
        else:
            print("Downloading and dividing MNIST dataset...")
            train_ds, test_ds = get_mnist(
                train_subset=getattr(self.config.env, "train_subset", None),
                test_subset=getattr(self.config.env, "test_subset", None),
                seed=getattr(self.config.env, "seed", 42)
            )
            self.in_channels = 1
        self.num_classes = 100 if dataset_name == "cifar100" else 10

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
        self.client_test_indices_t = {
            cid: torch.tensor(idxs, dtype=torch.long, device=self.device)
            for cid, idxs in self.client_test_indices.items()
        }

        # Check clustering method configuration
        cluster_method = self.config.topology.params.get("cluster_method", None)
        if cluster_method is None:
            cluster_method = "label_aware" if self.config.topology.params.get("cluster_by_label_dist", False) else "random"

        if cluster_method == "label_aware" and hasattr(self.topology, "build_distribution_aware"):
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
        is_ensemble = getattr(self.config.clients, "use_ensemble", False) or getattr(self.config.clients, "hierarchical_ensemble", False) or (getattr(self.config.clients, "compute_optimization_mode", "none") == "shared_backbone")
        compute_mode = getattr(self.config.clients, "compute_optimization_mode", "shared_backbone")
        model_name = getattr(self.config.clients, "model_name", "simple_cnn")
        
        if is_ensemble and compute_mode == "shared_backbone":
            from src.core.model import MultiHeadSimpleCNN, MultiHeadResNet9, MultiHeadMobileNetV3Small
            if model_name == "mobilenetv3":
                dummy_model = MultiHeadMobileNetV3Small(in_channels=self.in_channels, num_classes=self.num_classes)
            elif model_name == "resnet9":
                dummy_model = MultiHeadResNet9(in_channels=self.in_channels, num_classes=self.num_classes)
            else:
                dummy_model = MultiHeadSimpleCNN(in_channels=self.in_channels, num_classes=self.num_classes)
        else:
            from src.core.model import SimpleCNN, ResNet9, MobileNetV3Small
            if model_name == "mobilenetv3":
                dummy_model = MobileNetV3Small(in_channels=self.in_channels, num_classes=self.num_classes)
            elif model_name == "resnet9":
                dummy_model = ResNet9(in_channels=self.in_channels, num_classes=self.num_classes)
            else:
                dummy_model = SimpleCNN(in_channels=self.in_channels, num_classes=self.num_classes)

        initial_w = model_to_vector(dummy_model).detach().to(self.device)
        num_params = initial_w.numel()
        print(f"Model ({model_name}) instantiated with {num_params} parameters.")
        
        from src.core.updater import PyTorchLocalUpdater
        self.updater = PyTorchLocalUpdater(device=device, in_channels=self.in_channels, model_name=model_name, num_classes=self.num_classes)
        self.server_weights: torch.Tensor = initial_w.clone()
        
        # Track clients and pre-cache datasets
        from src.data.dataset import ClientDataset
        self.client_train_datasets = {
            cid: ClientDataset(self.train_dataset, idxs)
            for cid, idxs in self.client_indices.items()
        }
        self.client_test_datasets = {
            cid: ClientDataset(self.test_dataset, idxs)
            for cid, idxs in self.client_test_indices.items()
        }

        self.clients_state = {}
        for client_id in range(self.config.clients.num_clients):
            state = ClientState(client_id, initial_w.clone())
            state.data_samples = len(self.client_indices[client_id])
            if self.rng.rand() < getattr(self.config.robustness, "byzantine_rate", 0.0):
                state.is_byzantine = True
                state.byzantine_type = self.config.robustness.byzantine_type
            self.clients_state[client_id] = state
            
    def get_current_lr(self, round_num: int) -> float:
        base_lr = self.config.clients.local_lr
        num_rounds = getattr(self.config, "num_rounds", 50)
        if num_rounds <= 1:
            return base_lr
        min_lr = self.config.clients.min_lr
        import math
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * (round_num - 1) / (num_rounds - 1)))
        return min_lr + (base_lr - min_lr) * cosine_decay

    def run(self):
        print(f"Starting simulation: {self.config.experiment_name} | Topo: {self.config.topology.type}")
        for round_num in range(1, self.config.num_rounds + 1):
            self.run_round(round_num)
            print(f"Round {round_num}/{self.config.num_rounds} completed.")

        self.save_metrics()

    def should_evaluate(self, round_num: int) -> bool:
        """True on every eval_interval-th round and always on the final round."""
        interval = max(1, getattr(self.config, "eval_interval", 1))
        return round_num == self.config.num_rounds or round_num % interval == 0
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

        vector_to_model(weights.to(self.device), model)
        model.eval()
        
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        if hasattr(self.test_dataset, "images") and hasattr(self.test_dataset, "labels"):
            images = self.test_dataset.images
            labels = self.test_dataset.labels
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss = loss.item()
                predicted = outputs.argmax(dim=1)
                total_samples = labels.size(0)
                total_correct = (predicted == labels).sum().item()
        else:
            from src.data.dataset import get_fast_dataloader
            test_loader = get_fast_dataloader(self.test_dataset, batch_size=1024, shuffle=False)
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    predicted = outputs.argmax(dim=1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()

        acc = 100 * total_correct / total_samples
        avg_loss = total_loss / total_samples
        return acc, avg_loss

    def _fedbabu_deployment_heads(self):
        """Canonical FedBABU deployment stage (Oh et al., ICLR 2022): the head
        is trained on each client's local data after federated training ends
        (body frozen). Returns {cid: head_state_dict} for evaluation."""
        pers_method = getattr(self.config.clients, "personalization_method", "none")
        if pers_method != "fedbabu":
            return {}
        local_model = self.updater.local_model
        head_module = local_model.fc2 if hasattr(local_model, "fc2") else local_model.classifier
        steps = int(getattr(self.config.clients, "babu_finetune_steps", 100))
        lr = self.config.clients.local_lr
        heads = {}
        for client_id in range(self.config.clients.num_clients):
            state = self.clients_state[client_id]
            if getattr(state, "is_byzantine", False) or getattr(state, "local_weights", None) is None:
                continue
            vector_to_model(state.local_weights.to(self.device), local_model)
            for name, p in local_model.named_parameters():
                p.requires_grad_(name.startswith(("fc2", "classifier")))
            optimizer = torch.optim.SGD(
                [p for n, p in local_model.named_parameters() if n.startswith(("fc2", "classifier"))],
                lr=lr)
            train_ds = self.client_train_datasets.get(client_id)
            if train_ds is None or len(train_ds) == 0:
                continue
            from src.data.dataset import get_fast_dataloader
            loader = get_fast_dataloader(train_ds,
                                         batch_size=min(32, len(train_ds)),
                                         shuffle=True)
            local_model.train()
            done = 0
            while done < steps:
                for images, labels in loader:
                    if images.device != self.device:
                        images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.CrossEntropyLoss()(local_model(images), labels)
                    loss.backward()
                    optimizer.step()
                    done += 1
                    if done >= steps:
                        break
            heads[client_id] = {k: v.detach().clone() for k, v in head_module.state_dict().items()}
        for p in local_model.parameters():
            p.requires_grad_(True)
        return heads

    def evaluate_ensemble(self):
        use_ensemble = getattr(self.config.clients, "use_ensemble", False)
        pers_method = getattr(self.config.clients, "personalization_method", "none")
        if not use_ensemble and pers_method == "none":
            return 0.0, 0.0            
        alpha = getattr(self.config.clients, "ensemble_alpha", 0.5)
        babu_heads = self._fedbabu_deployment_heads() if pers_method == "fedbabu" else {}
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        per_client_stats = {}
        self._last_per_client_accuracy = {}
        
        global_model = self.updater.global_model
        local_model = self.updater.local_model
        global_model.eval()
        local_model.eval()
        vector_to_model(self.server_weights.to(self.device), global_model)

        if hasattr(self.test_dataset, "images") and hasattr(self.test_dataset, "labels"):
            images_all = self.test_dataset.images
            labels_all = self.test_dataset.labels
            with torch.no_grad():
                logits_global_all = global_model(images_all)
                for client_id in range(self.config.clients.num_clients):
                    state = self.clients_state[client_id]
                    if getattr(state, "is_byzantine", False) or getattr(state, "local_weights", None) is None:
                        continue
                    idxs = self.client_test_indices_t.get(client_id, None)
                    if idxs is None or len(idxs) == 0:
                        continue

                    c_correct, c_total = 0, 0
                    c_images = images_all[idxs]
                    c_labels = labels_all[idxs]
                    logits_global = logits_global_all[idxs]

                    vector_to_model(state.local_weights.to(self.device), local_model)
                    if client_id in babu_heads:
                        head_module = local_model.fc2 if hasattr(local_model, "fc2") else local_model.classifier
                        head_module.load_state_dict(babu_heads[client_id])
                    local_model.eval()
                    logits_local = local_model(c_images)

                    if pers_method in ("ditto", "fedala", "fedrep", "fedper", "fedbabu"):
                        logits_ensemble = logits_local
                    elif pers_method == "apfl":
                        apfl_a = getattr(state, "apfl_alpha", 0.5)
                        logits_ensemble = apfl_a * logits_local + (1.0 - apfl_a) * logits_global
                    else:
                        logits_ensemble = alpha * logits_local + (1.0 - alpha) * logits_global

                    loss = criterion(logits_ensemble, c_labels)
                    total_loss += loss.item()
                    predicted = logits_ensemble.argmax(dim=1)
                    hits = (predicted == c_labels).sum().item()
                    total_samples += c_labels.size(0)
                    total_correct += hits
                    c_correct += hits
                    c_total += c_labels.size(0)
                    if c_total > 0:
                        per_client_stats[client_id] = (c_correct / c_total) * 100.0
        else:
            from src.data.dataset import get_fast_dataloader
            for client_id in range(self.config.clients.num_clients):
                state = self.clients_state[client_id]
                if getattr(state, "is_byzantine", False) or getattr(state, "local_weights", None) is None:
                    continue
                client_test_ds = self.client_test_datasets[client_id]
                if len(client_test_ds) == 0:
                    continue
                c_correct, c_total = 0, 0
                test_loader = get_fast_dataloader(client_test_ds, batch_size=min(len(client_test_ds), 1024), shuffle=False)
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        logits_global = global_model(images)
                        vector_to_model(state.local_weights.to(self.device), local_model)
                        if client_id in babu_heads:
                            head_module = local_model.fc2 if hasattr(local_model, "fc2") else local_model.classifier
                            head_module.load_state_dict(babu_heads[client_id])
                        local_model.eval()
                        logits_local = local_model(images)

                        if pers_method in ("ditto", "fedala", "fedrep", "fedper", "fedbabu"):
                            logits_ensemble = logits_local
                        elif pers_method == "apfl":
                            apfl_a = getattr(state, "apfl_alpha", 0.5)
                            logits_ensemble = apfl_a * logits_local + (1.0 - apfl_a) * logits_global
                        else:
                            logits_ensemble = alpha * logits_local + (1.0 - alpha) * logits_global

                        loss = criterion(logits_ensemble, labels)
                        total_loss += loss.item()
                        predicted = logits_ensemble.argmax(dim=1)
                        hits = (predicted == labels).sum().item()
                        total_samples += labels.size(0)
                        total_correct += hits
                        c_correct += hits
                        c_total += labels.size(0)
                if c_total > 0:
                    per_client_stats[client_id] = (c_correct / c_total) * 100.0

        if total_samples == 0:
            return 0.0, 0.0

        self._last_per_client_accuracy = per_client_stats
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
