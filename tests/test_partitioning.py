import pytest
import numpy as np
from src.data.dataset import partition_data

class MockDataset:
    def __init__(self, num_samples):
        self.targets = torch.randint(0, 10, (num_samples,))
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples

import torch

def test_partition_data_iid():
    dataset = MockDataset(100)
    num_clients = 10
    client_indices = partition_data(dataset, num_clients, non_iid=False, seed=42)
    
    assert len(client_indices) == num_clients
    for i in range(num_clients):
        assert len(client_indices[i]) == 10
    
    # Check that all indices are unique and cover the dataset
    all_indices = []
    for i in range(num_clients):
        all_indices.extend(client_indices[i])
    assert len(set(all_indices)) == 100

def test_partition_data_non_iid():
    dataset = MockDataset(100)
    # Ensure targets are balanced for easy partitioning
    dataset.targets = torch.tensor([i % 10 for i in range(100)])
    
    num_clients = 5
    alpha = 0.5
    client_indices = partition_data(dataset, num_clients, non_iid=True, alpha=alpha, seed=42)
    
    assert len(client_indices) == num_clients
    for i in range(num_clients):
        assert len(client_indices[i]) > 0
    
    all_indices = []
    for i in range(num_clients):
        all_indices.extend(client_indices[i])

def test_partition_data_non_iid_skewness():
    """Verify that smaller alpha leads to higher label skew across clients."""
    num_samples = 1000
    num_classes = 10
    samples_per_class = num_samples // num_classes

    # Create a perfectly balanced dataset
    targets = torch.cat([torch.full((samples_per_class,), i) for i in range(num_classes)])

    class BalancedMockDataset:
        def __init__(self, targets):
            self.targets = targets
            self.num_samples = len(targets)
        def __len__(self):
            return self.num_samples
        def __getitem__(self, index):
            return None, self.targets[index]

    dataset = BalancedMockDataset(targets)
    num_clients = 5

    def calculate_skew(alpha):
        client_indices = partition_data(dataset, num_clients, non_iid=True, alpha=alpha, seed=42)

        # Matrix of counts: [client, class]
        counts = np.zeros((num_clients, num_classes))
        for client_id, indices in client_indices.items():
            for idx in indices:
                label = dataset.targets[idx].item()
                counts[client_id, label] += 1

        # Use coefficient of variation (std/mean) as a measure of skewness
        # Higher CV means more non-IID
        stds = np.std(counts, axis=0)
        means = np.mean(counts, axis=0)
        return np.mean(stds / means)

    skew_high = calculate_skew(0.1)   # Should be high skew
    skew_low = calculate_skew(100.0)  # Should be low skew (close to IID)

    assert skew_high > skew_low, f"Expected high skew ({skew_high}) > low skew ({skew_low})"
