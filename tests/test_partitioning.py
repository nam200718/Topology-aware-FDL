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
    # Check that all indices are unique and cover the dataset
    assert len(set(all_indices)) == 100
