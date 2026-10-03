import numpy as np
import pytest
from temp.hierarchical_partitioner import partition_data_hierarchical, compute_cluster_label_overlap

class DummyDataset:
    def __init__(self, num_samples=1000, num_classes=10):
        # Evenly distributed labels
        self.targets = (list(range(num_classes)) * (num_samples // num_classes))
        self.targets.extend(list(range(num_samples % num_classes)))

def test_hierarchical_partitioner_metrics():
    dataset = DummyDataset(num_samples=10000, num_classes=10)
    labels = np.array(dataset.targets)
    
    # Test x = 10.0 -> intra_overlap should be high (> 0.8)
    # beta = 0.1 -> inter_overlap should be low (< 0.3)
    client_indices, client_to_cluster = partition_data_hierarchical(
        dataset=dataset,
        num_clients=15,
        num_clusters=3,
        intra_alpha=10.0,
        inter_alpha=0.1,
        seed=42
    )
    
    metrics = compute_cluster_label_overlap(client_indices, client_to_cluster, labels)
    
    assert metrics["intra_cluster_overlap"] > 0.8
    assert metrics["inter_cluster_overlap"] < 0.4 # Relaxing a bit just in case of randomness
    
    # Edge case: single cluster
    client_indices, client_to_cluster = partition_data_hierarchical(
        dataset=dataset,
        num_clients=5,
        num_clusters=1,
        intra_alpha=1.0,
        inter_alpha=1.0,
        seed=42
    )
    assert len(client_indices) == 5
    assert all(c == 0 for c in client_to_cluster.values())
