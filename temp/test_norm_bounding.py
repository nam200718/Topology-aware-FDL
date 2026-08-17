import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pytest
from src.core.interfaces import ClientState
from src.defense.config import DefenseConfig
from src.defense.aggregator import SoftRejectionAggregator
from temp.attacks import check_inf_nan

def test_norm_bounding():
    config = DefenseConfig(norm_bounding_enabled=True, norm_bounding_multiplier=2.0)
    aggregator = SoftRejectionAggregator(config)
    
    # 3 clients, 1 normal, 1 normal, 1 malicious with huge norm
    s1 = ClientState(0, torch.tensor([1.0, 1.0]))
    s2 = ClientState(1, torch.tensor([1.2, 0.8]))
    s3 = ClientState(2, torch.tensor([100.0, 100.0])) # huge norm
    
    reference = torch.tensor([0.0, 0.0])
    
    # _apply_norm_bounding should clip s3's delta
    deltas = [s1.weights - reference, s2.weights - reference, s3.weights - reference]
    clipped_deltas = aggregator._apply_norm_bounding(deltas)
    
    assert clipped_deltas[0].allclose(deltas[0])
    assert clipped_deltas[1].allclose(deltas[1])
    assert clipped_deltas[2].norm() < deltas[2].norm()
    
def test_hard_rejection():
    config = DefenseConfig(hard_rejection_enabled=True, hard_rejection_threshold=0.0)
    aggregator = SoftRejectionAggregator(config)
    
    # similarities: 1.0, 0.5, -0.5 (malicious)
    similarities = [1.0, 0.5, -0.5]
    rejected = aggregator._apply_hard_rejection(similarities)
    
    assert rejected[0] == 1.0
    assert rejected[1] == 0.5
    assert rejected[2] == float('-inf')

def test_inf_nan_sentinel():
    initial = torch.tensor([1.0, 1.0])
    clean = torch.tensor([1.2, 0.8])
    malicious = torch.tensor([float('inf'), float('nan')])
    
    assert check_inf_nan(clean, initial) == True
    assert check_inf_nan(malicious, initial) == False
