import pytest
import numpy as np
from src.core.interfaces import ClientState
from src.core.aggregator import FedAvgAggregator

from src.topologies.star import StarTopology
from src.topologies.checks import check_star_invariant

import torch

def test_fedavg_aggregator():
    agg = FedAvgAggregator()
    s1 = ClientState(1, torch.tensor([1.0, 2.0]))
    s1.data_samples = 10
    
    s2 = ClientState(2, torch.tensor([3.0, 4.0]))
    s2.data_samples = 10
    
    # Simple unweighted average should be [2.0, 3.0] since data_samples are equal
    res = agg.aggregate([s1, s2])
    assert torch.allclose(res, torch.tensor([2.0, 3.0]))


def test_star_topology_invariant():
    topo = StarTopology()
    topo.build(10, seed=42)
    check_star_invariant(topo, 10)
    
    neighbors = topo.get_neighbors(1)
    # Client neighbor should just be server
    assert len(neighbors) == 1
    assert neighbors[0] == topo.server_id
