import pytest
import torch
import numpy as np
from src.core.model import MultiHeadSimpleCNN
from src.core.updater import PyTorchLocalUpdater
from src.core.interfaces import ClientState
from src.config import ClientConfig, SimulationConfig, TopologyConfig, NonIIDConfig
from src.topologies.hierarchical import HierarchicalTopology
from torch.utils.data import TensorDataset

def test_multi_head_cnn():
    model = MultiHeadSimpleCNN(in_channels=1, num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    
    # Test head="all"
    lr, lp, ll = model(x, head="all")
    assert lr.shape == (4, 10)
    assert lp.shape == (4, 10)
    assert ll.shape == (4, 10)
    
    # Test individual heads
    assert model(x, head="root").shape == (4, 10)
    assert model(x, head="parent").shape == (4, 10)
    assert model(x, head="local").shape == (4, 10)

def test_shared_backbone_updater():
    updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
    
    # Create dummy dataset
    images = torch.randn(20, 1, 28, 28)
    labels = torch.randint(0, 10, (20,))
    ds = TensorDataset(images, labels)
    
    state = ClientState(0, torch.nn.utils.parameters_to_vector(updater.multihead_model.parameters()).detach())
    
    cfg = ClientConfig(
        use_ensemble=True,
        hierarchical_ensemble=True,
        compute_optimization_mode="shared_backbone",
        ensemble_distillation=True,
        distillation_lambda=0.5,
        local_steps=1
    )
    
    updated_state = updater.update(state, ds, cfg, np.random.RandomState(42))
    assert updated_state.weights is not None
    assert updated_state.parent_weights is not None
    assert updated_state.local_weights is not None
    assert updated_state.weights.shape == state.weights.shape

def test_distribution_aware_clustering():
    topo = HierarchicalTopology(num_clusters=2)
    label_counts = {
        0: {0: 50, 1: 50},
        1: {0: 48, 1: 52},
        2: {8: 50, 9: 50},
        3: {8: 49, 9: 51}
    }
    topo.build_distribution_aware(4, label_counts, seed=42)
    
    # Clients 0 & 1 should be in the same cluster, and clients 2 & 3 in the same cluster
    head_0 = topo.get_neighbors(0)[0]
    head_1 = topo.get_neighbors(1)[0]
    head_2 = topo.get_neighbors(2)[0]
    head_3 = topo.get_neighbors(3)[0]
    
    assert head_0 == head_1
    assert head_2 == head_3
    assert head_0 != head_2

def test_resnet9_and_multihead():
    from src.core.model import ResNet9, MultiHeadResNet9
    model = ResNet9(in_channels=3, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    
    multi_model = MultiHeadResNet9(in_channels=3, num_classes=10)
    lr, lp, ll = multi_model(x, head="all")
    assert lr.shape == (2, 10)
    assert lp.shape == (2, 10)
    assert ll.shape == (2, 10)

def test_ditto_and_apfl_updaters():
    updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
    images = torch.randn(20, 1, 28, 28)
    labels = torch.randint(0, 10, (20,))
    ds = TensorDataset(images, labels)
    
    initial_w = torch.nn.utils.parameters_to_vector(updater.global_model.parameters()).detach()
    state = ClientState(0, initial_w)
    
    # Test Ditto
    cfg_ditto = ClientConfig(personalization_method="ditto", ditto_lambda=0.1, local_steps=1)
    state_ditto = updater.update(state.copy(), ds, cfg_ditto, np.random.RandomState(42))
    assert state_ditto.weights is not None
    assert state_ditto.local_weights is not None
    
    # Test APFL
    cfg_apfl = ClientConfig(personalization_method="apfl", apfl_alpha=0.5, local_steps=1)
    state_apfl = updater.update(state.copy(), ds, cfg_apfl, np.random.RandomState(42))
    assert state_apfl.weights is not None
    assert state_apfl.local_weights is not None
    assert 0.0 <= state_apfl.apfl_alpha <= 1.0


def test_update_similarity_clustering():
    from src.topologies.hierarchical import HierarchicalTopology, _random_project
    
    # 1. Test random projection
    v = np.random.randn(10, 1000)
    proj = _random_project(v, target_dim=256, seed=42)
    assert proj.shape == (10, 256)

    # 2. Test build_update_similarity
    topo = HierarchicalTopology(num_clusters=2)
    
    base_1 = np.tile(np.array([1.0, 0.0]), 50)
    base_2 = np.tile(np.array([0.0, 1.0]), 50)
    
    updates = {
        0: base_1 + np.random.randn(100) * 0.01,
        1: base_1 + np.random.randn(100) * 0.01,
        2: base_2 + np.random.randn(100) * 0.01,
        3: base_2 + np.random.randn(100) * 0.01,
    }
    
    topo.build_update_similarity(4, updates, seed=42)
    
    head_0 = topo.get_neighbors(0)[0]
    head_1 = topo.get_neighbors(1)[0]
    head_2 = topo.get_neighbors(2)[0]
    head_3 = topo.get_neighbors(3)[0]
    
    assert head_0 == head_1
    assert head_2 == head_3
    assert head_0 != head_2



