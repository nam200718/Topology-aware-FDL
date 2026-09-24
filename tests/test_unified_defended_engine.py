"""
Unit tests for UnifiedDefendedEngine.
Verifies multi-tier defense instantiation, reputation tracking, and sentinel guard.
"""

import pytest
import torch

from src.config import SimulationConfig, TopologyConfig, ClientConfig, EnvironmentConfig
from src.topologies.hierarchical import HierarchicalTopology
from src.core.aggregator import FedAvgAggregator
from src.core.unified_defended_engine import UnifiedDefendedEngine
from src.defense.config import DefenseConfig


def test_unified_defended_engine_init():
    sim_cfg = SimulationConfig(
        num_rounds=2,
        topology=TopologyConfig(type="hierarchical_ensemble", params={"num_clusters": 2}),
        clients=ClientConfig(num_clients=4, model_name="simple_cnn"),
        env=EnvironmentConfig(seed=42, dataset="synthetic")
    )
    topo = HierarchicalTopology(num_clusters=2)
    topo.build(num_clients=4, seed=42)
    agg = FedAvgAggregator()

    def_cfg = DefenseConfig(
        defense_mode="soft_cosine",
        temperature=0.5,
        defense_scope="both"
    )

    engine = UnifiedDefendedEngine(sim_cfg, topo, agg, device="cpu", defense_config=def_cfg)

    assert engine is not None
    assert engine.defense_config.defense_scope == "both"
    assert len(engine.agent_reputations) == 4
    assert all(rep == 1.0 for rep in engine.agent_reputations.values())
    assert engine.cluster_defense_aggregator is not None
    assert engine.global_defense_aggregator is not None


def test_unified_defended_engine_byzantine_mitigation():
    from torch.utils.data import TensorDataset
    from src.data.dataset import FastDataset, ClientDataset

    sim_cfg = SimulationConfig(
        num_rounds=2,
        topology=TopologyConfig(type="hierarchical_ensemble", params={"num_clusters": 2}),
        clients=ClientConfig(num_clients=4, model_name="simple_cnn", compute_optimization_mode="shared_backbone", hierarchical_ensemble=True),
        env=EnvironmentConfig(seed=42, dataset="synthetic")
    )
    topo = HierarchicalTopology(num_clusters=2)
    topo.build(num_clients=4, seed=42)
    agg = FedAvgAggregator()
    def_cfg = DefenseConfig(defense_mode="soft_cosine", temperature=0.5, defense_scope="both")
    engine = UnifiedDefendedEngine(sim_cfg, topo, agg, device="cpu", defense_config=def_cfg)

    x = torch.randn(40, 1, 28, 28)
    y = torch.randint(0, 10, (40,))
    td = TensorDataset(x, y)
    fd = FastDataset(td, device="cpu")
    engine.client_train_datasets = [ClientDataset(fd, list(range(i * 10, (i + 1) * 10))) for i in range(4)]
    engine.client_test_datasets = [ClientDataset(fd, list(range(i * 10, (i + 1) * 10))) for i in range(4)]

    # Client 0 is an adversarial sign-flipping Byzantine agent
    engine.clients_state[0].is_byzantine = True
    engine.clients_state[0].byzantine_type = "sign_flip"

    engine.run_round(1)

    # Reputation of Byzantine client must be strictly lower than honest clients
    assert engine.agent_reputations[0] < engine.agent_reputations[1]
    assert engine.agent_reputations[1] > 0.80


def test_unified_defended_engine_sentinel_nan_guard():
    from torch.utils.data import TensorDataset
    from src.data.dataset import FastDataset, ClientDataset

    sim_cfg = SimulationConfig(
        num_rounds=2,
        topology=TopologyConfig(type="hierarchical_ensemble", params={"num_clusters": 2}),
        clients=ClientConfig(num_clients=4, model_name="simple_cnn", compute_optimization_mode="shared_backbone", hierarchical_ensemble=True),
        env=EnvironmentConfig(seed=42, dataset="synthetic")
    )
    topo = HierarchicalTopology(num_clusters=2)
    topo.build(num_clients=4, seed=42)
    agg = FedAvgAggregator()
    def_cfg = DefenseConfig(defense_mode="soft_cosine", temperature=0.5, defense_scope="both")
    engine = UnifiedDefendedEngine(sim_cfg, topo, agg, device="cpu", defense_config=def_cfg)

    x = torch.randn(40, 1, 28, 28)
    y = torch.randint(0, 10, (40,))
    td = TensorDataset(x, y)
    fd = FastDataset(td, device="cpu")
    engine.client_train_datasets = [ClientDataset(fd, list(range(i * 10, (i + 1) * 10))) for i in range(4)]
    engine.client_test_datasets = [ClientDataset(fd, list(range(i * 10, (i + 1) * 10))) for i in range(4)]

    # Mock updater to inject NaN for client 0
    orig_update = engine.updater.update
    def mock_update(state, *args, **kwargs):
        res = orig_update(state, *args, **kwargs)
        if state.client_id == 0:
            res.weights[0] = float("nan")
        return res
    engine.updater.update = mock_update

    engine.run_round(1)

    # Sentinel guard must catch the NaN and immediately flag client 0
    assert getattr(engine.clients_state[0], "is_confirmed_malicious", False) is True

