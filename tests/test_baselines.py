"""Tests for baseline implementations: FedAvg, FedProx, Multi-Krum, SCAFFOLD, Ditto, and Topo.
"""
import pytest
import torch
import numpy as np

from src.config import SimulationConfig, TopologyConfig, ClientConfig, EnvironmentConfig
from src.core.interfaces import ClientState
from src.baselines.config import BaselineSimulationConfig, BaselineClientConfig
from src.baselines.fedprox_updater import FedProxUpdater
from src.baselines.multi_krum_aggregator import MultiKrumAggregator, multi_krum_scores
from src.baselines.scaffold_updater import ScaffoldUpdater
from src.baselines.scaffold_engine import ScaffoldEngine
from src.baselines.factory import build_baseline_engine
from src.baselines.multi_seed_runner import MultiSeedRunner
from src.topologies.star import StarTopology


def test_multi_krum_scores_and_aggregation():
    # 5 clients: 3 honest (near 1.0), 2 malicious (huge values)
    honest = torch.ones(3, 10) + torch.randn(3, 10) * 0.05
    malicious = torch.ones(2, 10) * 100.0
    stacked = torch.cat([honest, malicious], dim=0)

    # With N=5, f=1: k = 5 - 1 - 2 = 2 nearest neighbors
    scores = multi_krum_scores(stacked, f=1)
    assert scores.shape == (5,)

    # Honest scores must be much smaller than malicious
    assert scores[:3].max() < scores[3:].min()

    # Multi-Krum aggregator should select honest updates
    states = [ClientState(i, stacked[i].clone()) for i in range(5)]
    aggregator = MultiKrumAggregator(f=1, m=2)
    aggregated = aggregator.aggregate(states)

    assert aggregated.shape == (10,)
    # Aggregated output should be close to 1.0, not contaminated by 100.0
    assert torch.allclose(aggregated, torch.ones(10), atol=0.5)


def test_fedprox_updater_runs():
    updater = FedProxUpdater(device="cpu", in_channels=1, model_name="simple_cnn", num_classes=10)
    dummy_model = updater.global_model
    from src.core.model import model_to_vector
    initial_w = model_to_vector(dummy_model).detach()

    state = ClientState(client_id=0, initial_weights=initial_w.clone())
    cfg = BaselineClientConfig(
        personalization_method="fedprox",
        fedprox_mu=0.05,
        local_steps=1,
        local_lr=0.01,
    )

    from torch.utils.data import TensorDataset
    x = torch.randn(20, 1, 28, 28)
    y = torch.randint(0, 10, (20,))
    ds = TensorDataset(x, y)

    updated_state = updater.update(state, ds, cfg, rng=np.random.RandomState(42))
    assert updated_state.weights.shape == initial_w.shape
    # Weights must have changed
    assert not torch.equal(updated_state.weights, initial_w)


def test_scaffold_updater_and_engine():
    config = BaselineSimulationConfig(
        experiment_name="test_scaffold",
        num_rounds=1,
        eval_interval=1,
        env=EnvironmentConfig(dataset="synthetic", seed=42),
        clients=BaselineClientConfig(
            num_clients=3,
            model_name="simple_cnn",
            personalization_method="scaffold",
            local_steps=1,
            local_lr=0.01,
        ),
    )

    topo = StarTopology()
    from src.core.aggregator import FedAvgAggregator
    engine = ScaffoldEngine(config, topo, FedAvgAggregator(), device="cpu")

    # Verify server control variate initialized
    assert hasattr(engine, "server_control_variate")
    assert engine.server_control_variate.numel() > 0

    engine.run_round(1)
    history = engine.metrics.get_history()
    assert len(history) == 1
    assert "test_accuracy" in history[0]


@pytest.mark.parametrize("method_id", [
    "fedavg",
    "fedprox",
    "multikrum",
    "scaffold",
    "ditto",
    "topo",
    "topo_defended",
])
def test_all_baselines_build_and_step(method_id):
    """Smoke test ensuring all 6 target algorithms can build and execute 1 round."""
    config = BaselineSimulationConfig(
        experiment_name=f"smoke_{method_id}",
        num_rounds=1,
        eval_interval=1,
        env=EnvironmentConfig(dataset="synthetic", seed=42),
        topology=TopologyConfig(
            type="hierarchical_ensemble" if "topo" in method_id else "star",
            params={"num_clusters": 2, "defense_mode": "soft_cosine" if method_id == "topo_defended" else "none"},
        ),
        clients=BaselineClientConfig(
            num_clients=4,
            model_name="simple_cnn",
            personalization_method="fedprox" if method_id == "fedprox"
            else "scaffold" if method_id == "scaffold"
            else "ditto" if method_id == "ditto"
            else "none",
            use_ensemble="topo" in method_id,
            hierarchical_ensemble="topo" in method_id,
            local_steps=1,
            local_lr=0.01,
        ),
    )

    topology, aggregator, engine = build_baseline_engine(
        config=config,
        method_id=method_id,
        device="cpu",
    )

    assert topology is not None
    assert aggregator is not None
    assert engine is not None

    engine.run_round(1)
    history = engine.metrics.get_history()
    assert len(history) == 1
    assert "test_accuracy" in history[0]
