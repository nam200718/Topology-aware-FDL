import os
import json
import tempfile
import pytest
import torch
from torch.utils.data import TensorDataset

from src.core.interfaces import ClientState
from src.core.updater import PyTorchLocalUpdater
from src.config import SimulationConfig, TopologyConfig, ClientConfig, EnvironmentConfig
from src.topologies.hierarchical import HierarchicalTopology
from src.core.aggregator import FedAvgAggregator
from src.core.unified_defended_engine import UnifiedDefendedEngine
from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine
from src.defense.config import DefenseConfig
from src.data.dataset import FastDataset, ClientDataset, _acquire_download_lock, _release_download_lock, _auto_link_dataset
from src.utils.device import resolve_device
from src.baselines.run_all_baselines import _atomic_json_dump, run_personalization_suite, run_byzantine_suite


def test_client_state_copy_attributes():
    cs = ClientState(client_id=1, initial_weights=torch.ones(10))
    cs.is_confirmed_malicious = True
    cs.participation_count = 5
    cs.control_variate = torch.tensor([1.0, 2.0, 3.0])
    cs._scaffold_delta_c = torch.tensor([0.1, 0.2, 0.3])
    cs.active_mask = torch.tensor([True, False, True])

    copy_cs = cs.copy()
    assert copy_cs.client_id == 1
    assert copy_cs.is_confirmed_malicious is True
    assert copy_cs.participation_count == 5
    assert torch.equal(copy_cs.control_variate, cs.control_variate)
    assert torch.equal(copy_cs._scaffold_delta_c, cs._scaffold_delta_c)
    assert torch.equal(copy_cs.active_mask, cs.active_mask)

    # Ensure deep copy
    copy_cs.control_variate[0] = 999.0
    assert cs.control_variate[0] == 1.0


def test_hierarchical_ensemble_participation_count_increment():
    sim_cfg = SimulationConfig(
        num_rounds=3,
        topology=TopologyConfig(type="hierarchical_ensemble", params={"num_clusters": 2}),
        clients=ClientConfig(num_clients=4, model_name="simple_cnn", compute_optimization_mode="shared_backbone", hierarchical_ensemble=True),
        env=EnvironmentConfig(seed=42, dataset="synthetic")
    )
    topo = HierarchicalTopology(num_clusters=2)
    topo.build(num_clients=4, seed=42)
    agg = FedAvgAggregator()
    engine = HierarchicalEnsembleEngine(sim_cfg, topo, agg, device="cpu")

    x = torch.randn(40, 1, 28, 28)
    y = torch.randint(0, 10, (40,))
    td = TensorDataset(x, y)
    fd = FastDataset(td, device="cpu")
    engine.client_train_datasets = [ClientDataset(fd, list(range(i * 10, (i + 1) * 10))) for i in range(4)]
    engine.client_test_datasets = [ClientDataset(fd, list(range(i * 10, (i + 1) * 10))) for i in range(4)]

    # Initial participation count is 0
    assert all(engine.clients_state[i].participation_count == 0 for i in range(4))

    # Round 1
    engine.run_round(1)
    assert all(engine.clients_state[i].participation_count == 1 for i in range(4))

    # Round 2
    engine.run_round(2)
    assert all(engine.clients_state[i].participation_count == 2 for i in range(4))


def test_updater_byzantine_attack_on_parent_and_local_weights():
    updater = PyTorchLocalUpdater(device="cpu", num_classes=10)
    state = ClientState(client_id=0, initial_weights=torch.zeros(20))
    state.weights = torch.ones(20)
    state.parent_weights = torch.ones(20) * 2.0
    state.local_weights = torch.ones(20) * 3.0
    state.is_byzantine = True

    # Test sign_flip
    state.byzantine_type = "sign_flip"
    initial_weights = torch.zeros(20)
    updated = updater._apply_byzantine_attack(state, initial_weights)
    # delta for weights = 1 - 0 = 1; attack = 0 - 1.5 * 1 = -1.5
    assert torch.allclose(updated.weights, torch.full((20,), -1.5))
    # delta for parent_weights = 2 - 0 = 2; attack = 0 - 1.5 * 2 = -3.0
    assert torch.allclose(updated.parent_weights, torch.full((20,), -3.0))
    # delta for local_weights = 3 - 0 = 3; attack = 0 - 1.5 * 3 = -4.5
    assert torch.allclose(updated.local_weights, torch.full((20,), -4.5))

    # Test gradient_ascent
    state.weights = torch.ones(20)
    state.parent_weights = torch.ones(20)
    state.local_weights = torch.ones(20)
    state.byzantine_type = "gradient_ascent"
    updated = updater._apply_byzantine_attack(state, initial_weights)
    # delta = 1 - 0 = 1; attack = 0 - 5.0 * 1 = -5.0
    assert torch.allclose(updated.weights, torch.full((20,), -5.0))
    assert torch.allclose(updated.parent_weights, torch.full((20,), -5.0))
    assert torch.allclose(updated.local_weights, torch.full((20,), -5.0))


def test_sentinel_nan_guard_and_trust_indexing_bounds():
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

    # Mock updater to inject Inf for client 1
    orig_update = engine.updater.update
    def mock_update(state, *args, **kwargs):
        res = orig_update(state, *args, **kwargs)
        if state.client_id == 1:
            res.weights[0] = float("inf")
        return res
    engine.updater.update = mock_update

    # Mock get_last_trust_scores to return a 1-element tensor (fewer than clean_states)
    engine.cluster_defense_aggregator.get_last_trust_scores = lambda: torch.tensor([0.9])
    engine.global_defense_aggregator.get_last_trust_scores = lambda: torch.tensor([0.9])

    # Should run round without raising IndexError or propagating Inf
    engine.run_round(1)
    assert engine.clients_state[1].is_confirmed_malicious is True
    assert not torch.isnan(engine.server_weights).any()
    assert not torch.isinf(engine.server_weights).any()


def test_aclm_evaluation_loss_stability():
    sim_cfg = SimulationConfig(
        num_rounds=1,
        topology=TopologyConfig(type="hierarchical_ensemble", params={"num_clusters": 2}),
        clients=ClientConfig(
            num_clients=2,
            model_name="simple_cnn",
            compute_optimization_mode="shared_backbone",
            hierarchical_ensemble=True,
            active_class_inference_mask=True,
        ),
        env=EnvironmentConfig(seed=42, dataset="synthetic")
    )
    topo = HierarchicalTopology(num_clusters=2)
    topo.build(num_clients=2, seed=42)
    agg = FedAvgAggregator()
    engine = HierarchicalEnsembleEngine(sim_cfg, topo, agg, device="cpu")

    # Set up client datasets with labels
    x = torch.randn(20, 1, 28, 28)
    y = torch.randint(0, 10, (20,))
    td = TensorDataset(x, y)
    fd = FastDataset(td, device="cpu")
    engine.client_train_datasets = [ClientDataset(fd, list(range(10))), ClientDataset(fd, list(range(10, 20)))]
    engine.client_test_datasets = [ClientDataset(fd, list(range(10))), ClientDataset(fd, list(range(10, 20)))]

    # Only class 0 is active for client 0; test data has classes 0-9
    mask = torch.zeros(10, dtype=torch.bool)
    mask[0] = True
    engine.clients_state[0].active_mask = mask
    engine.clients_state[1].active_mask = mask

    acc, loss = engine.evaluate_ensemble(round_num=1)
    # Loss should be finite and not exploded
    assert not torch.isnan(torch.tensor(loss))
    assert not torch.isinf(torch.tensor(loss))
    assert loss < 100.0  # Exploded masked logits produced losses > 1e8


def test_download_lock_uses_tempdir():
    with tempfile.TemporaryDirectory() as fake_readonly_dir:
        # Acquire lock for mnist
        lock = _acquire_download_lock(fake_readonly_dir, "test_mnist")
        assert lock is not None
        # Verify the lock file is in tempfile.gettempdir()
        expected_lock_path = os.path.join(tempfile.gettempdir(), ".test_mnist.lock")
        assert os.path.exists(expected_lock_path)
        _release_download_lock(lock)


def test_auto_link_cleans_broken_symlink(tmp_path):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    target_dir = os.path.join(data_dir, "cifar-10-batches-py")

    # Create broken symlink
    os.symlink("/nonexistent/path/to/cifar", target_dir)
    assert os.path.islink(target_dir)
    assert not os.path.exists(target_dir)

    # Calling _auto_link_dataset should clean up broken symlink
    _auto_link_dataset(data_dir, "cifar10")
    assert not os.path.islink(target_dir)


def test_resolve_device_forced_cuda_indexing(monkeypatch):
    # Verify forced "cuda:0" or "cuda:1" is parsed
    monkeypatch.setenv("HEP_FORCE_DEVICE", "cuda:1")
    # If cuda is not available on this CPU machine, it falls back to cpu
    # But if mock probe passes, it returns "cuda:1"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    from src.utils import device as dev_module
    monkeypatch.setattr(dev_module, "_probe_op_suite", lambda d: True if d == "cuda:1" else False)
    assert resolve_device() == "cuda:1"


def test_atomic_json_dump_and_checkpoint_resumption(tmp_path):
    out_dir = str(tmp_path / "ckpt_test")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_file = os.path.join(out_dir, "test.json")

    data = [{"method": "fedavg", "regime": "iid", "mean_acc": 85.0}]
    _atomic_json_dump(data, ckpt_file)

    assert os.path.exists(ckpt_file)
    with open(ckpt_file) as f:
        loaded = json.load(f)
    assert loaded == data


def test_sentinel_catches_nan_in_parent_weights_and_preserves_trust_alignment():
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

    # Mock updater: client 0 is honest, client 1 has NaN in parent_weights (weights is clean)
    orig_update = engine.updater.update
    def mock_update(state, *args, **kwargs):
        res = orig_update(state, *args, **kwargs)
        if state.client_id == 1:
            res.parent_weights[0] = float("nan")
        return res
    engine.updater.update = mock_update

    engine.run_round(1)
    # Sentinel guard must have caught NaN in parent_weights
    assert engine.clients_state[1].is_confirmed_malicious is True
    # In round 2, client 1 must remain isolated
    engine.run_round(2)
    assert engine.clients_state[1].is_confirmed_malicious is True
    assert not torch.isnan(engine.server_weights).any()

