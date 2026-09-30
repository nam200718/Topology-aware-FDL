import os
import pytest
import tempfile
from src.config import SimulationConfig, EnvironmentConfig, TopologyConfig, ClientConfig, ExperimentConfig, TopologyEntry
from src.experiments.builder import TopologyEngineFactory, check_invariants, detect_device
from src.experiments.runner import SingleExperimentRunner, ExperimentRunner
from src.experiments.cli import build_cli_parser, main as cli_main


def test_detect_device():
    device = detect_device()
    assert isinstance(device, str)
    assert len(device) > 0


def test_topology_engine_factory():
    sim_config = SimulationConfig(
        experiment_name="test_factory",
        num_rounds=1,
        topology=TopologyConfig(type="star"),
        clients=ClientConfig(num_clients=5),
    )
    topology, engine_cls = TopologyEngineFactory.build(sim_config)
    assert topology is not None
    assert engine_cls.__name__ == "CentralizedEngine"


def test_single_experiment_runner(tmp_path):
    sim_config = SimulationConfig(
        experiment_name="test_single_run",
        num_rounds=1,
        env=EnvironmentConfig(output_dir=str(tmp_path), seed=42),
        topology=TopologyConfig(type="star"),
        clients=ClientConfig(num_clients=5, model_dim=0, local_lr=0.01, local_steps=1),
    )
    runner = SingleExperimentRunner(sim_config)
    history = runner.run()
    assert len(history) == 1
    assert "test_accuracy" in history[0]


def test_experiment_runner_preset_resolution(tmp_path):
    runner = ExperimentRunner.from_yaml("comparison", overrides={"num_rounds": 1, "output_dir": str(tmp_path)})
    assert runner.exp_config.num_rounds == 1
    assert runner.exp_config.env.output_dir == str(tmp_path)


def test_cli_parser():
    parser = build_cli_parser()
    args = parser.parse_args(["comparison", "--num-rounds", "3", "--dataset", "mnist"])
    assert args.target == "comparison"
    assert args.num_rounds == 3
    assert args.dataset == "mnist"


def test_run_all_baselines_cli_parsing():
    import sys
    from unittest.mock import patch
    from src.baselines.run_all_baselines import parse_args, _parse_list_arg
    from src.baselines.experiment_configs import get_method_meta, get_regime_meta, get_attack_meta

    # 1. Test space-separated seeds, methods, regimes, and track alias
    test_argv = [
        "run_all_baselines.py",
        "--track", "personalization",
        "--seeds", "42", "123", "7",
        "--methods", "hep_fl", "fedavg",
        "--regimes", "0.1", "0.5",
    ]
    with patch.object(sys, "argv", test_argv):
        args = parse_args()
        assert args.job == "personalization"
        assert _parse_list_arg(args.seeds, int) == [42, 123, 7]
        assert [get_method_meta(m)["id"] for m in _parse_list_arg(args.methods, str)] == ["topo", "fedavg"]
        assert [get_regime_meta(r)["id"] for r in _parse_list_arg(args.regimes, str)] == ["severe", "moderate"]

    # 2. Test comma-separated values
    test_argv_2 = [
        "run_all_baselines.py",
        "--job", "byzantine",
        "--seeds", "42,123,7",
        "--attacks", "label_flip,noise",
        "--rates", "0.1,0.2",
    ]
    with patch.object(sys, "argv", test_argv_2):
        args_2 = parse_args()
        assert _parse_list_arg(args_2.seeds, int) == [42, 123, 7]
        assert [get_attack_meta(a)["id"] for a in _parse_list_arg(args_2.attacks, str)] == ["label_flip", "random_noise"]
        assert _parse_list_arg(args_2.rates, float) == [0.1, 0.2]

