import os
import sys
import argparse
from typing import List, Optional

from src.config import SimulationConfig, EnvironmentConfig, TopologyConfig, ClientConfig
from src.experiments.runner import ExperimentRunner, SingleExperimentRunner


def list_presets() -> List[str]:
    """List built-in YAML experiment presets in configs/."""
    configs_dir = "configs"
    if not os.path.exists(configs_dir):
        return []
    files = [f for f in os.listdir(configs_dir) if f.endswith(".yaml") or f.endswith(".yml")]
    return sorted(files)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Topology-aware Federated Deep Learning (TopoFDL) Simulation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py comparison                               # Run comparison preset
  python main.py byzantine_matrix                        # Run byzantine matrix preset
  python main.py configs/comparison.yaml                 # Run custom YAML config
  python main.py comparison --num-rounds 5 --dataset cifar10
  python main.py smoke-test --dataset mnist               # Run quick smoke test
  python main.py list                                    # List available presets
        """
    )

    # Positional target argument (optional, e.g. preset name or yaml path or 'smoke-test')
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Preset name ('comparison', 'byzantine_matrix', 'ensemble'), YAML path, or 'smoke-test' / 'list'",
    )

    # Flag options
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file or preset name")
    parser.add_argument("--matrix", action="store_true", help="Run the full Byzantine robustness matrix experiment")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test simulation")

    # CLI parameter overrides
    parser.add_argument("--num-rounds", type=int, default=None, help="Override number of simulation rounds")
    parser.add_argument("--dataset", type=str, default=None, choices=["mnist", "cifar10"], help="Override dataset ('mnist' or 'cifar10')")
    parser.add_argument("--seed", type=int, default=None, help="Override global seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")

    return parser


def main(args: Optional[List[str]] = None):
    parser = build_cli_parser()
    parsed_args = parser.parse_args(args)

    target = parsed_args.target

    # Action: list presets
    if target == "list":
        print("Available Experiment Presets:")
        presets = list_presets()
        for p in presets:
            print(f"  - {p}")
        return

    # Determine execution mode
    config_file_or_preset = None

    if parsed_args.matrix or target == "byzantine_matrix":
        config_file_or_preset = "configs/byzantine_matrix.yaml"
    elif parsed_args.config:
        config_file_or_preset = parsed_args.config
    elif target and target not in ("smoke-test", "smoketest"):
        config_file_or_preset = target

    # Handle smoke-test or default single run
    if parsed_args.smoke_test or target in ("smoke-test", "smoketest"):
        dataset = parsed_args.dataset or "mnist"
        rounds = parsed_args.num_rounds or 5
        print(f"Running Smoke Test (Star Topology | Dataset: {dataset} | Rounds: {rounds})...")
        config = SimulationConfig(
            experiment_name=f"smoke_test_{dataset}",
            num_rounds=rounds,
            env=EnvironmentConfig(
                dataset=dataset,
                output_dir=parsed_args.output_dir or "./outputs",
                seed=parsed_args.seed or 42,
            ),
            topology=TopologyConfig(type="star"),
            clients=ClientConfig(num_clients=5, model_dim=0, local_lr=0.01, local_steps=1),
        )
        runner = SingleExperimentRunner(config)
        runner.run()
        return

    # If no config or target provided, default to smoke test
    if not config_file_or_preset:
        dataset = parsed_args.dataset or "mnist"
        rounds = parsed_args.num_rounds or 5
        print("No experiment config specified. Defaulting to Smoke Test (Star Topology)...")
        print("Tip: Run 'python main.py comparison' or 'python main.py --help' to see all experiment presets.\n")
        config = SimulationConfig(
            experiment_name=f"smoke_test_{dataset}",
            num_rounds=rounds,
            env=EnvironmentConfig(
                dataset=dataset,
                output_dir=parsed_args.output_dir or "./outputs",
                seed=parsed_args.seed or 42,
            ),
            topology=TopologyConfig(type="star"),
            clients=ClientConfig(num_clients=5, model_dim=0, local_lr=0.01, local_steps=1),
        )
        runner = SingleExperimentRunner(config)
        runner.run()
        return

    # Run ExperimentRunner with YAML or preset
    overrides = {
        "num_rounds": parsed_args.num_rounds,
        "dataset": parsed_args.dataset,
        "seed": parsed_args.seed,
        "output_dir": parsed_args.output_dir,
    }

    runner = ExperimentRunner.from_yaml(config_file_or_preset, overrides=overrides)
    runner.run()


if __name__ == "__main__":
    main()
