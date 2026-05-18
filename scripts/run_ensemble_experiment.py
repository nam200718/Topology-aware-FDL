import os
import sys
import argparse

# Add the project root to the path so we can import from src and main
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)

# --- Virtual environment check ---
if not os.environ.get("VIRTUAL_ENV") and sys.prefix == sys.base_prefix:
    print("ERROR: Virtual environment is not activated.", file=sys.stderr)
    print(f"  Run:  source {_project_root}/.venv/bin/activate", file=sys.stderr)
    sys.exit(1)

from src.config import ExperimentConfig
from main import run_experiment
from src.utils.visualizer import plot_experiment_results

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "ensemble_experiment.yaml")

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Ensemble Experiment")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to YAML config file")
    args = parser.parse_args()

    exp_cfg = ExperimentConfig.from_yaml(args.config)

    # Build SimulationConfig (single experiment)
    entries = exp_cfg.build_configs()
    config = entries[0]["config"]

    print("Starting Hierarchical Ensemble Experiment...")
    print(f"Config: {args.config}")
    print(f"Topology: {config.topology.type}")
    print(f"Clients: {config.clients.num_clients}, Params: {config.topology.params}")

    history = run_experiment(config)

    # Generate charts
    metrics_path = os.path.join(config.env.output_dir, config.experiment_name, "metrics.json")
    plot_experiment_results(metrics_path)

    print("\nExperiment Completed!")
    print(f"Final Global Test Accuracy: {history[-1]['test_accuracy']:.2f}%")
    if "ensemble_test_accuracy" in history[-1]:
        print(f"Final Ensemble Test Accuracy: {history[-1]['ensemble_test_accuracy']:.2f}%")

    print(f"Results saved in {os.path.join(config.env.output_dir, config.experiment_name)}")

if __name__ == "__main__":
    main()
