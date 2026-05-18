import os
import sys
import argparse
from datetime import datetime

# Add the project root to the path so we can import from src and main
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)

# --- Virtual environment check ---
if not os.environ.get("VIRTUAL_ENV") and sys.prefix == sys.base_prefix:
    print("ERROR: Virtual environment is not activated.", file=sys.stderr)
    print(f"  Run:  source {_project_root}/.venv/bin/activate", file=sys.stderr)
    sys.exit(1)

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import ExperimentConfig
from main import run_experiment

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "byzantine_matrix.yaml")

def main():
    parser = argparse.ArgumentParser(description="Byzantine Robustness Matrix Experiment")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to YAML config file")
    args = parser.parse_args()

    exp_cfg = ExperimentConfig.from_yaml(args.config)

    # Create a unique timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = os.path.join(exp_cfg.env.output_dir, f"byzantine_matrix_{timestamp}")
    plots_dir = os.path.join(experiment_root, "plots")
    metrics_dir = os.path.join(experiment_root, "metrics")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Build all SimulationConfig instances from the YAML
    entries = exp_cfg.build_configs(metrics_dir=metrics_dir)

    results = []

    print(f"Starting Byzantine Robustness Matrix")
    print(f"Config: {args.config}")
    print(f"Output directory: {experiment_root}")
    print(f"Byzantine Rates: {exp_cfg.byzantine_rates}")
    print("-" * 65)

    for entry in entries:
        config = entry["config"]
        topo_label = entry["topo_label"]
        rate = entry["byzantine_rate"]

        print(f"Topo: {topo_label:15} | Byz Rate: {rate:3.1f}", end=" ", flush=True)

        hx = run_experiment(config)

        final_acc = hx[-1].get("test_accuracy", 0.0)
        is_ensemble = (config.topology.type == "hierarchical_ensemble")
        if is_ensemble and "ensemble_test_accuracy" in hx[-1]:
            final_acc = hx[-1]["ensemble_test_accuracy"]

        results.append({
            "Topology": topo_label,
            "Byzantine Rate": rate,
            "Final Accuracy": final_acc
        })
        print(f"| Final Acc: {final_acc:6.2f}%")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(experiment_root, "matrix_results.csv"), index=False)

    # Plotting
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    sns.lineplot(data=df, x="Byzantine Rate", y="Final Accuracy", hue="Topology", marker="o")
    plt.title(f"Byzantine Robustness Matrix (after {exp_cfg.num_rounds} rounds)")
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Byzantine Rate (Proportion of Malicious Clients)")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "byzantine_robustness_matrix.png"))

    print("-" * 65)
    print(f"Matrix experiment complete!")
    print(f"Summary plot saved to: {os.path.join(plots_dir, 'byzantine_robustness_matrix.png')}")

if __name__ == "__main__":
    main()
