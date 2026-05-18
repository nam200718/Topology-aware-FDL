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
from src.utils.visualizer import plot_comparison

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "comparison.yaml")

def main():
    parser = argparse.ArgumentParser(description="Topology Comparison Study")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to YAML config file")
    args = parser.parse_args()

    exp_cfg = ExperimentConfig.from_yaml(args.config)

    # Create a unique timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = os.path.join(exp_cfg.env.output_dir, f"comparison_study_{timestamp}")
    plots_dir = os.path.join(experiment_root, "plots")
    metrics_dir = os.path.join(experiment_root, "metrics")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Build all SimulationConfig instances from the YAML
    entries = exp_cfg.build_configs(metrics_dir=metrics_dir)

    summary_results = []
    # Track experiment directories for comparison plots
    scenario_experiments = {s.id: [] for s in exp_cfg.scenarios}

    print(f"Starting Topology Comparison Study")
    print(f"Config: {args.config}")
    print(f"Output directory: {experiment_root}")
    print("-" * 50)

    for entry in entries:
        config = entry["config"]
        topo_label = entry["topo_label"]
        scenario_id = entry["scenario_id"]
        scenario_label = entry["scenario_label"]

        print(f"Running: {config.topology.type:22} | {scenario_label:25}", end=" ", flush=True)

        hx = run_experiment(config)

        # Store path for convergence plot
        exp_dir = os.path.join(metrics_dir, config.experiment_name)
        scenario_experiments[scenario_id].append((exp_dir, topo_label))

        # Get global accuracy (consistent across all topologies)
        global_acc = hx[-1].get("test_accuracy", 0.0)
        
        summary_results.append({
            "Topology": topo_label,
            "Scenario": scenario_label,
            "Final Accuracy": global_acc,
            "Metric": "Global"
        })

        # For ensemble models, also track personalized accuracy separately
        is_ensemble = (config.topology.type == "hierarchical_ensemble")
        if is_ensemble and "ensemble_test_accuracy" in hx[-1]:
            pers_acc = hx[-1]["ensemble_test_accuracy"]
            summary_results.append({
                "Topology": f"{topo_label} (Pers.)",
                "Scenario": scenario_label,
                "Final Accuracy": pers_acc,
                "Metric": "Personalized"
            })
        
        print(f"| Accuracy (Global): {global_acc:6.2f}%", end="")
        if is_ensemble and "ensemble_test_accuracy" in hx[-1]:
            print(f" | (Pers.): {hx[-1]['ensemble_test_accuracy']:6.2f}%")
        else:
            print("")

    print("-" * 50)
    print("Generating Visualization Charts...")

    # Generate convergence charts for each scenario
    for scenario in exp_cfg.scenarios:
        exps = scenario_experiments[scenario.id]
        if not exps:
            continue
        
        dirs, labels = zip(*exps)
        plot_path = os.path.join(plots_dir, f"convergence_{scenario.id}.png")
        plot_comparison(dirs, labels, plot_path)

    # Generate summary plot
    df_summary = pd.DataFrame(summary_results)
    summary_plot = os.path.join(plots_dir, "robustness_summary.png")
    
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    sns.barplot(data=df_summary, x="Topology", y="Final Accuracy", hue="Scenario")
    plt.title(f"Robustness Comparison across Topologies (after {exp_cfg.num_rounds} rounds)")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(summary_plot)
    plt.close()
    
    # Save the summary to JSON
    with open(os.path.join(experiment_root, "summary.json"), "w", encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4)
    
    print(f"\nStudy Complete!")
    print(f"Results organized in: {experiment_root}")
    print(f"  - Plots:   {plots_dir}")
    print(f"  - Metrics: {metrics_dir}")

if __name__ == "__main__":
    main()
