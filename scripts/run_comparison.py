import os
import sys
from datetime import datetime

# Add the project root to the path so we can import from src and main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import SimulationConfig, TopologyConfig, ClientConfig, EnvironmentConfig, RobustnessConfig, NonIIDConfig
from main import run_experiment
from src.utils.visualizer import plot_comparison

def main():
    # Create a unique timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = os.path.join("./outputs", f"comparison_study_{timestamp}")
    plots_dir = os.path.join(experiment_root, "plots")
    metrics_dir = os.path.join(experiment_root, "metrics")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    topologies = [
        ("star", {}),
        ("star_randomized", {}),
        ("ring", {}),
        ("gossip", {"degree_k": 3}),
        ("hierarchical", {"num_clusters": 3}),
        ("hierarchical_ensemble", {"num_clusters": 3}),
        ("layered", {"layers": [15, 5, 1]})
    ]
    
    scenarios = [
        ("iid", "IID (Baseline)", {"non_iid": False, "byzantine_rate": 0.0}),
        ("non_iid", "Non-IID", {"non_iid": True, "byzantine_rate": 0.0}),
        ("attack", "Byzantine Attack (20%)", {"non_iid": False, "byzantine_rate": 0.2})
    ]
    
    summary_results = []
    
    # Common client config
    client_config_base = {
        "num_clients": 15,
        "local_lr": 0.05,
        "local_steps": 1,
    }
    
    num_rounds = 15
    env_config = EnvironmentConfig(
        seed=42, 
        output_dir=metrics_dir, # Save metrics in the metrics dir
        train_subset=10000,
        test_subset=2000
    )
    
    # Track experiment directories for comparison plots
    scenario_experiments = {sid: [] for sid, _, _ in scenarios}
    
    print(f"Starting Topology Comparison Study")
    print(f"Output directory: {experiment_root}")
    print("-" * 50)
    
    for topo_type, params in topologies:
        for scenario_id, scenario_label, scenario_params in scenarios:
            exp_name = f"{topo_type}_{scenario_id}"
            print(f"Running: {topo_type:22} | {scenario_label:25}", end=" ", flush=True)
            
            # Adjust config for ensemble
            is_ensemble = (topo_type == "hierarchical_ensemble")
            
            config = SimulationConfig(
                experiment_name=exp_name,
                num_rounds=num_rounds,
                env=env_config,
                topology=TopologyConfig(type=topo_type, params=params),
                clients=ClientConfig(
                    **client_config_base,
                    use_ensemble=is_ensemble,
                    hierarchical_ensemble=is_ensemble,
                ),
                robustness=RobustnessConfig(
                    byzantine_rate=scenario_params["byzantine_rate"],
                    byzantine_type="label_flip"
                ),
                non_iid=NonIIDConfig(
                    enabled=scenario_params["non_iid"],
                    num_shards=20
                )
            )
            
            # Silence internal engine prints for cleaner output
            import contextlib
            with contextlib.redirect_stdout(None):
                hx = run_experiment(config)
            
            # Store path for convergence plot
            exp_dir = os.path.join(metrics_dir, exp_name)
            scenario_experiments[scenario_id].append((exp_dir, topo_type.replace("_", " ").capitalize()))
            
            # Get final accuracy
            final_acc = hx[-1].get("test_accuracy", 0.0)
            if is_ensemble and "ensemble_test_accuracy" in hx[-1]:
                final_acc = hx[-1]["ensemble_test_accuracy"]
                
            summary_results.append({
                "Topology": topo_type.replace("_", " ").capitalize(),
                "Scenario": scenario_label,
                "Final Accuracy": final_acc
            })
            print(f"| Accuracy: {final_acc:6.2f}%")
            
    print("-" * 50)
    print("Generating Visualization Charts...")
    
    # Generate convergence charts for each scenario
    for scenario_id, scenario_label, _ in scenarios:
        exps = scenario_experiments[scenario_id]
        if not exps: continue
        
        dirs, labels = zip(*exps)
        plot_path = os.path.join(plots_dir, f"convergence_{scenario_id}.png")
        plot_comparison(dirs, labels, plot_path)

    # Generate summary plot
    df_summary = pd.DataFrame(summary_results)
    summary_plot = os.path.join(plots_dir, "robustness_summary.png")
    
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    sns.barplot(data=df_summary, x="Topology", y="Final Accuracy", hue="Scenario")
    plt.title(f"Robustness Comparison across Topologies (after {num_rounds} rounds)")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(summary_plot)
    plt.close()
    
    # Save the summary to JSON
    with open(os.path.join(experiment_root, "summary.json"), "w") as f:
        json.dump(summary_results, f, indent=4)
    
    print(f"\nStudy Complete!")
    print(f"Results organized in: {experiment_root}")
    print(f"  - Plots:   {plots_dir}")
    print(f"  - Metrics: {metrics_dir}")

if __name__ == "__main__":
    main()
