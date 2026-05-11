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

def main():
    # Create a unique timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = os.path.join("./outputs", f"byzantine_matrix_{timestamp}")
    plots_dir = os.path.join(experiment_root, "plots")
    metrics_dir = os.path.join(experiment_root, "metrics")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    topologies = [
        ("star", "Star", {}),
        ("star_randomized", "Star (Rand)", {}),
        ("ring", "Ring", {}),
        ("gossip", "Gossip", {"degree_k": 3}),
        ("hierarchical", "Hierarchical", {"num_clusters": 3}),
        ("hierarchical_ensemble", "Ensemble", {"num_clusters": 3}),
        ("layered", "Layered", {"layers": [15, 5, 1]})
    ]
    
    byzantine_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    results = []
    
    # Common client config
    client_config_base = {
        "num_clients": 15,
        "local_lr": 0.05,
        "local_steps": 1,
    }
    
    num_rounds = 20 # Increased for better convergence
    env_config = EnvironmentConfig(
        seed=42, 
        output_dir=metrics_dir,
        train_subset=None, # Use full training set
        test_subset=None   # Use full test set
    )
    
    print(f"Starting Byzantine Robustness Matrix")
    print(f"Output directory: {experiment_root}")
    print(f"Byzantine Rates: {byzantine_rates}")
    print("-" * 65)
    
    for topo_type, topo_label, params in topologies:
        for rate in byzantine_rates:
            exp_name = f"{topo_type}_byz_{int(rate*100)}"
            print(f"Topo: {topo_label:15} | Byz Rate: {rate:3.1f}", end=" ", flush=True)
            
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
                    byzantine_rate=rate,
                    byzantine_type="label_flip"
                ),
                non_iid=NonIIDConfig(enabled=False) # Keep it IID to isolate Byzantine impact
            )
            
            import contextlib
            with contextlib.redirect_stdout(None):
                hx = run_experiment(config)
            
            final_acc = hx[-1].get("test_accuracy", 0.0)
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
    plt.title(f"Byzantine Robustness Matrix (after {num_rounds} rounds)")
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
