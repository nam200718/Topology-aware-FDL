import os
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import SimulationConfig, TopologyConfig, ClientConfig, EnvironmentConfig, RobustnessConfig, NonIIDConfig
from main import run_experiment
from src.utils.visualizer import plot_comparison

def main():
    topologies = [
        ("star", {}),
        ("ring", {}),
        ("gossip", {"degree_k": 3}),
        ("hierarchical", {"num_clusters": 3}),
        ("hierarchical_ensemble", {"num_clusters": 3})
    ]
    
    scenarios = [
        ("iid", "IID (Baseline)", {"non_iid": False, "byzantine_rate": 0.0}),
        ("non_iid", "Non-IID", {"non_iid": True, "byzantine_rate": 0.0}),
        ("attack", "Byzantine Attack (20%)", {"non_iid": False, "byzantine_rate": 0.2})
    ]
    
    experiment_base = "robustness_comparison"
    summary_results = []
    
    # Common client config
    client_config_base = {
        "num_clients": 15,
        "local_lr": 0.02,
        "local_steps": 1,
    }
    
    num_rounds = 5
    
    for topo_type, params in topologies:
        for scenario_id, scenario_label, scenario_params in scenarios:
            exp_name = f"{experiment_base}/{topo_type}_{scenario_id}"
            print(f"\n>>> Running Experiment: {topo_type} | Scenario: {scenario_label} <<<")
            
            # Adjust config for ensemble
            is_ensemble = (topo_type == "hierarchical_ensemble")
            
            config = SimulationConfig(
                experiment_name=exp_name,
                num_rounds=num_rounds,
                env=EnvironmentConfig(seed=42, output_dir="./outputs"),
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
                    num_shards=30 # 2 shards per client approx
                )
            )
            
            hx = run_experiment(config)
            
            # Get final accuracy
            final_acc = hx[-1].get("test_accuracy", 0.0)
            if is_ensemble and "ensemble_test_accuracy" in hx[-1]:
                final_acc = hx[-1]["ensemble_test_accuracy"]
                
            summary_results.append({
                "Topology": topo_type.replace("_", " ").capitalize(),
                "Scenario": scenario_label,
                "Final Accuracy": final_acc
            })
            
    # Generate summary plot
    df_summary = pd.DataFrame(summary_results)
    output_plot = os.path.join("./outputs", experiment_base, "robustness_summary.png")
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    sns.barplot(data=df_summary, x="Topology", y="Final Accuracy", hue="Scenario")
    plt.title(f"Robustness Comparison across Topologies (after {num_rounds} rounds)")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()
    
    # Also save the results to JSON
    with open(os.path.join("./outputs", experiment_base, "summary.json"), "w") as f:
        json.dump(summary_results, f, indent=4)
    
    print(f"\nRobustness comparison complete!")
    print(f"Summary chart saved to: {output_plot}")
    print(f"Detailed results saved to: {os.path.join('./outputs', experiment_base, 'summary.json')}")

if __name__ == "__main__":
    main()
