import os
import torch
from src.config import SimulationConfig, TopologyConfig, ClientConfig, EnvironmentConfig
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
    
    experiment_base = "comparison_study"
    results_dirs = []
    labels = []
    
    # Common client config
    client_config_base = {
        "num_clients": 20,
        "local_lr": 0.01,
        "local_steps": 2,
    }
    
    for topo_type, params in topologies:
        exp_name = f"{experiment_base}/{topo_type}"
        print(f"\n>>> Running Experiment: {topo_type} <<<")
        
        # Adjust config for ensemble
        is_ensemble = (topo_type == "hierarchical_ensemble")
        
        config = SimulationConfig(
            experiment_name=exp_name,
            num_rounds=10,
            env=EnvironmentConfig(seed=42, output_dir="./outputs"),
            topology=TopologyConfig(type=topo_type, params=params),
            clients=ClientConfig(
                **client_config_base,
                use_ensemble=is_ensemble,
                hierarchical_ensemble=is_ensemble,
                ensemble_alpha=0.33 if is_ensemble else 0.5,
                ensemble_beta=0.33 if is_ensemble else 0.0
            )
        )
        
        run_experiment(config)
        
        results_dirs.append(os.path.join("./outputs", exp_name))
        labels.append(topo_type.replace("_", " ").capitalize())
        
    # Generate comparison plot
    output_plot = os.path.join("./outputs", experiment_base, "topology_comparison.png")
    plot_comparison(results_dirs, labels, output_plot)
    
    print(f"\nComparison complete! Summary chart saved to: {output_plot}")

if __name__ == "__main__":
    main()
