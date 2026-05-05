import os
import sys

# Add the project root to the path so we can import from src and main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import SimulationConfig, TopologyConfig, ClientConfig, EnvironmentConfig
from main import run_experiment
from src.utils.visualizer import plot_experiment_results

def main():
    # Configure the hierarchical ensemble experiment
    config = SimulationConfig(
        experiment_name="hierarchical_ensemble_test",
        num_rounds=10,
        env=EnvironmentConfig(seed=42, output_dir="./outputs"),
        topology=TopologyConfig(
            type="hierarchical_ensemble", 
            params={"num_clusters": 3}
        ),
        clients=ClientConfig(
            num_clients=15, 
            local_lr=0.01, 
            local_steps=2,
            use_ensemble=True,           # Enable local model
            hierarchical_ensemble=True,   # Enable parent model training
            ensemble_alpha=0.33,         # Local model weight
            ensemble_beta=0.33           # Parent model weight
        )
    )
    
    print("Starting Hierarchical Ensemble Experiment...")
    print(f"Topology: {config.topology.type}")
    print(f"Clients: {config.clients.num_clients}, Clusters: {config.topology.params['num_clusters']}")
    
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
