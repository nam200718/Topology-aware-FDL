import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.topologies.star import StarTopology
from src.topologies.hierarchical import HierarchicalTopology
from src.core.centralized_engine import CentralizedEngine
from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine
from src.config import ExperimentConfig
from src.core.aggregator import FedAvgAggregator

from temp.ring import RingTopology
from temp.gossip import GossipTopology
from temp.decentralized_engine import DecentralizedEngine

TOPOLOGY_REGISTRY = {
    "star":      (StarTopology,         CentralizedEngine),
    "ring":      (RingTopology,         DecentralizedEngine),
    "gossip":    (GossipTopology,       DecentralizedEngine),
    "hier_agg":  (HierarchicalTopology, HierarchicalEnsembleEngine),
    "hier_ens":  (HierarchicalTopology, HierarchicalEnsembleEngine),
}

def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_single_topology(topo_config: dict, common_config_dict: dict) -> dict:
    topo_type = topo_config["type"]
    label = topo_config["label"]
    print(f"\n{'='*50}\nRunning Topology: {label}\n{'='*50}")

    # Build MainConfig
    config_dict = dict(common_config_dict)
    config_dict["topologies"] = [topo_config]
    config_dict["experiment_type"] = "single"
    
    # We must construct a MainConfig from the dict
    # But since MainConfig usually takes a path, we can either write a temp yaml or construct directly.
    # To keep it simple, let's write a temp config for this run.
    temp_path = "temp_run_config.yaml"
    with open(temp_path, "w") as f:
        yaml.dump(config_dict, f)
        
    exp_config = ExperimentConfig.from_yaml(temp_path)
    main_config = exp_config.build_configs()[0]["config"]
    os.remove(temp_path)

    # Instantiate
    TopoClass, EngineClass = TOPOLOGY_REGISTRY[topo_type]
    
    # Init Topology
    topo_params = topo_config.get("params", {})
    if topo_type == "gossip":
        topology = TopoClass(degree_k=topo_params.get("degree_k", 3))
    else:
        topology = TopoClass()
        
    # Init Engine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aggregator = FedAvgAggregator()
    engine = EngineClass(main_config, topology, aggregator, device)
    
    # Override client config if needed (hier_agg)
    if topo_type == "hier_agg":
        main_config.clients.use_ensemble = False
        main_config.clients.hierarchical_ensemble = False
        main_config.clients.compute_optimization_mode = "none"

    engine.run()
    
    # Extract metrics
    metrics = engine.metrics.history
    final_global_acc = metrics[-1].get("test_accuracy", 0.0) if metrics else 0.0
    final_ens_acc = metrics[-1].get("ensemble_test_accuracy", final_global_acc) if metrics else 0.0
    
    # For personalized, our codebase might log it differently or APFL/Ditto logs it. 
    # For this script we will return what we have.
    
    return {
        "label": label,
        "global_acc": final_global_acc,
        "ens_acc": final_ens_acc,
        "metrics": metrics
    }

def run_all(config_path: str):
    config = load_yaml_config(config_path)
    topologies = config.pop("topologies")
    
    out_dir = config["env"].get("output_dir", "./outputs/baseline_comparison")
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    all_metrics = {}
    
    for topo in topologies:
        res = run_single_topology(topo, config)
        results.append({
            "Method": res["label"],
            "Global Acc": res["global_acc"],
            "Ens Acc": res["ens_acc"]
        })
        all_metrics[res["label"]] = res["metrics"]
        
        # Save individual metrics
        topo_safe_name = res["label"].replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("=", "")
        with open(os.path.join(out_dir, f"{topo_safe_name}_metrics.json"), "w") as f:
            json.dump(res["metrics"], f, indent=4)
            
    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
    print("\nSummary:\n", df)
    
    # Plot
    plt.figure(figsize=(10, 6))
    for label, metrics in all_metrics.items():
        rounds = [m["round"] for m in metrics]
        # Use ensemble acc if available and greater than 0, otherwise global acc
        accs = [m.get("ensemble_test_accuracy", m.get("test_accuracy", 0)) for m in metrics]
        plt.plot(rounds, accs, label=label)
        
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Baseline Comparison (Alsaluli specs)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "comparison_accuracy_curves.png"))
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="temp/baseline_comparison.yaml")
    args = parser.parse_args()
    
    run_all(args.config)
