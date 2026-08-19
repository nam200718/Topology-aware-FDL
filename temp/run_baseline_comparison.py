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
    "hierarchical":  (HierarchicalTopology, HierarchicalEnsembleEngine),
    "hierarchical_ensemble":  (HierarchicalTopology, HierarchicalEnsembleEngine),
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
    
    defense_mode = topo_params.get("defense_mode", "none")
    if defense_mode != "none":
        from src.defense.config import DefenseConfig
        from src.defense.aggregator import SoftRejectionAggregator
        defense_cfg = DefenseConfig(
            defense_mode=defense_mode,
            norm_bounding_enabled=True,
            hard_rejection_enabled=True,
            hard_rejection_threshold=0.0,
        )
        aggregator = SoftRejectionAggregator(defense_cfg)
    else:
        aggregator = FedAvgAggregator()
        
    engine = EngineClass(main_config, topology, aggregator, device)

    # Hierarchical Partition Injection
    if common_config_dict.get("use_hierarchical_partition", False):
        from temp.hierarchical_partitioner import partition_data_hierarchical
        hp_config = common_config_dict.get("hierarchical_partition_config", {})
        num_clusters = hp_config.get("num_clusters", 5)
        intra_alpha = hp_config.get("intra_alpha", 1.0)
        inter_alpha = hp_config.get("inter_alpha", 0.1)
        
        engine.client_indices, _ = partition_data_hierarchical(
            dataset=engine.train_dataset,
            num_clients=main_config.clients.num_clients,
            num_clusters=num_clusters,
            intra_alpha=intra_alpha,
            inter_alpha=inter_alpha,
            seed=main_config.env.seed,
        )
        engine.client_test_indices, _ = partition_data_hierarchical(
            dataset=engine.test_dataset,
            num_clients=main_config.clients.num_clients,
            num_clusters=num_clusters,
            intra_alpha=intra_alpha,
            inter_alpha=inter_alpha,
            seed=main_config.env.seed + 1,
        )
        from src.data.dataset import ClientDataset
        engine.client_train_datasets = {
            cid: ClientDataset(engine.train_dataset, idxs)
            for cid, idxs in engine.client_indices.items()
        }
        engine.client_test_datasets = {
            cid: ClientDataset(engine.test_dataset, idxs)
            for cid, idxs in engine.client_test_indices.items()
        }
        engine.client_test_indices_t = {
            cid: torch.tensor(idxs, dtype=torch.long, device=engine.device)
            for cid, idxs in engine.client_test_indices.items()
        }
        print(f"  [Hierarchical Partition] clusters={num_clusters}, x={intra_alpha}, β={inter_alpha}")
    
    # Override client config if needed (hierarchical)
    if topo_type == "hierarchical":
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
    byzantine_rates = config.pop("byzantine_rates", [0.0])
    # Extract byzantine_types if provided, otherwise fallback to robustness.byzantine_type or "none"
    default_byz_type = config.get("robustness", {}).get("byzantine_type", "none")
    byzantine_types = config.pop("byzantine_types", [default_byz_type])
    
    out_dir = config["env"].get("output_dir", "./outputs/baseline_comparison")
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    all_metrics = {}
    
    for byz_type in byzantine_types:
        for byz_rate in byzantine_rates:
            # Skip duplicate baseline runs (rate=0.0) for subsequent attack types
            if byz_rate == 0.0 and byz_type != byzantine_types[0]:
                continue
                
            if "robustness" not in config:
                config["robustness"] = {}
            config["robustness"]["byzantine_rate"] = byz_rate
            config["robustness"]["byzantine_type"] = byz_type
            
            for topo in topologies:
                res = run_single_topology(topo, config)
                
                # Format label dynamically
                suffix = ""
                if byz_rate > 0:
                    suffix = f" ({byz_type} {byz_rate})"
                elif len(byzantine_rates) > 1 or len(byzantine_types) > 1:
                    suffix = " (No Attack)"
                    
                label = f"{res['label']}{suffix}"
                
                results.append({
                    "Method": label,
                    "Byz Type": byz_type,
                    "Byz Rate": byz_rate,
                    "Global Acc": res["global_acc"],
                    "Ens Acc": res["ens_acc"]
                })
                all_metrics[label] = res["metrics"]
                
                # Save individual metrics
                topo_safe_name = label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("=", "").replace(".", "_")
                with open(os.path.join(out_dir, f"{topo_safe_name}_metrics.json"), "w") as f:
                    json.dump(res["metrics"], f, indent=4)
            
    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
    print("\nSummary:\n", df)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Subplot 1: Global Test Accuracy
    for label, metrics in all_metrics.items():
        rounds = [m["round"] for m in metrics]
        accs = [m.get("test_accuracy", 0) for m in metrics]
        ax1.plot(rounds, accs, label=label, marker='o', markersize=3)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Global Model Accuracy")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Ensemble/Personalized Accuracy  
    for label, metrics in all_metrics.items():
        ens_accs = [m.get("ensemble_test_accuracy", None) for m in metrics]
        if any(a is not None and a > 0 for a in ens_accs):
            rounds = [m["round"] for m in metrics]
            accs = [a if a else 0 for a in ens_accs]
            ax2.plot(rounds, accs, label=label, marker='s', markersize=3)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Ensemble / Personalized Accuracy")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_accuracy_curves.png"), dpi=150)
    plt.close()
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="temp/baseline_comparison.yaml")
    args = parser.parse_args()
    
    run_all(args.config)
