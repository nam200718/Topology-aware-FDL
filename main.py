import os
import sys
import argparse

# --- Virtual environment check ---
_in_venv = os.environ.get("VIRTUAL_ENV") or (sys.prefix != sys.base_prefix)
if not _in_venv:
    _project_root = os.path.dirname(os.path.abspath(__file__))
    print("ERROR: Virtual environment is not activated.", file=sys.stderr)
    print(f"  Run:  source {_project_root}/.venv/bin/activate", file=sys.stderr)
    print(f"  Then: python {' '.join(sys.argv)}", file=sys.stderr)
    sys.exit(1)

from typing import List
from src.config import SimulationConfig, TopologyConfig, ClientConfig, EnvironmentConfig, RobustnessConfig, TopologyType, ExperimentConfig
from src.topologies.star import StarTopology
from src.topologies.ring import RingTopology
from src.topologies.gossip import GossipTopology
from src.topologies.hierarchical import HierarchicalTopology
from src.topologies.layered import LayeredTopology
from src.topologies.checks import check_star_invariant, check_ring_invariant, check_gossip_invariant, check_hierarchical_invariant, check_layered_invariant

from src.core.aggregator import FedAvgAggregator
from src.core.centralized_engine import CentralizedEngine
from src.core.decentralized_engine import DecentralizedEngine
from src.core.hierarchical_engine import HierarchicalEngine
from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine
from src.core.layered_engine import LayeredEngine
from src.utils.random import set_seed

def build_topology_and_engine(config: SimulationConfig):
    if config.topology.type == "star" or config.topology.type == "star_randomized":
        topology = StarTopology()
        engine_cls = CentralizedEngine
    elif config.topology.type == "ring":
        topology = RingTopology()
        engine_cls = DecentralizedEngine
    elif config.topology.type == "gossip":
        degree = config.topology.params.get("degree_k", 3)
        topology = GossipTopology(degree_k=degree)
        engine_cls = DecentralizedEngine
    elif config.topology.type == "hierarchical":
        clusters = config.topology.params.get("num_clusters", 5)
        topology = HierarchicalTopology(num_clusters=clusters)
        engine_cls = HierarchicalEngine
    elif config.topology.type == "hierarchical_ensemble":
        clusters = config.topology.params.get("num_clusters", 5)
        topology = HierarchicalTopology(num_clusters=clusters)
        engine_cls = HierarchicalEnsembleEngine
    elif config.topology.type == "layered":
        layers = config.topology.params.get("layers", [config.clients.num_clients, 4, 2, 1])
        gossip_steps = config.topology.params.get("gossip_steps", 1)
        topology = LayeredTopology(layers=layers, gossip_steps=gossip_steps)
        engine_cls = LayeredEngine
    else:
        raise ValueError(f"Unknown topology {config.topology.type}")
        
    return topology, engine_cls

def check_invariants(topology, config):
    if config.topology.type == "star":
        check_star_invariant(topology, config.clients.num_clients)
    elif config.topology.type == "ring":
        check_ring_invariant(topology, config.clients.num_clients)
    elif config.topology.type == "gossip":
        check_gossip_invariant(topology, config.clients.num_clients)
    elif config.topology.type == "hierarchical" or config.topology.type == "hierarchical_ensemble":
        check_hierarchical_invariant(topology, config.clients.num_clients)
    elif config.topology.type == "layered":
        check_layered_invariant(topology, config.clients.num_clients)

def run_experiment(config: SimulationConfig):
    set_seed(config.env.seed)
    
    topology, engine_cls = build_topology_and_engine(config)
    aggregator = FedAvgAggregator()
    
    import torch
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        acc = torch.accelerator.current_accelerator()
        if acc is not None:
            device = acc.type
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        # Check for DirectML (common for AMD on Windows)
        try:
            import torch_directml # type: ignore
            device = torch_directml.device()
        except ImportError:
            device = "cpu"

    print(f"Using device: {device}")
    engine = engine_cls(config=config, topology=topology, aggregator=aggregator, device=device)
    # Check invariants right after engine mapping builds topology in its __init__
    check_invariants(topology, config)
    
    engine.run()
    return engine.metrics.get_history()

def run_from_config(config_path: str):
    """Load a YAML config file and run the appropriate experiment(s)."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from datetime import datetime
    import json
    from src.utils.visualizer import plot_experiment_results, plot_comparison

    exp_cfg = ExperimentConfig.from_yaml(config_path)
    
    # Create unique timestamped directory for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if exp_cfg.experiment_type == "single":
        dir_prefix = "single_run"
    elif exp_cfg.experiment_type == "comparison":
        dir_prefix = "comparison_study"
    elif exp_cfg.experiment_type == "byzantine_matrix":
        dir_prefix = "byzantine_matrix"
    else:
        dir_prefix = "experiment"
        
    experiment_root = os.path.join(exp_cfg.env.output_dir, f"{dir_prefix}_{timestamp}")
    plots_dir = os.path.join(experiment_root, "plots")
    metrics_dir = os.path.join(experiment_root, "metrics")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    entries = exp_cfg.build_configs(metrics_dir=metrics_dir)

    print(f"Loaded config: {config_path}")
    print(f"Experiment type: {exp_cfg.experiment_type}")
    print(f"Total runs: {len(entries)}")
    print(f"Output directory: {experiment_root}")
    print("-" * 65)

    all_histories = []
    
    if exp_cfg.experiment_type == "single":
        config = entries[0]["config"]
        label = entries[0]["topo_label"]
        print(f"\nRunning: {label} ({config.experiment_name})")
        hx = run_experiment(config)
        all_histories.append({"entry": entries[0], "history": hx})
        
        final_acc = hx[-1].get("test_accuracy", 0.0)
        is_ensemble = (config.topology.type == "hierarchical_ensemble")
        if is_ensemble and "ensemble_test_accuracy" in hx[-1]:
            final_acc = hx[-1]["ensemble_test_accuracy"]
        print(f"  Final Accuracy: {final_acc:.2f}%")
        
        # Plotting
        metrics_json_path = os.path.join(metrics_dir, config.experiment_name, "metrics.json")
        plot_experiment_results(metrics_json_path, output_dir=plots_dir)
        
    elif exp_cfg.experiment_type == "comparison":
        summary_results = []
        scenario_experiments = {s.id: [] for s in exp_cfg.scenarios}
        
        for entry in entries:
            config = entry["config"]
            topo_label = entry["topo_label"]
            scenario_id = entry["scenario_id"]
            scenario_label = entry["scenario_label"]
            
            print(f"Running: {config.topology.type:22} | {scenario_label:25}", end=" ", flush=True)
            hx = run_experiment(config)
            all_histories.append({"entry": entry, "history": hx})
            
            exp_dir = os.path.join(metrics_dir, config.experiment_name)
            scenario_experiments[scenario_id].append((exp_dir, topo_label))
            
            global_acc = hx[-1].get("test_accuracy", 0.0)
            summary_results.append({
                "Topology": topo_label,
                "Scenario": scenario_label,
                "Final Accuracy": global_acc,
                "Metric": "Global"
            })
            
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
        ax = sns.barplot(data=df_summary, x="Topology", y="Final Accuracy", hue="Scenario")
        plt.title(f"Robustness Comparison across Topologies (after {exp_cfg.num_rounds} rounds)")
        plt.ylabel("Test Accuracy (%)")
        plt.ylim(0, 110)
        
        for p in ax.patches:
            if isinstance(p, mpatches.Rectangle):
                height = p.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.1f}%', 
                               (p.get_x() + p.get_width() / 2., height), 
                               ha='center', va='center', 
                               xytext=(0, 9), 
                               textcoords='offset points',
                               fontsize=8)
                           
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Scenario", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(summary_plot, bbox_inches='tight')
        plt.close()
        
        # Save summary files
        with open(os.path.join(experiment_root, "summary.json"), "w", encoding='utf-8') as f:
            json.dump(summary_results, f, indent=4)
        df_summary.to_csv(os.path.join(experiment_root, "comparison_results.csv"), index=False)
        
    elif exp_cfg.experiment_type == "byzantine_matrix":
        results = []
        for entry in entries:
            config = entry["config"]
            topo_label = entry["topo_label"]
            rate = entry["byzantine_rate"]
            
            print(f"Topo: {topo_label:15} | Byz Rate: {rate:3.1f}", end=" ", flush=True)
            hx = run_experiment(config)
            all_histories.append({"entry": entry, "history": hx})
            
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
        plt.close()
        
    print("-" * 65)
    print("All experiments complete!")
    print(f"Results organized in: {experiment_root}")
    return all_histories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedlEARNING Simulation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"], help="Dataset to use (mnist or cifar10)")
    args = parser.parse_args()
    
    if args.config:
        run_from_config(args.config)
    else:
        # Default single run (smoke test)
        config = SimulationConfig(
            experiment_name=f"smoke_test_{args.dataset}",
            num_rounds=5,
            env=EnvironmentConfig(dataset=args.dataset),
            topology=TopologyConfig(type="star"),
            clients=ClientConfig(num_clients=5, model_dim=0, local_lr=0.01, local_steps=1),
        )
        run_experiment(config)

