import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import SimulationConfig, TopologyConfig, ClientConfig, EnvironmentConfig, RobustnessConfig
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
        device = torch.accelerator.current_accelerator().type
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        # Check for DirectML (common for AMD on Windows)
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            device = "cpu"

    print(f"Using device: {device}")
    engine = engine_cls(config=config, topology=topology, aggregator=aggregator, device=device)
    # Check invariants right after engine mapping builds topology in its __init__
    check_invariants(topology, config)
    
    engine.run()
    return engine.metrics.get_history()

def run_matrix():
    topologies = ["star", "ring", "gossip", "hierarchical", "hierarchical_ensemble", "layered"]
    # Matrix of byzantine failures instead of stragglers specifically, or both
    failure_rates = [0.0, 0.1, 0.3]
    
    all_results = {}
    
    base_output = "./outputs/matrix"
    os.makedirs(base_output, exist_ok=True)
    
    for topo in topologies:
        all_results[topo] = {}
        for rate in failure_rates:
            exp_name = f"{topo}_byz_{rate}"
            print(f"--- Running {exp_name} ---")
            
            topo_params = {}
            if topo == "layered":
                topo_params = {"layers": [10, 4, 2, 1]}
            
            config = SimulationConfig(
                experiment_name=f"matrix/{exp_name}",
                num_rounds=5,
                env=EnvironmentConfig(seed=42, output_dir="./outputs"),
                topology=TopologyConfig(type=topo, params=topo_params),
                # Scale down for MNIST execution: 10 clients, 2 local steps, bigger LR
                clients=ClientConfig(num_clients=10, model_dim=0, local_lr=0.01, local_steps=2),
                robustness=RobustnessConfig(byzantine_rate=rate)
            )
            
            hx = run_experiment(config)
            all_results[topo][rate] = hx
            
    # Plot results
    plot_matrix(all_results, base_output)
    print("Matrix execution completed!")

def plot_matrix(all_results, out_dir):
    sns.set_style("whitegrid")
    
    topologies = list(all_results.keys())
    num_topos = len(topologies)
    fig, axes = plt.subplots(1, num_topos, figsize=(5 * num_topos, 5), sharey=True)
    
    for i, topo in enumerate(topologies):
        ax = axes[i]
        for rate, hx in all_results[topo].items():
            rounds = [d["round"] for d in hx]
            vals = [d.get("test_accuracy", 0.0) for d in hx]
            ax.plot(rounds, vals, label=f"Byz Rate {rate}")
            
        ax.set_title(f"{topo.capitalize()} Topology")
        ax.set_xlabel("Round")
        if i == 0:
            ax.set_ylabel("Test Accuracy (%)")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_matrix.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedlEARNING Simulation")
    parser.add_argument("--matrix", action="store_true", help="Run the experiment matrix")
    args = parser.parse_args()
    
    if args.matrix:
        run_matrix()
    else:
        # Default single run
        config = SimulationConfig(
            experiment_name="smoke_test_mnist",
            num_rounds=5,
            topology=TopologyConfig(type="star"),
            clients=ClientConfig(num_clients=5, model_dim=0, local_lr=0.01, local_steps=1),
        )
        run_experiment(config)
