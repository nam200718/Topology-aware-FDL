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
    exp_cfg = ExperimentConfig.from_yaml(config_path)
    entries = exp_cfg.build_configs()

    print(f"Loaded config: {config_path}")
    print(f"Experiment type: {exp_cfg.experiment_type}")
    print(f"Total runs: {len(entries)}")
    print("-" * 50)

    all_histories = []
    for entry in entries:
        config = entry["config"]
        label = entry["topo_label"]
        print(f"\nRunning: {label} ({config.experiment_name})")
        
        hx = run_experiment(config)
        all_histories.append({"entry": entry, "history": hx})

        final_acc = hx[-1].get("test_accuracy", 0.0)
        is_ensemble = (config.topology.type == "hierarchical_ensemble")
        if is_ensemble and "ensemble_test_accuracy" in hx[-1]:
            final_acc = hx[-1]["ensemble_test_accuracy"]
        print(f"  Final Accuracy: {final_acc:.2f}%")

    print("-" * 50)
    print("All experiments complete!")
    return all_histories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedlEARNING Simulation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()
    
    if args.config:
        run_from_config(args.config)
    else:
        # Default single run (smoke test)
        config = SimulationConfig(
            experiment_name="smoke_test_mnist",
            num_rounds=5,
            topology=TopologyConfig(type="star"),
            clients=ClientConfig(num_clients=5, model_dim=0, local_lr=0.01, local_steps=1),
        )
        run_experiment(config)
