import torch
try:
    import torch_directml  # type: ignore
except ImportError:
    torch_directml = None
from typing import Union
from src.config import SimulationConfig
from src.topologies.star import StarTopology
from src.topologies.ring import RingTopology
from src.topologies.gossip import GossipTopology
from src.topologies.hierarchical import HierarchicalTopology
from src.topologies.layered import LayeredTopology
from src.topologies.checks import (
    check_star_invariant,
    check_ring_invariant,
    check_gossip_invariant,
    check_hierarchical_invariant,
    check_layered_invariant,
)

from src.core.aggregator import FedAvgAggregator
from src.core.centralized_engine import CentralizedEngine
from src.core.decentralized_engine import DecentralizedEngine
from src.core.hierarchical_engine import HierarchicalEngine
from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine
from src.core.layered_engine import LayeredEngine


def detect_device() -> str:
    """Detect available accelerator hardware (CUDA, MPS, DirectML) or fallback to CPU."""
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        acc = torch.accelerator.current_accelerator()
        if acc is not None:
            return acc.type
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        if torch_directml is not None:
            try:
                if torch_directml.is_available():
                    return str(torch_directml.device())
            except Exception:
                pass
        return "cpu"


class TopologyEngineFactory:
    """Factory for instantiating topologies and matching simulation engines."""

    @staticmethod
    def build(config: SimulationConfig):
        topo_type = config.topology.type
        params = config.topology.params

        if topo_type in ("star", "star_randomized"):
            topology = StarTopology()
            engine_cls = CentralizedEngine
        elif topo_type == "ring":
            topology = RingTopology()
            engine_cls = DecentralizedEngine
        elif topo_type == "gossip":
            degree = params.get("degree_k", 3)
            topology = GossipTopology(degree_k=degree)
            engine_cls = DecentralizedEngine
        elif topo_type == "hierarchical":
            clusters = params.get("num_clusters", 5)
            topology = HierarchicalTopology(num_clusters=clusters)
            engine_cls = HierarchicalEngine
        elif topo_type == "hierarchical_ensemble":
            clusters = params.get("num_clusters", 5)
            topology = HierarchicalTopology(num_clusters=clusters)
            # engine_cls = HierarchicalEnsembleEngine
            # --- Coordinate the defense model ---

            # Check if defense is requested via topology params
            defense_mode = params.get("defense_mode", "none") 
            if defense_mode != "none":
                from src.defense.engine import DefendedEnsembleEngine
                engine_cls = DefendedEnsembleEngine 
            else:
                engine_cls = HierarchicalEnsembleEngine
            
            #
        elif topo_type == "layered":
            layers = params.get("layers", [config.clients.num_clients, 4, 2, 1])
            gossip_steps = params.get("gossip_steps", 1)
            topology = LayeredTopology(layers=layers, gossip_steps=gossip_steps)
            engine_cls = LayeredEngine
        else:
            raise ValueError(f"Unknown topology type: {topo_type}")

        return topology, engine_cls


def build_topology_and_engine(config: SimulationConfig):
    """Facade helper for backward compatibility."""
    return TopologyEngineFactory.build(config)


def check_invariants(topology, config: SimulationConfig):
    """Check graph and structural invariants for a constructed topology."""
    topo_type = config.topology.type
    num_clients = config.clients.num_clients

    if topo_type == "star":
        check_star_invariant(topology, num_clients)
    elif topo_type == "ring":
        check_ring_invariant(topology, num_clients)
    elif topo_type == "gossip":
        check_gossip_invariant(topology, num_clients)
    elif topo_type in ("hierarchical", "hierarchical_ensemble"):
        check_hierarchical_invariant(topology, num_clients)
    elif topo_type == "layered":
        check_layered_invariant(topology, num_clients)
