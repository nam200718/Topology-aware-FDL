import torch
try:
    import torch_directml  # type: ignore
except ImportError:
    torch_directml = None
from typing import Union
from src.config import SimulationConfig
from src.topologies.star import StarTopology
from src.topologies.hierarchical import HierarchicalTopology
from src.topologies.checks import (
    check_star_invariant,
    check_hierarchical_invariant,
)

from src.core.aggregator import FedAvgAggregator
from src.core.centralized_engine import CentralizedEngine
from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine


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
                    dev = torch_directml.device()
                    return dev if isinstance(dev, str) else str(dev)
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
        elif topo_type == "hierarchical_ensemble":
            clusters = params.get("num_clusters", 3)
            topology = HierarchicalTopology(num_clusters=clusters)
            defense_mode = params.get("defense_mode", "none") 
            if defense_mode != "none":
                from src.defense.engine import DefendedEnsembleEngine
                engine_cls = DefendedEnsembleEngine 
            else:
                engine_cls = HierarchicalEnsembleEngine
        else:
            raise ValueError(f"Unknown or unsupported topology type: {topo_type}")

        return topology, engine_cls


def build_topology_and_engine(config: SimulationConfig):
    """Facade helper for backward compatibility."""
    return TopologyEngineFactory.build(config)


def check_invariants(topology, config: SimulationConfig):
    """Check graph and structural invariants for a constructed topology."""
    topo_type = config.topology.type
    num_clients = config.clients.num_clients

    if topo_type in ("star", "star_randomized"):
        check_star_invariant(topology, num_clients)
    elif topo_type == "hierarchical_ensemble":
        check_hierarchical_invariant(topology, num_clients)
