import torch
from typing import Union
from src.config import SimulationConfig
from src.topologies.star import StarTopology
from src.topologies.hierarchical import HierarchicalTopology
from src.topologies.checks import (
    check_star_invariant,
    check_hierarchical_invariant,
)
from src.utils.device import resolve_device

from src.core.aggregator import FedAvgAggregator
from src.core.centralized_engine import CentralizedEngine
from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine


def detect_device() -> str:
    """Detect a verified accelerator (DirectML/CUDA/MPS) or fall back to CPU.

    Delegates to resolve_device(), which self-verifies the accelerator op
    suite and honors the HEP_FORCE_DEVICE environment variable.
    """
    dev = resolve_device()
    return dev if isinstance(dev, str) else str(dev)


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
