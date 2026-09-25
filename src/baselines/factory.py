"""Factory for constructing topologies, aggregators, and simulation engines for all baselines.

Supports:
1. FedAvg
2. FedProx
3. Multi-Krum
4. SCAFFOLD
5. Ditto
6. Proposed Topo (HEP / Defended H-ResFL)
"""
from typing import Tuple, Optional, Any, Dict
import torch

from src.config import SimulationConfig
from src.topologies.star import StarTopology
from src.topologies.hierarchical import HierarchicalTopology
from src.topologies.checks import check_star_invariant, check_hierarchical_invariant

from src.core.aggregator import FedAvgAggregator
from src.core.centralized_engine import CentralizedEngine
from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine
from src.utils.device import resolve_device

from src.baselines.fedprox_updater import FedProxUpdater
from src.baselines.multi_krum_aggregator import MultiKrumAggregator
from src.baselines.scaffold_engine import ScaffoldEngine


class FedProxEngine(CentralizedEngine):
    """CentralizedEngine equipped with FedProxUpdater."""

    def __init__(self, config, topology, aggregator, device="cpu"):
        super().__init__(config, topology, aggregator, device)
        model_name = getattr(config.clients, "model_name", "simple_cnn")
        self.updater = FedProxUpdater(
            device=self.device,
            in_channels=self.in_channels,
            model_name=model_name,
            num_classes=self.num_classes,
        )


def detect_accelerator() -> str:
    """Detect available accelerator or fall back to CPU."""
    dev = resolve_device()
    return dev if isinstance(dev, str) else str(dev)


def build_baseline_engine(
    config: SimulationConfig,
    method_id: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """Construct (topology, aggregator, engine) for a baseline experiment.

    Args:
        config: Simulation configuration.
        method_id: One of 'fedavg', 'fedprox', 'multikrum', 'scaffold', 'ditto', 'topo', 'topo_defended'.
                   If None, deduced from config.
        device: Device string ('cuda', 'cpu', 'directml').
    """
    dev = device or detect_accelerator()

    if method_id is None:
        # Deduce from config
        pers = getattr(config.clients, "personalization_method", "none")
        topo_type = config.topology.type
        if topo_type in ("hierarchical_ensemble", "hierarchical"):
            def_mode = config.topology.params.get("defense_mode", "none")
            method_id = "topo_defended" if def_mode != "none" else "topo"
        elif pers == "fedprox":
            method_id = "fedprox"
        elif pers == "scaffold":
            method_id = "scaffold"
        elif pers == "ditto":
            method_id = "ditto"
        elif getattr(config.clients, "robust_aggregation_mode", "") == "multi_krum":
            method_id = "multikrum"
        else:
            method_id = "fedavg"

    method_id = method_id.lower()
    num_clients = config.clients.num_clients

    if method_id == "fedavg":
        topology = StarTopology()
        aggregator = FedAvgAggregator()
        engine = CentralizedEngine(config, topology, aggregator, device=dev)
        check_star_invariant(topology, num_clients)

    elif method_id == "fedprox":
        topology = StarTopology()
        aggregator = FedAvgAggregator()
        engine = FedProxEngine(config, topology, aggregator, device=dev)
        check_star_invariant(topology, num_clients)

    elif method_id in ("multikrum", "multi_krum"):
        topology = StarTopology()
        f_byz = getattr(config.clients, "krum_num_byzantine", 0)
        m_sel = getattr(config.clients, "krum_num_selected", 3)
        aggregator = MultiKrumAggregator(f=f_byz, m=m_sel)
        engine = CentralizedEngine(config, topology, aggregator, device=dev)
        check_star_invariant(topology, num_clients)

    elif method_id == "scaffold":
        topology = StarTopology()
        aggregator = FedAvgAggregator()
        engine = ScaffoldEngine(config, topology, aggregator, device=dev)
        check_star_invariant(topology, num_clients)

    elif method_id == "ditto":
        topology = StarTopology()
        aggregator = FedAvgAggregator()
        engine = CentralizedEngine(config, topology, aggregator, device=dev)
        check_star_invariant(topology, num_clients)

    elif method_id in ("topo", "hep"):
        clusters = config.topology.params.get("num_clusters", 3)
        topology = HierarchicalTopology(num_clusters=clusters)
        aggregator = FedAvgAggregator()
        engine = HierarchicalEnsembleEngine(config, topology, aggregator, device=dev)
        check_hierarchical_invariant(topology, num_clients)

    elif method_id in ("topo_defended", "hep_defense", "hep_defended"):
        clusters = config.topology.params.get("num_clusters", 3)
        topology = HierarchicalTopology(num_clusters=clusters)
        aggregator = FedAvgAggregator()
        from src.defense.engine import DefendedEnsembleEngine
        engine = DefendedEnsembleEngine(config, topology, aggregator, device=dev)
        check_hierarchical_invariant(topology, num_clients)

    else:
        raise ValueError(
            f"Unsupported baseline method_id: '{method_id}'. "
            "Supported: 'fedavg', 'fedprox', 'multikrum', 'scaffold', 'ditto', 'topo', 'topo_defended'"
        )

    return topology, aggregator, engine
