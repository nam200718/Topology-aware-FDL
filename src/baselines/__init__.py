"""Baseline implementations for AAMAS 2027 comparison.

Isolated from core codebase to prevent any merge conflicts with ongoing work.
Supports: FedAvg, FedProx, Multi-Krum, SCAFFOLD, Ditto, and proposed Topo (HEP).
"""
from src.baselines.config import BaselineClientConfig, BaselineSimulationConfig
from src.baselines.fedprox_updater import FedProxUpdater
from src.baselines.multi_krum_aggregator import MultiKrumAggregator, multi_krum_scores
from src.baselines.scaffold_updater import ScaffoldUpdater
from src.baselines.scaffold_engine import ScaffoldEngine
from src.baselines.factory import build_baseline_engine
from src.baselines.multi_seed_runner import MultiSeedRunner
from src.baselines.experiment_configs import (
    METHODS,
    REGIMES,
    ATTACK_TYPES,
    BYZANTINE_RATES,
    SEEDS,
    CIFAR100_DEFAULTS,
    create_personalization_config,
    create_byzantine_config,
)

__all__ = [
    "BaselineClientConfig",
    "BaselineSimulationConfig",
    "FedProxUpdater",
    "MultiKrumAggregator",
    "multi_krum_scores",
    "ScaffoldUpdater",
    "ScaffoldEngine",
    "build_baseline_engine",
    "MultiSeedRunner",
    "METHODS",
    "REGIMES",
    "ATTACK_TYPES",
    "BYZANTINE_RATES",
    "SEEDS",
    "CIFAR100_DEFAULTS",
    "create_personalization_config",
    "create_byzantine_config",
]
