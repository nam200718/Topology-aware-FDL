"""
Oreh package: Multi-tier Byzantine evaluation suite for FedHEP.
Contains ALIE attack implementation, targeted tier updates, and multi-scope profiling.
"""

from oreh.alie_attack import compute_alie_z_max, generate_alie_delta
from oreh.updater import OrehLocalUpdater
from oreh.engine import OrehDefendedEngine, OrehCentralizedEngine
from oreh.comparison_runner import OrehComparisonRunner

__all__ = [
    "compute_alie_z_max",
    "generate_alie_delta",
    "OrehLocalUpdater",
    "OrehDefendedEngine",
    "OrehCentralizedEngine",
    "OrehComparisonRunner",
]
