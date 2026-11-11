"""
Oreh package: Multi-tier Byzantine evaluation suite for FedHEP.
Contains ALIE/IPM/Adaptive/Delayed attack implementations, targeted tier updates, and multi-scope profiling.
"""

from oreh.attacks import (
    compute_alie_z_max,
    generate_alie_delta,
    generate_ipm_delta,
    generate_adaptive_cosine_delta,
)
from oreh.updater import OrehLocalUpdater
from oreh.engine import OrehDefendedEngine, OrehCentralizedEngine
from oreh.comparison_runner import OrehComparisonRunner

__all__ = [
    "compute_alie_z_max",
    "generate_alie_delta",
    "generate_ipm_delta",
    "generate_adaptive_cosine_delta",
    "OrehLocalUpdater",
    "OrehDefendedEngine",
    "OrehCentralizedEngine",
    "OrehComparisonRunner",
]
