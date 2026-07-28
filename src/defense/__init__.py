"""
Byzantine Defense Module for Hierarchical Ensemble FL.
Provides Soft Rejection aggregation with trust-based scoring.
"""

from src.defense.config import DefenseConfig
from src.defense.aggregator import SoftRejectionAggregator
from src.defense.engine import DefendedEnsembleEngine
from src.defense.trust_tracker import TrustTracker
from src.defense.visualizer import DefenseVisualizer

__all__ = [
    "DefenseConfig",
    "SoftRejectionAggregator",
    "DefendedEnsembleEngine",
    "TrustTracker",
    "DefenseVisualizer",
]
