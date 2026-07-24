"""
Experiment management package for Topology-aware-FDL.
"""
from src.experiments.builder import TopologyEngineFactory, detect_device, check_invariants
from src.experiments.runner import ExperimentRunner, SingleExperimentRunner
from src.experiments.visualizer import (
    plot_experiment_results,
    plot_comparison_convergence,
    plot_robustness_summary,
    plot_byzantine_matrix,
)

__all__ = [
    "TopologyEngineFactory",
    "detect_device",
    "check_invariants",
    "ExperimentRunner",
    "SingleExperimentRunner",
    "plot_experiment_results",
    "plot_comparison_convergence",
    "plot_robustness_summary",
    "plot_byzantine_matrix",
]
