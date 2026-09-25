"""Extended config for baseline experiments.

Inherits from src.config without altering core configuration schemas.
"""
from typing import Optional, Literal
from pydantic import Field
from src.config import ClientConfig, SimulationConfig


class BaselineClientConfig(ClientConfig):
    """ClientConfig extended with parameters for baselines."""
    # FedProx proximal regularization strength
    fedprox_mu: float = 0.01

    # Multi-Krum server-side parameters
    krum_num_byzantine: int = 0      # Expected Byzantine count (0 = auto-detect based on rate)
    krum_num_selected: int = 3       # Number of updates selected and averaged (m)

    # General baseline method identifier
    # Allows "none", "ditto", "fedprox", "scaffold", "multikrum", "hep", etc.
    personalization_method: str = "none"

    # Default to single-head model for standard baselines (overridden to shared_backbone for Topo)
    compute_optimization_mode: str = "none"


class BaselineSimulationConfig(SimulationConfig):
    """SimulationConfig specialized for baseline experiments."""
    clients: BaselineClientConfig = Field(default_factory=BaselineClientConfig)
