from dataclasses import dataclass, field
from typing import Literal

@dataclass
class DefenseConfig:
    """Configuration for Byzantine defense mechanisms."""
    # Defense strategy: "none" = FedAvg baseline, "soft_cosine" = cosine similarity
    # "soft_norm" will be added in the future
    defense_mode: Literal["none", "soft_cosine", "soft_norm"] = "soft_cosine"

    # Temperature for softmax trust scoring
    # Higher temperature -> convergent trust scores, lower temperature -> divergent trust scores
    temperature: float = 1.0

    # Temperature decay rate (label after each round)
    # Ensure that defense increase by the time

    temperature_decay: float = 0.95 
    temperature_min: float = 0.1 

    # Defense application 
    # "cluster" = Cluster head (default)
    # "global" = only Global server
    # "both" = both Cluster head and Global server (future development)
    defense_scope: Literal["cluster", "global", "both"] = "cluster"

    # Threshold for norm-based rejection.
    # Updates if norm > threshold * median_norm will decrease trust
    norm_threshold: float = 2.0

    # NEW: Norm Bounding — clip delta norm trước aggregation
    norm_bounding_enabled: bool = False
    norm_bounding_multiplier: float = 2.0   # clip tại median_norm × multiplier

    # NEW: Hard Rejection — loại bỏ client có cosine score < threshold
    hard_rejection_enabled: bool = False
    hard_rejection_threshold: float = 0.0   # cosine sim < threshold → reject
