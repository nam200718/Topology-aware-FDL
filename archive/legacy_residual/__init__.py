"""
Legacy Additive Hierarchical Residual Parameterization (HRC / H-ResFL).
Archived for historical reference and backward compatibility.
The canonical FedHEP architecture uses MultiHeadResNet9 and MultiHeadMobileNetV3Small.
"""

from .model import (
    HierarchicalResidualLinear,
    HierarchicalLoRALinear,
    HierarchicalResidualResNet9,
    HierarchicalResidualMobileNetV3Small,
)
from .loss import compute_hierarchical_residual_penalty

__all__ = [
    "HierarchicalResidualLinear",
    "HierarchicalLoRALinear",
    "HierarchicalResidualResNet9",
    "HierarchicalResidualMobileNetV3Small",
    "compute_hierarchical_residual_penalty",
]
