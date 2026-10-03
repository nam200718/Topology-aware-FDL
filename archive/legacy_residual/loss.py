"""
Legacy Additive Hierarchical Residual Penalty Loss.
Archived for historical reference and backward compatibility.
"""

import torch
import torch.nn as nn
from .model import HierarchicalResidualLinear


def compute_hierarchical_residual_penalty(model: nn.Module, r_skew: float, mu: float = 1e-3) -> torch.Tensor:
    """
    Computes continuous entropy-gated L2 shrinkage regularization on residual weights.
    Under IID (r_skew -> 1.0): strongly penalizes weight_local -> 0 (collapses naturally to FedAvg).
    Under Extreme Skew (r_skew -> 0.0): penalizes weight_cluster -> 0, allowing private specialization.
    """
    device = next(model.parameters()).device
    reg_loss = torch.tensor(0.0, device=device)
    for module in model.modules():
        if isinstance(module, HierarchicalResidualLinear):
            # Local residual shrinkage: mu * r_skew * ||Delta W_local||^2
            reg_loss = reg_loss + 0.5 * mu * float(r_skew) * torch.sum(module.weight_local ** 2)
            if module.use_bias and module.bias_local is not None:
                reg_loss = reg_loss + 0.5 * mu * float(r_skew) * torch.sum(module.bias_local ** 2)

            # Cluster residual shrinkage: mu * (1 - r_skew) * ||Delta W_cluster||^2
            reg_loss = reg_loss + 0.5 * mu * float(1.0 - r_skew) * torch.sum(module.weight_cluster ** 2)
            if module.use_bias and module.bias_cluster is not None:
                reg_loss = reg_loss + 0.5 * mu * float(1.0 - r_skew) * torch.sum(module.bias_cluster ** 2)
    return reg_loss
