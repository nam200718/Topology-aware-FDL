"""
Loss functions and regularizers for Hierarchical Residual Personalization (HEP / HRC).
Includes Active Class Logit Masking (ACLM) and Continuous Entropy-Gated Residual Regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActiveMaskedCrossEntropyLoss(nn.Module):
    """
    Active Class Logit Masking (ACLM) Cross-Entropy Loss.
    Masks out inactive/unobserved classes on the client during local updates,
    preventing irrelevant classes from injecting noisy negative gradients into the shared backbone.
    """
    def __init__(self, mask_value: float = -1e9):
        super(ActiveMaskedCrossEntropyLoss, self).__init__()
        self.mask_value = mask_value

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, active_mask: torch.Tensor = None) -> torch.Tensor:
        if active_mask is not None:
            if active_mask.dim() == 1:
                active_mask = active_mask.unsqueeze(0)
            logits = logits.masked_fill(~active_mask, self.mask_value)
        return F.cross_entropy(logits, targets)


class ClassFrequencyBalancedMaskedLoss(nn.Module):
    """
    Class-Frequency-Balanced Active Class Logit Masking (CF-ACLM) Loss.
    Combines active class masking with inverse-frequency sample weighting,
    preventing local majority classes from dominating the local classification head.
    """
    def __init__(self, mask_value: float = -1e9, gamma: float = 0.5):
        super(ClassFrequencyBalancedMaskedLoss, self).__init__()
        self.mask_value = mask_value
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, active_mask: torch.Tensor = None, class_counts: torch.Tensor = None) -> torch.Tensor:
        if active_mask is not None:
            if active_mask.dim() == 1:
                active_mask = active_mask.unsqueeze(0)
            logits = logits.masked_fill(~active_mask, self.mask_value)

        if class_counts is not None and len(class_counts) > 0:
            # Inverse frequency class weighting
            weights = 1.0 / (class_counts.float() ** self.gamma + 1e-4)
            # Normalize active class weights
            if active_mask is not None:
                mask_1d = active_mask[0] if active_mask.dim() == 2 else active_mask
                weights = weights * mask_1d.float()
            weights = weights / (weights[weights > 0].mean() + 1e-8)
            return F.cross_entropy(logits, targets, weight=weights)
        else:
            return F.cross_entropy(logits, targets)


def compute_binomial_loss_weights(r_skew: float, anchor_min: float = 0.15):
    """
    Continuous differentiable binomial entropy loss weighting:
        lambda_r = anchor_min + (1 - anchor_min) * R_skew^2
        lambda_p = 2 * R_skew * (1 - R_skew)
        lambda_l = (1 - R_skew)^2
    Guarantees:
        - When R_skew = 1.0 (IID): lambda_r = 1.0, lambda_p = 0.0, lambda_l = 0.0 (Pure FedAvg)
        - When R_skew = 0.5 (Moderate): lambda_p is maximum, clean cluster sharing
        - When R_skew = 0.0 (Extreme): lambda_l = 1.0 (Personalized), lambda_r = anchor_min (Anchors backbone)
    """
    lr = anchor_min + (1.0 - anchor_min) * (r_skew ** 2)
    lp = 2.0 * r_skew * (1.0 - r_skew)
    ll = (1.0 - r_skew) ** 2
    total = lr + lp + ll
    alpha_r = lr / total
    alpha_p = lp / total
    alpha_l = ll / total
    return lr, lp, ll, alpha_r, alpha_p, alpha_l



def compute_hierarchical_residual_penalty(model: nn.Module, r_skew: float, mu: float = 1e-3) -> torch.Tensor:
    """
    Computes continuous entropy-gated L2 shrinkage regularization on residual weights.
    Under IID (r_skew -> 1.0): strongly penalizes weight_local -> 0 (collapses naturally to FedAvg).
    Under Extreme Skew (r_skew -> 0.0): penalizes weight_cluster -> 0, allowing private specialization.
    """
    from src.core.model import HierarchicalResidualLinear

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
