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
            logits = logits.masked_fill(~active_mask.to(logits.device), self.mask_value)
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
            logits = logits.masked_fill(~active_mask.to(logits.device), self.mask_value)

        if class_counts is not None and len(class_counts) > 0:
            # Inverse frequency class weighting
            weights = 1.0 / (class_counts.float().to(logits.device) ** self.gamma + 1e-4)
            # Normalize active class weights
            if active_mask is not None:
                mask_1d = active_mask[0] if active_mask.dim() == 2 else active_mask
                weights = weights * mask_1d.float().to(logits.device)
            if (weights > 0).any():
                weights = weights / (weights[weights > 0].mean() + 1e-8)
            else:
                weights = torch.ones_like(weights)
            if weights[targets].sum() > 0:
                return F.cross_entropy(logits, targets, weight=weights)
            else:
                return F.cross_entropy(logits, targets)
        else:
            return F.cross_entropy(logits, targets)


def compute_normalized_entropy_r_skew(probs: torch.Tensor, num_classes: int) -> float:
    """
    Normalized exponential Shannon entropy label skew metric:
        R_skew = (exp(H(p)) - 1) / (C - 1)
    where exp(H(p)) is the effective number of classes observed.
    Properties:
        - 1 class observed (Dirac skew): H = 0 -> R_skew = 0.0 (Pure Local Specialization)
        - Uniform k classes observed: H = ln(k) -> R_skew = (k - 1) / (C - 1)
        - Uniform C classes observed (IID): H = ln(C) -> R_skew = 1.0 (Pure Global Consensus)
    """
    import math
    if num_classes <= 1:
        return 1.0
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 1.0
    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
    eff_classes = math.exp(entropy)
    r_skew = (eff_classes - 1.0) / float(num_classes - 1.0)
    return float(min(1.0, max(0.0, r_skew)))


# Backward compatibility alias
compute_hill_number_r_skew = compute_normalized_entropy_r_skew


def compute_dynamic_binomial_loss_weights(
    r_skew: float, num_classes: int = 10, local_classes: int = 2,
    num_clusters: int = 3, enable_parent_head: bool = True,
):
    """
    Dynamic partition-of-unity head weighting with data-support anchor floor:
        a_i = max(1 / (2*K), |Y_i| / C)
        lambda_r = a_i + (1 - a_i) * R_skew^2
        lambda_p = 2 * R_skew * (1 - R_skew) if enable_parent_head else 0.0
        lambda_l = (1 - R_skew)^2
    Guarantees:
        - Exact continuous partition of unity: sum(alpha_k) == 1.0 for all clients
        - Data-support scaled anchor: clients observing narrow label subsets retain sufficient root anchoring
        - Zero manual hyperparameters: a_i is computed dynamically per client from system invariants.
    """
    a_i = max(1.0 / (2.0 * max(1, num_clusters)), float(local_classes) / float(max(1, num_classes)))
    lr = a_i + (1.0 - a_i) * (r_skew ** 2)
    lp = 2.0 * r_skew * (1.0 - r_skew) if enable_parent_head else 0.0
    ll = (1.0 - r_skew) ** 2
    total = lr + lp + ll
    alpha_r = lr / total
    alpha_p = lp / total if enable_parent_head else 0.0
    alpha_l = ll / total
    return lr, lp, ll, alpha_r, alpha_p, alpha_l


def compute_binomial_loss_weights(
    r_skew: float, anchor_min: float = 0.15, enable_parent_head: bool = True,
):
    """
    Continuous differentiable binomial entropy loss weighting:
        lambda_r = anchor_min + (1 - anchor_min) * R_skew^2
        lambda_p = 2 * R_skew * (1 - R_skew) if enable_parent_head else 0.0
        lambda_l = (1 - R_skew)^2
    Guarantees:
        - When R_skew = 1.0 (IID): lambda_r = 1.0, lambda_p = 0.0, lambda_l = 0.0 (Pure FedAvg)
        - When R_skew = 0.5 (Moderate): lambda_p is maximum, clean cluster sharing
        - When R_skew = 0.0 (Extreme): lambda_l = 1.0 (Personalized), lambda_r = anchor_min (Anchors backbone)
    """
    lr = anchor_min + (1.0 - anchor_min) * (r_skew ** 2)
    lp = 2.0 * r_skew * (1.0 - r_skew) if enable_parent_head else 0.0
    ll = (1.0 - r_skew) ** 2
    total = lr + lp + ll
    alpha_r = lr / total
    alpha_p = lp / total if enable_parent_head else 0.0
    alpha_l = ll / total
    return lr, lp, ll, alpha_r, alpha_p, alpha_l




def __getattr__(name: str):
    if name == "compute_hierarchical_residual_penalty":
        import warnings
        from archive.legacy_residual.loss import compute_hierarchical_residual_penalty
        warnings.warn(
            "compute_hierarchical_residual_penalty is deprecated and has been archived. "
            "FedHEP uses anchored binomial loss weighting with ActiveMaskedCrossEntropyLoss.",
            DeprecationWarning,
            stacklevel=2,
        )
        return compute_hierarchical_residual_penalty
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

