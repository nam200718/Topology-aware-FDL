"""
Byzantine Attack Primitives for FedHEP Post-Midterm Evaluation Suite.
Implements:
1. ALIE (A Little Is Enough, Baruch et al., 2019)
2. IPM (Inner Product Manipulation, Xie et al., 2020)
3. Adaptive Cosine Attack (ACA) - targets Temporal Trust Tracking
4. Delayed Onset Attack (DOA) - targets TTT memory and EMA smoothing
"""

import math
from typing import List, Optional
import torch
import torch.nn.functional as F

from oreh.alie_attack import compute_alie_z_max, generate_alie_delta


def generate_ipm_delta(
    honest_deltas: List[torch.Tensor],
    epsilon: float = 0.1,
    scale_norm: bool = True,
) -> torch.Tensor:
    """
    Inner Product Manipulation (IPM) Attack (Xie et al., 2020).
    
    Crafts an adversarial update vector that has a small, controlled positive
    cosine similarity (epsilon > 0) with the honest consensus direction, ensuring
    it passes directional subspace filters (SCCF and hard rejection threshold),
    while orienting the remaining orthogonal energy in an adversarial perturbing direction.
    
    Formula:
        mu = mean(honest_deltas)
        mu_hat = normalize(mu)
        v_dir = -sign(mu)  # Adversarial direction opposing parameter movement
        v_perp = v_dir - (v_dir . mu_hat) * mu_hat  # Gram-Schmidt orthogonalization
        v_perp_hat = normalize(v_perp)
        u = epsilon * mu_hat - sqrt(1 - epsilon^2) * v_perp_hat
        delta_ipm = norm_honest * u
        
    Args:
        honest_deltas: List of benign update tensors [D].
        epsilon: Target inner product (cosine similarity) with honest mean. Default: 0.1.
        scale_norm: If True, scale the attack delta to the median L2 norm of honest updates.
        
    Returns:
        ipm_delta: Adversarially poisoned update vector [D].
    """
    if not honest_deltas:
        raise ValueError("Cannot compute IPM delta from an empty list of honest deltas.")

    device = honest_deltas[0].device
    dtype = honest_deltas[0].dtype

    stacked = torch.stack(honest_deltas, dim=0).to(device=device, dtype=dtype)
    mu = torch.mean(stacked, dim=0)
    mu_norm = torch.norm(mu)

    if mu_norm < 1e-10:
        return torch.randn_like(mu) * 0.01

    mu_hat = mu / mu_norm

    # Adversarial direction opposing parameter movement
    v_dir = -torch.sign(mu)
    if torch.all(v_dir == 0):
        v_dir = torch.randn_like(mu)

    # Gram-Schmidt orthogonalization against mu_hat
    proj = torch.dot(v_dir, mu_hat)
    v_perp = v_dir - proj * mu_hat
    v_perp_norm = torch.norm(v_perp)

    if v_perp_norm < 1e-10:
        # Fallback to random orthogonal vector
        rand_v = torch.randn_like(mu)
        v_perp = rand_v - torch.dot(rand_v, mu_hat) * mu_hat
        v_perp_norm = torch.norm(v_perp)

    v_perp_hat = v_perp / (v_perp_norm + 1e-10)

    # Combine along consensus and perpendicular directions
    eps = max(0.01, min(0.99, float(epsilon)))
    ortho_coeff = math.sqrt(max(0.0, 1.0 - eps ** 2))
    u = eps * mu_hat - ortho_coeff * v_perp_hat

    if scale_norm:
        honest_norms = torch.norm(stacked, dim=1)
        target_norm = torch.median(honest_norms)
        ipm_delta = u * target_norm
    else:
        ipm_delta = u * mu_norm

    return ipm_delta


def generate_adaptive_cosine_delta(
    honest_deltas: List[torch.Tensor],
    prev_trust_score: float = 1.0,
    high_threshold: float = 0.7,
    low_threshold: float = 0.4,
) -> torch.Tensor:
    """
    Adaptive Cosine Attack (ACA) targeting Temporal Trust Tracking (TTT).
    
    Dynamically modulates the attack intensity based on the adversary's cumulative
    trust score in the previous round. When trust is high, the adversary attacks
    aggressively. When trust drops toward the isolation boundary (tau_iso = 0.25),
    the adversary behaves honestly to recover trust before attacking again.
    This creates an oscillatory trust pattern that exploits EMA smoothing (beta=0.85).
    
    Formula:
        if prev_trust > high_threshold:
            attack_weight = 0.85  # Aggressive corruption
        elif prev_trust > low_threshold:
            attack_weight = 0.50  # Moderate disruption
        else:
            attack_weight = 0.05  # Trust rehabilitation (mostly benign)
            
        Delta_ACA = (1 - attack_weight) * mu - attack_weight * sign(mu) * |mu|
        
    Args:
        honest_deltas: List of benign update tensors [D].
        prev_trust_score: Adversary's trust score from previous round (0.0 to 1.0).
        high_threshold: Threshold above which aggressive attack is mounted.
        low_threshold: Threshold below which adversary enters rehabilitation.
        
    Returns:
        adaptive_delta: Adaptively poisoned update vector [D].
    """
    if not honest_deltas:
        raise ValueError("Cannot compute Adaptive Cosine delta from empty honest deltas.")

    device = honest_deltas[0].device
    dtype = honest_deltas[0].dtype

    stacked = torch.stack(honest_deltas, dim=0).to(device=device, dtype=dtype)
    mu = torch.mean(stacked, dim=0)

    t = float(prev_trust_score)
    if t > high_threshold:
        attack_weight = 0.85
    elif t > low_threshold:
        attack_weight = 0.50
    else:
        attack_weight = 0.05

    # Interpolate between honest mean and inverted/corrupted update
    honest_part = (1.0 - attack_weight) * mu
    corrupt_part = -attack_weight * torch.sign(mu) * torch.abs(mu)
    adaptive_delta = honest_part + corrupt_part

    return adaptive_delta

