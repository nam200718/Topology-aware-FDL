"""
A Little Is Enough (ALIE) Attack Module.
Based on Baruch, Gilad, et al. "A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS 2019 / arXiv:1902.06156).

In ALIE, m Byzantine attackers manipulate model coordinates by shifting the mean by a small multiple
of the standard deviation of honest clients:
    Delta_w_ALIE = mu - z_max * sigma  (or mu + z_max * sigma * sign)

where z_max is computed such that Phi(z_max) < (n - s) / n, ensuring the poisoned update
remains within the statistical variance of benign workers and bypasses coordinate-wise median,
trimmed mean, Krum, and threshold-based defenses.
"""

import math
from typing import List, Optional
import torch


def compute_alie_z_max(num_total: int, num_attackers: int, s: Optional[int] = None) -> float:
    """
    Computes theoretical z_max parameter according to Baruch et al. (2019).
    
    Formula:
        s = floor(n / 2) - m + 1  (support margin)
        p = (n - s) / n
        z_max = Phi^-1(p)
    
    Args:
        num_total: Total number of clients (n).
        num_attackers: Number of malicious Byzantine clients (m).
        s: Number of negative deviations allowed before majority/threshold rejection.
           Default is floor(n/2) - m + 1 (for median/majority defenses).
           
    Returns:
        z_max: Float scalar representing the standard deviation multiplier.
    """
    if num_attackers <= 0 or num_total <= num_attackers:
        return 0.0

    n = int(num_total)
    m = int(num_attackers)
    
    if s is None:
        # Theoretical threshold for majority-based defenses (Baruch et al. Section 3.3)
        s = max(1, math.floor(n / 2) - m + 1)
        
    # Probability bound p = (n - s) / n
    p = (n - s) / n
    
    # Clip p to numerically stable range for standard normal inverse CDF
    p = max(0.5001, min(p, 0.9999))
    
    # Inverse CDF of standard normal distribution: Phi^-1(p)
    prob_tensor = torch.tensor(p, dtype=torch.float32)
    z_max = float(torch.special.ndtri(prob_tensor).item())
    
    return max(0.1, z_max)


def generate_alie_delta(
    honest_deltas: List[torch.Tensor],
    z_max: Optional[float] = None,
    num_total: Optional[int] = None,
    num_attackers: Optional[int] = None,
    directed: bool = False,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Generates the ALIE perturbation delta vector given a set of honest client deltas.
    
    Args:
        honest_deltas: List of delta tensors [D] from honest clients in current round.
        z_max: Specific z_max value to use. If None, computed theoretically from (num_total, num_attackers).
        num_total: Total number of clients (needed if z_max is None).
        num_attackers: Total number of attackers (needed if z_max is None).
        directed: If True, scale by sign(mu) to bias explicitly against descent direction.
        epsilon: Numerical stability epsilon for standard deviation.
        
    Returns:
        alie_delta: Tensor [D] representing the poisoned delta.
    """
    if not honest_deltas:
        raise ValueError("Cannot compute ALIE delta from an empty list of honest deltas.")

    device = honest_deltas[0].device
    dtype = honest_deltas[0].dtype

    # Stack honest deltas: shape [H, D]
    stacked = torch.stack(honest_deltas, dim=0).to(device=device, dtype=dtype)
    num_honest = stacked.size(0)

    # Coordinate-wise mean mu
    mu = torch.mean(stacked, dim=0)

    # Coordinate-wise standard deviation sigma
    if num_honest > 1:
        sigma = torch.std(stacked, dim=0, unbiased=True) + epsilon
    else:
        sigma = torch.abs(mu) * 0.1 + epsilon

    # Determine z_max
    if z_max is None:
        if num_total is None or num_attackers is None:
            effective_z = 1.0
        else:
            effective_z = compute_alie_z_max(num_total, num_attackers)
    else:
        effective_z = float(z_max)

    # Calculate perturbed delta: Delta_ALIE = mu - z_max * sigma
    if directed:
        alie_delta = mu - effective_z * sigma * torch.sign(mu)
    else:
        alie_delta = mu - effective_z * sigma

    return alie_delta
