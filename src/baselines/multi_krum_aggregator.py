"""Multi-Krum Byzantine-resilient aggregation rule (Blanchard et al., NeurIPS 2017).

Selects m client model updates with the lowest Krum scores (sum of Euclidean
distances to their N - f - 2 nearest neighbors) and computes their mean.
"""
from typing import List, Optional
import torch
from src.core.interfaces import Aggregator, ClientState


def multi_krum_scores(stacked: torch.Tensor, f: int) -> torch.Tensor:
    """Compute Krum score for each model vector.

    Args:
        stacked: Tensor of shape (N, D) representing N flat parameter vectors.
        f: Expected number of Byzantine attackers.

    Returns:
        Tensor of shape (N,) containing Krum scores.
    """
    n = stacked.size(0)
    # Byzantine threshold: 2f + 2 < n => f <= (n - 3) // 2
    f_clamped = min(f, max(0, (n - 3) // 2))
    k = max(1, n - f_clamped - 2)

    # Vectorized pairwise squared Euclidean distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    norms_sq = (stacked ** 2).sum(dim=1)
    dists = norms_sq.unsqueeze(0) + norms_sq.unsqueeze(1) - 2.0 * (stacked @ stacked.T)
    dists = dists.clamp(min=0.0)
    dists.fill_diagonal_(float("inf"))

    # Sum of distances to k nearest neighbors
    topk_dists, _ = torch.topk(dists, k=min(k, n - 1), dim=1, largest=False)
    return topk_dists.sum(dim=1)


class MultiKrumAggregator(Aggregator):
    """Multi-Krum aggregation policy.

    Args:
        f: Number of suspected Byzantine workers. If <= 0, automatically
           estimated based on known state or defaults to 1.
        m: Number of candidates with lowest Krum scores to average (m >= 1).
    """

    def __init__(self, f: int = 0, m: int = 3):
        self.f = f
        self.m = m

    def aggregate(self, states: List[ClientState]) -> torch.Tensor:
        if not states:
            raise ValueError("Cannot aggregate empty list of states.")

        if len(states) <= 2:
            return torch.stack([s.weights for s in states]).mean(dim=0)

        stacked = torch.stack([s.weights for s in states], dim=0)
        n = stacked.size(0)

        # Estimate f if not explicitly provided
        if self.f > 0:
            f = self.f
        else:
            byz_count = sum(1 for s in states if getattr(s, "is_byzantine", False))
            f = max(1, byz_count) if byz_count > 0 else max(1, (n - 3) // 2)

        scores = multi_krum_scores(stacked, f)

        # Multi-Krum chooses m updates with lowest scores
        # Ensure m is within valid range [1, n]
        m = max(1, min(self.m, n))
        _, selected_indices = torch.topk(scores, k=m, largest=False)

        selected_weights = stacked[selected_indices]
        return selected_weights.mean(dim=0)
