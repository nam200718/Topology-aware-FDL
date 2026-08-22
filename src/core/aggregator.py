from typing import List
import numpy as np
from src.core.interfaces import Aggregator, ClientState

import torch
import torch.nn as nn
import torch.nn.functional as F

class FedAvgAggregator(Aggregator):
    def aggregate(self, states: List[ClientState]) -> torch.Tensor:
        if not states:
            raise ValueError("Cannot aggregate empty list of states.")
        
        total_samples = sum(state.data_samples for state in states)
        stacked_weights = torch.stack([state.weights for state in states])
        if total_samples == 0:
            # Fallback to simple average
            return stacked_weights.mean(dim=0)
            
        # Weighted average based on number of local data samples via BLAS matrix-vector product (w^T A)
        device = states[0].weights.device
        sample_weights = torch.tensor(
            [state.data_samples / total_samples for state in states],
            device=device,
            dtype=states[0].weights.dtype
        )
        return sample_weights @ stacked_weights

class RandomizedAggregator(Aggregator):
    """
    Assigns a random weight to each client's update during aggregation.
    Can be used to test if random noise in aggregation provides any 
    inherent robustness against biased attacks.
    """
    def aggregate(self, states: List[ClientState]) -> torch.Tensor:
        if not states:
            raise ValueError("Cannot aggregate empty list of states.")
        
        device = states[0].weights.device
        weights = torch.rand(len(states), device=device, dtype=states[0].weights.dtype)
        normalized_weights = weights / weights.sum()
        stacked_weights = torch.stack([state.weights for state in states])
        return normalized_weights @ stacked_weights


class TrimmedMeanAggregator(Aggregator):
    """
    Coordinate-Wise Trimmed Mean Aggregator.
    Discards the top and bottom beta fraction of values per coordinate,
    providing provable Byzantine resilience against sign-flipping and gradient noise.
    """
    def __init__(self, beta: float = 0.20):
        self.beta = beta

    def aggregate(self, states: List[ClientState]) -> torch.Tensor:
        if not states:
            raise ValueError("Cannot aggregate empty list of states.")
        if len(states) <= 2:
            return torch.stack([s.weights for s in states]).mean(dim=0)

        stacked = torch.stack([s.weights for s in states], dim=0)
        return trimmed_mean(stacked, self.beta)


# ----------------------------------------------------------------------
# Delta-space robust aggregation primitives (tensor level)
# ----------------------------------------------------------------------

def trimmed_mean(stack: torch.Tensor, beta: float) -> torch.Tensor:
    """Coordinate-wise trimmed mean over dim 0 of a [N, D] stack."""
    n = stack.size(0)
    k = int(n * beta)
    if k == 0 or 2 * k >= n:
        return stack.mean(dim=0)
    sorted_stack, _ = torch.sort(stack, dim=0)
    return sorted_stack[k:n - k].mean(dim=0)


def coordinate_median(stack: torch.Tensor) -> torch.Tensor:
    """Coordinate-wise median over dim 0 of a [N, D] stack.

    Sort-based: torch.median(dim=0) is silently broken on some accelerators
    (DirectML CPU-fallback yields an empty values tensor). For even N this
    returns the true statistical median (mean of the two middle values).
    """
    sorted_stack, _ = torch.sort(stack, dim=0)
    n = sorted_stack.size(0)
    if n % 2 == 1:
        return sorted_stack[n // 2]
    return 0.5 * (sorted_stack[n // 2 - 1] + sorted_stack[n // 2])


def soft_cosine_trust(deltas: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Trust weights from cosine similarity to the unit-normalized centroid.

    Sign-flipped updates point against the honest centroid, so they receive
    near-zero trust without any attack-specific heuristic. Trust is
    scale-invariant by construction; magnitude-inflation attacks are handled
    separately by bounding the deltas before the weighted combination
    (see DeltaSpaceRobustAggregator).

    deltas: [N, D] update vectors. Returns [N] trust weights summing to 1.
    """
    n = deltas.size(0)
    if n == 1:
        return torch.ones(1, device=deltas.device)

    unit = F.normalize(deltas, dim=1, eps=1e-10)
    centroid = unit.mean(dim=0)
    norm_centroid = centroid.norm()
    # When directions nearly cancel there is no dominant honest direction;
    # cosine statistics against a near-zero centroid only amplify float noise,
    # so fall back to uniform trust.
    if norm_centroid < 0.2:
        return torch.full((n,), 1.0 / n, device=deltas.device)

    sims = F.cosine_similarity(unit, centroid.unsqueeze(0).expand_as(unit), dim=1, eps=1e-10)
    return F.softmax(sims / temperature, dim=0)


def _lower_quartile(v: torch.Tensor) -> torch.Tensor:
    """Lower-quartile element of a 1-D tensor (sort-based).

    Used as the norm-bounding anchor because a median anchor can be inflated
    by the very attacks it is meant to bound when they are numerous.
    """
    s = torch.sort(v).values
    idx = min(s.size(0) - 1, int(0.25 * s.size(0)))
    return s[idx]


def bound_update_norms(stacked: torch.Tensor, k: float) -> torch.Tensor:
    """Clamp each update's norm to k * lower-quartile norm (row-wise rescale).

    Neutralizes magnitude-inflation attacks before trust-weighted summation;
    direction-preserving, so downstream cosine statistics are unchanged.
    The quartile anchor (rather than median/mean) keeps the allowance immune
    to attackers inflating the statistic with their own oversized updates.
    """
    if k is None or k <= 0:
        return stacked
    norms = stacked.norm(dim=1)
    scale = (k * _lower_quartile(norms) / norms).clamp(max=1.0)
    return stacked * scale.unsqueeze(1)


class DeltaSpaceRobustAggregator:
    """
    Server-side robust aggregation over client update deltas.

    Aggregates deltas (w_i - w_ref) instead of raw weights so that directional
    statistics (cosine trust) and coordinate-wise trimming are meaningful, then
    maps back through the reference. BatchNorm running buffers can optionally
    be aggregated via coordinate-wise median, since poisoned statistics are an
    attack vector that parametric filtering alone does not cover.

    Zero client-side cost: everything runs on the server on flat vectors.
    """
    def __init__(self, mode: str = "soft_cosine", beta: float = 0.20,
                 temperature: float = 0.5, norm_bound_k: float = 3.0,
                 buffer_slices=None):
        if mode not in ("trimmed_mean", "soft_cosine"):
            raise ValueError(f"Unknown robust aggregation mode '{mode}'")
        self.mode = mode
        self.beta = beta
        self.temperature = temperature
        self.norm_bound_k = norm_bound_k
        # Optional list of (start, end) index ranges marking buffer coordinates
        # in the flattened parameter vector (e.g. BN running stats).
        self.buffer_slices = buffer_slices or []
        self._idx_cache = {}

    def _split_indices(self, dim: int, device):
        """Cached (param_idx, buffer_idx) integer index tensors. Integer
        indexing is used because boolean advanced indexing is unsupported on
        some accelerators (e.g. DirectML)."""
        key = (dim, str(device))
        if key not in self._idx_cache:
            buf = []
            for start, end in self.buffer_slices:
                buf.extend(range(start, end))
            buf_set = set(buf)
            keep = [i for i in range(dim) if i not in buf_set]
            self._idx_cache[key] = (
                torch.tensor(keep, dtype=torch.long, device=device),
                torch.tensor(buf, dtype=torch.long, device=device),
            )
        return self._idx_cache[key]

    def aggregate_deltas(self, deltas: List[torch.Tensor], reference: torch.Tensor = None) -> torch.Tensor:
        if not deltas:
            raise ValueError("Cannot aggregate empty list of deltas.")
        stacked = torch.stack(deltas, dim=0)
        # Some accelerators nondeterministically promote dtypes across
        # CPU-fallback ops; pin everything to the incoming dtype.
        stacked = stacked.float()
        out_dtype = stacked.dtype

        if self.buffer_slices:
            dim = stacked.size(1)
            param_idx, buf_idx = self._split_indices(dim, stacked.device)
            agg = torch.empty_like(stacked[0])
            agg.index_copy_(0, param_idx,
                            self._aggregate_param_deltas(stacked.index_select(1, param_idx)).float())
            agg.index_copy_(0, buf_idx,
                            coordinate_median(stacked.index_select(1, buf_idx)).float())
            return agg.to(out_dtype)
        return self._aggregate_param_deltas(stacked).float()

    def _aggregate_param_deltas(self, stacked: torch.Tensor) -> torch.Tensor:
        if self.mode == "trimmed_mean":
            return trimmed_mean(stacked, self.beta)
        bounded = bound_update_norms(stacked, self.norm_bound_k)
        trust = soft_cosine_trust(bounded, self.temperature)
        return (trust.unsqueeze(1) * bounded).sum(dim=0)


def compute_buffer_slices(model: nn.Module):
    """(start, end) flat-vector ranges of state_dict entries that are buffers
    (e.g. BatchNorm running_mean/var), following model_to_vector's ordering.

    Uses state-dict key sets rather than tensor storage pointers so it works
    on accelerators that do not expose data_ptr (e.g. DirectML).
    """
    param_keys = {k for k, _ in model.named_parameters()}
    slices = []
    offset = 0
    for key, v in model.state_dict().items():
        numel = v.numel()
        if key not in param_keys:
            slices.append((offset, offset + numel))
        offset += numel
    return slices

