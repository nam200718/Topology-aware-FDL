"""
Clustering utilities for Oreh FedHEP experiments.

Provides weights-only vector serialization (stripping BatchNorm buffers such as
running_mean, running_var, and num_batches_tracked) without modifying src/.
"""

from typing import Set
import torch
import torch.nn as nn


def _get_trainable_param_keys(model: nn.Module) -> Set[str]:
    """Return the set of state_dict keys corresponding to trainable parameters."""
    return {name for name, _ in model.named_parameters()}


def model_to_vector(model: nn.Module, weights_only: bool = False) -> torch.Tensor:
    """Serialize model tensors to a flat 1D float vector in state_dict order.

    Args:
        model: PyTorch module to serialize.
        weights_only: If True, exclude non-parameter buffers (e.g. BatchNorm
            running_mean, running_var, num_batches_tracked) and include only
            trainable parameters. If False, matches src.core.model.model_to_vector.
    """
    with torch.no_grad():
        if not weights_only:
            tensors = [v.data.reshape(-1).float() for v in model.state_dict().values()]
        else:
            param_keys = _get_trainable_param_keys(model)
            tensors = [
                v.data.reshape(-1).float()
                for k, v in model.state_dict().items()
                if k in param_keys
            ]
        if not tensors:
            return torch.tensor([], dtype=torch.float32)
        return torch.cat(tensors)


def compute_weight_indices(model: nn.Module) -> torch.Tensor:
    """Compute flat-vector indices of trainable parameters in full state_dict layout.

    Given a full-state delta vector ``d`` produced in ``model_to_vector(..., weights_only=False)``
    ordering, ``d[compute_weight_indices(model)]`` yields the exact weights-only delta vector
    with all BatchNorm counters and running statistics stripped.
    """
    param_keys = _get_trainable_param_keys(model)
    ranges = []
    offset = 0
    for key, val in model.state_dict().items():
        numel = val.numel()
        if key in param_keys and numel > 0:
            ranges.append(torch.arange(offset, offset + numel, dtype=torch.long))
        offset += numel
    if not ranges:
        return torch.tensor([], dtype=torch.long)
    return torch.cat(ranges)

