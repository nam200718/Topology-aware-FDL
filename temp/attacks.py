"""
Utility functions cho defense layer.
- check_inf_nan: phát hiện gradient explosion
- apply_magnitude_inflation: attack post-hoc (dùng trong experiment scripts)
"""
import torch

def check_inf_nan(weights, initial_weights) -> bool:
    """True = CLEAN, False = MALICIOUS (Inf/NaN detected)."""
    delta = weights - initial_weights
    return not (torch.isnan(delta).any() or torch.isinf(delta).any())

def apply_magnitude_inflation(weights, initial_weights, scale=50.0):
    """Scale delta lên 50x. Dùng trong experiment scripts."""
    delta = weights - initial_weights
    return initial_weights + delta * scale
