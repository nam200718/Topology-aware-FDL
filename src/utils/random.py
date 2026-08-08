import numpy as np
import random
import os
import torch

def set_seed(seed: int):
    """Sets the seed for reproducibility across all relevant RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_random_state() -> np.random.RandomState:
    """Returns a new independent RandomState to avoid global rng pollution for localized events."""
    # We can seed it dynamically yet deterministically
    rand_int = np.random.randint(0, 2**31 - 1)
    return np.random.RandomState(rand_int)
