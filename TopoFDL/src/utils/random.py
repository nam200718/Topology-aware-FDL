import numpy as np
import random
import os

def set_seed(seed: int):
    """Sets the seed for reproducibility across all relevant RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    # If standard torch/tf were used, we'd set them here too
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_random_state() -> np.random.RandomState:
    """Returns a new independent RandomState to avoid global rng pollution for localized events."""
    # We can seed it dynamically yet deterministically
    rand_int = np.random.randint(0, 2**31 - 1)
    return np.random.RandomState(rand_int)
