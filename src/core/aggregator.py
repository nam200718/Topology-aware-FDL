from typing import List
import numpy as np
from src.core.interfaces import Aggregator, ClientState

import torch

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
