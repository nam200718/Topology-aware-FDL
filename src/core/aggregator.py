from typing import List
import numpy as np
from src.core.interfaces import Aggregator, ClientState

import torch

class FedAvgAggregator(Aggregator):
    def aggregate(self, states: List[ClientState]) -> torch.Tensor:
        if not states:
            raise ValueError("Cannot aggregate empty list of states.")
        
        total_samples = sum(state.data_samples for state in states)
        if total_samples == 0:
            # Fallback to simple average
            stacked_weights = torch.stack([state.weights for state in states])
            return stacked_weights.mean(dim=0)
            
        # Weighted average based on number of local data samples
        weighted_sum = states[0].weights.new_zeros(states[0].weights.shape)
        for state in states:
            weight = state.data_samples / total_samples
            weighted_sum += state.weights * weight
            
        return weighted_sum

class RandomizedAggregator(Aggregator):
    """
    Assigns a random weight to each client's update during aggregation.
    Can be used to test if random noise in aggregation provides any 
    inherent robustness against biased attacks.
    """
    def aggregate(self, states: List[ClientState]) -> torch.Tensor:
        if not states:
            raise ValueError("Cannot aggregate empty list of states.")
        
        # Generate random weights for each client update
        # We use torch.rand to stay on the same device/seed logic
        weights = torch.rand(len(states))
        total = weights.sum()
        normalized_weights = weights / total
        
        weighted_sum = states[0].weights.new_zeros(states[0].weights.shape)
        for i, state in enumerate(states):
            weighted_sum += state.weights * normalized_weights[i]
            
        return weighted_sum
