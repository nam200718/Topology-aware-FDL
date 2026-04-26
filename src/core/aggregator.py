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
            return torch.mean(stacked_weights, dim=0)
            
        # Weighted average based on number of local data samples
        weighted_sum = torch.zeros_like(states[0].weights)
        for state in states:
            weight = state.data_samples / total_samples
            weighted_sum += state.weights * weight
            
        return weighted_sum
