import torch
import torch.nn.functional as F
from typing import List, Dict, Optional

from src.core.interfaces import Aggregator, ClientState
from src.core.aggregator import FedAvgAggregator
from src.defense.config import DefenseConfig


class SoftRejectionAggregator(Aggregator):
    """
    Aggregator with Soft Rejection mechanism.
    
    Calculate trust score for each client based on cosine similarity with centroid,
    then use trust score as aggregation weight.
    
    Workflow:
    1. Calculate update delta: Δw_i = w_i_after - w_reference
    2. Calculate centroid = mean(all Δw_i)
    3. Cosine similarity between each Δw_i and centroid
    4. Temperature scaling → trust score ∈ (0, 1)
    5. Weighted average according to trust score
    """

    def __init__(self, defense_config: DefenseConfig):
        self.config = defense_config
        self.current_temperature = defense_config.temperature
        self._last_trust_scores: Dict[int, float] = {}
        self._fallback_aggregator = FedAvgAggregator()

    def aggregate(
        self, states: List[ClientState], reference_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        """
        Aggregate client states with trust-weighted scoring.

        Args:
            states: List of ClientState objects after local training.
            reference_weights: Model weights before client train.
                              If None, use centroid as reference.

        Returns:
            Aggregated weight tensor.
        """
        if not states:
            raise ValueError("Cannot aggregate empty list of states.")

        # If defense is off -> fallback FedAvg
        if self.config.defense_mode == "none":
            return self._fallback_aggregator.aggregate(states)

        # Cosine similarity-based soft rejection
        if self.config.defense_mode == "soft_cosine":
            return self._aggregate_cosine(states, reference_weights)

        if self.config.defense_mode == "soft_norm":
            return self._aggregate_norm(states, reference_weights)

        # Unknown mode → fallback
        return self._fallback_aggregator.aggregate(states)

    def _aggregate_cosine(
        self, states: List[ClientState], reference_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Cosine similarity-based soft rejection."""
        n = len(states)

        # If only one client -> no defense
        if n == 1:
            self._last_trust_scores = {states[0].client_id: 1.0}
            return states[0].weights.clone()

        # Calculate update deltas
        if reference_weights is not None:
            deltas = [s.weights - reference_weights for s in states]
        else:
            # No reference -> use centroid of raw weights
            stacked = torch.stack([s.weights for s in states])
            centroid_raw = stacked.mean(dim=0)
            deltas = [s.weights - centroid_raw for s in states]

        # Calculate directional centroid using unit-normalized deltas
        # This protects against Magnitude Inflation (Scaling) Attacks
        normalized_deltas = [d / (d.norm() + 1e-10) for d in deltas]
        stacked_deltas = torch.stack(normalized_deltas)
        centroid = stacked_deltas.mean(dim=0)

        # Cosine similarity between each delta and centroid
        centroid_norm = centroid.norm()
        if centroid_norm < 1e-10:
            # All updates are the same -> trust = 1.0
            self._last_trust_scores = {s.client_id: 1.0 for s in states}
            return self._fallback_aggregator.aggregate(states)

        similarities = []
        for delta in deltas:
            delta_norm = delta.norm()
            if delta_norm < 1e-10:
                similarities.append(0.0)
            else:
                cos_sim = F.cosine_similarity(
                    delta.unsqueeze(0), centroid.unsqueeze(0)
                ).item()
                similarities.append(cos_sim)

        # Temperature scaling → trust scores
        # trust_i = softmax(similarity_i / temperature)
        sim_tensor = torch.tensor(similarities)
        trust_scores = F.softmax(sim_tensor / self.current_temperature, dim=0)

        # Save trust scores (for TrustTracker)
        self._last_trust_scores = {
            states[i].client_id: trust_scores[i].item() for i in range(n)
        }

        # Weighted average according to trust scores
        aggregated = states[0].weights.new_zeros(states[0].weights.shape)
        for i, state in enumerate(states):
            aggregated += state.weights * trust_scores[i]

        return aggregated

    def _aggregate_norm(
        self, states: List[ClientState], reference_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Norm-based soft rejection (Phase 2)."""
        n = len(states)

        if n == 1:
            self._last_trust_scores = {states[0].client_id: 1.0}
            return states[0].weights.clone()

        # Calculate update deltas
        if reference_weights is not None:
            deltas = [s.weights - reference_weights for s in states]
        else:
            stacked = torch.stack([s.weights for s in states])
            centroid_raw = stacked.mean(dim=0)
            deltas = [s.weights - centroid_raw for s in states]

        # Calculate norm of each delta
        norms = torch.tensor([d.norm().item() for d in deltas])
        median_norm = norms.median()

        # Trust score based on norm ratio
        # Client with norm too high (compared to median) -> low trust
        norm_ratios = norms / (median_norm + 1e-10)
        # Clamp: if norm_ratio > threshold -> penalize
        penalties = torch.clamp(norm_ratios / self.config.norm_threshold, max=1.0)
        raw_trust = 1.0 - (penalties - 1.0).clamp(min=0.0)
        trust_scores = F.softmax(raw_trust / self.current_temperature, dim=0)

        self._last_trust_scores = {
            states[i].client_id: trust_scores[i].item() for i in range(n)
        }

        aggregated = states[0].weights.new_zeros(states[0].weights.shape)
        for i, state in enumerate(states):
            aggregated += state.weights * trust_scores[i]

        return aggregated

    def get_last_trust_scores(self) -> Dict[int, float]:
        """Return trust scores from the last aggregate."""
        return dict(self._last_trust_scores)

    def decay_temperature(self):
        """Call after each round to gradually decrease temperature -> stronger defense."""
        self.current_temperature = max(
            self.config.temperature_min,
            self.current_temperature * self.config.temperature_decay,
        )
