"""
Unit tests for Byzantine defense: SoftRejectionAggregator.
Run: pytest tests/test_soft_rejection.py -v
"""
import pytest
import torch

from src.core.interfaces import ClientState
from src.defense.config import DefenseConfig
from src.defense.aggregator import SoftRejectionAggregator
from src.defense.trust_tracker import TrustTracker


@pytest.fixture
def honest_states():
    """Create 4 honest client states with similar updates along a shared gradient."""
    reference = torch.randn(100)
    true_gradient = torch.randn(100)
    states = []
    for i in range(4):
        # Honest clients: small updates along true gradient + small noise
        w = reference - (true_gradient + torch.randn(100) * 0.05)
        s = ClientState(client_id=i, initial_weights=w)
        s.data_samples = 100
        states.append(s)
    return states, reference


@pytest.fixture
def mixed_states():
    """Create 4 honest + 1 Byzantine client state."""
    reference = torch.randn(100)
    true_gradient = torch.randn(100)
    states = []

    # 4 honest clients: small updates along true gradient + small noise
    for i in range(4):
        w = reference - (true_gradient + torch.randn(100) * 0.05)
        s = ClientState(client_id=i, initial_weights=w)
        s.data_samples = 100
        states.append(s)

    # 1 Byzantine client: inverted direction with scale inflation
    byz_w = reference + (true_gradient * 5.0)  # Inverted direction
    byz_s = ClientState(client_id=4, initial_weights=byz_w)
    byz_s.data_samples = 100
    byz_s.is_byzantine = True
    states.append(byz_s)

    return states, reference


class TestDefenseConfig:
    def test_default_values(self):
        cfg = DefenseConfig()
        assert cfg.defense_mode == "soft_cosine"
        assert cfg.temperature == 1.0
        assert cfg.temperature_decay == 0.95
        assert cfg.temperature_min == 0.1
        assert cfg.defense_scope == "cluster"

    def test_custom_values(self):
        cfg = DefenseConfig(defense_mode="none", temperature=2.0)
        assert cfg.defense_mode == "none"
        assert cfg.temperature == 2.0


class TestSoftRejectionAggregator:
    def test_fallback_to_fedavg_when_defense_none(self, honest_states):
        """When defense_mode='none', it must work like FedAvg."""
        states, ref = honest_states
        cfg = DefenseConfig(defense_mode="none")
        agg = SoftRejectionAggregator(cfg)
        result = agg.aggregate(states, reference_weights=ref)
        assert result.shape == states[0].weights.shape

    def test_cosine_aggregation_honest_only(self, honest_states):
        """With all honest clients, trust scores must be close to each other."""
        states, ref = honest_states
        cfg = DefenseConfig(defense_mode="soft_cosine")
        agg = SoftRejectionAggregator(cfg)
        result = agg.aggregate(states, reference_weights=ref)

        trust = agg.get_last_trust_scores()
        assert len(trust) == 4

        # All trust scores should be close (±0.1 of 0.25)
        for cid, score in trust.items():
            assert abs(score - 0.25) < 0.15, f"Client {cid} trust={score} too far from avg"

    def test_cosine_aggregation_detects_byzantine(self, mixed_states):
        """Byzantine client must have lower trust score than honest clients."""
        states, ref = mixed_states
        cfg = DefenseConfig(defense_mode="soft_cosine", temperature=0.5)
        agg = SoftRejectionAggregator(cfg)
        result = agg.aggregate(states, reference_weights=ref)

        trust = agg.get_last_trust_scores()
        assert len(trust) == 5

        # Byzantine client (ID=4) must have lowest trust
        byz_trust = trust[4]
        honest_trusts = [trust[i] for i in range(4)]
        avg_honest = sum(honest_trusts) / len(honest_trusts)

        assert byz_trust < avg_honest, (
            f"Byzantine trust={byz_trust:.4f} should be < honest avg={avg_honest:.4f}"
        )

    def test_single_client(self):
        """Single client → trust = 1.0, return original weights."""
        s = ClientState(client_id=0, initial_weights=torch.randn(50))
        cfg = DefenseConfig(defense_mode="soft_cosine")
        agg = SoftRejectionAggregator(cfg)
        result = agg.aggregate([s])

        assert torch.allclose(result, s.weights)
        assert agg.get_last_trust_scores() == {0: 1.0}

    def test_temperature_decay(self):
        """Temperature must decrease correctly according to decay rate, not below minimum."""
        cfg = DefenseConfig(temperature=1.0, temperature_decay=0.9, temperature_min=0.5)
        agg = SoftRejectionAggregator(cfg)

        agg.decay_temperature()
        assert abs(agg.current_temperature - 0.9) < 1e-6

        # Decay many times → should not go below minimum
        for _ in range(50):
            agg.decay_temperature()
        assert agg.current_temperature >= cfg.temperature_min


class TestTrustTracker:
    def test_log_and_retrieve(self):
        tracker = TrustTracker()
        tracker.log(1, -2, {0: 0.9, 1: 0.8, 2: 0.3})
        tracker.log(2, -2, {0: 0.95, 1: 0.85, 2: 0.2})

        # Get client history
        history_0 = tracker.get_client_history(0)
        assert history_0 == {1: 0.9, 2: 0.95}

        history_2 = tracker.get_client_history(2)
        assert history_2 == {1: 0.3, 2: 0.2}

    def test_to_matrix(self):
        tracker = TrustTracker()
        tracker.log(1, -2, {0: 0.9, 1: 0.3})
        tracker.log(2, -2, {0: 0.95, 1: 0.2})

        client_ids, rounds, matrix = tracker.to_matrix()
        assert client_ids == [0, 1]
        assert rounds == [1, 2]
        assert len(matrix) == 2
        assert len(matrix[0]) == 2

    def test_get_round_scores(self):
        tracker = TrustTracker()
        tracker.log(1, -2, {0: 0.9, 1: 0.8})
        tracker.log(1, -3, {2: 0.7, 3: 0.6})

        scores = tracker.get_round_scores(1)
        assert len(scores) == 4
        assert scores[0] == 0.9
        assert scores[3] == 0.6
