import torch
import pytest

from src.core.aggregator import (
    DeltaSpaceRobustAggregator,
    FedAvgAggregator,
    bound_update_norms,
    compute_buffer_slices,
    coordinate_median,
    soft_cosine_trust,
    trimmed_mean,
)
from src.core.interfaces import ClientState
from src.core.model import MultiHeadSimpleCNN


def _make_states(weight_vectors):
    return [ClientState(i, w.clone()) for i, w in enumerate(weight_vectors)]


class TestSoftCosineTrust:
    def test_sign_flipped_update_gets_near_zero_trust(self):
        d = 64
        honest = [torch.randn(d) * 0.01 + torch.ones(d) * 0.1 for _ in range(9)]
        attacker = -(honest[0])  # exact opposite direction
        trust = soft_cosine_trust(torch.stack(honest + [attacker]), temperature=0.5)
        assert trust[-1].item() < 0.01
        assert trust[:-1].min().item() > trust[-1].item()

    def test_uniform_when_all_aligned(self):
        d = 32
        base = torch.randn(d)
        stack = torch.stack([base * s for s in [0.5, 1.0, 2.0]])
        trust = soft_cosine_trust(stack, temperature=0.5)
        assert torch.allclose(trust, torch.full((3,), 1 / 3), atol=1e-5)

    def test_degenerate_opposing_updates_get_uniform_trust(self):
        """When updates exactly cancel there is no dominant direction; trust
        must fall back to uniform rather than amplify float noise."""
        d = 32
        base = torch.randn(d)
        stack = torch.stack([base, -base])
        trust = soft_cosine_trust(stack, temperature=0.5)
        assert torch.allclose(trust, torch.full((2,), 0.5), atol=1e-6)

    def test_norm_bounding_neutralizes_inflation(self):
        """A sign-flipped, magnitude-inflated attacker must not dominate the
        aggregated delta when norm bounding is active."""
        d = 32
        honest = torch.randn(d) * 0.1
        attacker = -honest * 1000.0

        agg_bounded = DeltaSpaceRobustAggregator(mode="soft_cosine", temperature=0.5, norm_bound_k=3.0)
        agg_free = DeltaSpaceRobustAggregator(mode="soft_cosine", temperature=0.5, norm_bound_k=None)

        result_bounded = agg_bounded.aggregate_deltas([honest, attacker])
        result_free = agg_free.aggregate_deltas([honest, attacker])

        assert result_bounded.norm() < 3.1 * honest.norm()
        assert result_free.norm() > 100.0 * honest.norm()

    def test_bound_preserves_direction(self):
        d = 16
        v = torch.randn(d)
        bounded = bound_update_norms(v.unsqueeze(0), 3.0).squeeze(0)
        cos = torch.nn.functional.cosine_similarity(bounded, v, dim=0)
        assert cos.item() == pytest.approx(1.0, abs=1e-6)

    def test_single_delta_returns_one(self):
        trust = soft_cosine_trust(torch.randn(1, 16), temperature=0.5)
        assert trust.item() == 1.0


class TestTrimmedMean:
    def test_rejects_coordinate_outlier(self):
        stack = torch.zeros(10, 8)
        stack[9, 0] = 100.0  # one client poisons one coordinate
        result = trimmed_mean(stack, beta=0.2)
        assert result[0].item() == pytest.approx(0.0)
        assert torch.all(result[1:] == 0)

    def test_equals_manual_trim(self):
        torch.manual_seed(0)
        stack = torch.randn(10, 5)
        k = int(10 * 0.2)
        expected = torch.sort(stack, dim=0).values[k:10 - k].mean(dim=0)
        assert torch.allclose(trimmed_mean(stack, 0.2), expected)

    def test_small_n_falls_back_to_mean(self):
        stack = torch.randn(4, 5)
        assert torch.allclose(trimmed_mean(stack, 0.2), stack.mean(dim=0))


class TestCoordinateMedian:
    def test_majority_wins(self):
        stack = torch.zeros(5, 3)
        stack[0] = 99.0
        med = coordinate_median(stack)
        assert torch.all(med == 0)


class TestDeltaSpaceRobustAggregator:
    def test_sign_flip_recovery(self):
        """Aggregate of mostly-honest updates with a sign-flip attacker stays
        strictly closer to the honest mean than the naive mean."""
        d = 128
        honest_deltas = [torch.randn(d) * 0.05 + 0.2 for _ in range(8)]
        attacker = -honest_deltas[0]  # exact opposite direction of one honest client

        agg = DeltaSpaceRobustAggregator(mode="soft_cosine", temperature=0.2)
        reference = torch.zeros(d)
        robust_result = agg.aggregate_deltas(honest_deltas + [attacker], reference)
        honest_mean = torch.stack(honest_deltas).mean(dim=0)
        naive_result = torch.stack(honest_deltas + [attacker]).mean(dim=0)

        err_robust = (robust_result - honest_mean).norm()
        err_naive = (naive_result - honest_mean).norm()
        assert err_robust < err_naive

        # The attacker must receive less trust than every honest client
        trust = soft_cosine_trust(torch.stack(honest_deltas + [attacker]), temperature=0.2)
        assert trust[-1] < trust[:-1].min()

    def test_trimmed_mode(self):
        agg = DeltaSpaceRobustAggregator(mode="trimmed_mean", beta=0.2)
        d = 16
        deltas = [torch.zeros(d) for _ in range(9)]
        deltas[0][3] = 50.0
        result = agg.aggregate_deltas(deltas)
        assert result[3].item() == pytest.approx(0.0)

    def test_buffer_coordinates_use_median(self):
        agg = DeltaSpaceRobustAggregator(mode="soft_cosine", buffer_slices=[(4, 6)])
        deltas = []
        for i in range(5):
            d_i = torch.ones(8) * 0.01 * (i + 1)
            d_i[4:6] = float(i + 1)  # buffer coords: 1..5 -> median 3
            deltas.append(d_i)
        result = agg.aggregate_deltas(deltas)
        assert result[4].item() == pytest.approx(3.0)
        assert result[5].item() == pytest.approx(3.0)

    def test_empty_raises(self):
        agg = DeltaSpaceRobustAggregator(mode="soft_cosine")
        with pytest.raises(ValueError):
            agg.aggregate_deltas([])


class TestBufferSlices:
    def test_resnet9_buffers_detected(self):
        from src.core.model import MultiHeadResNet9
        model = MultiHeadResNet9(in_channels=3, num_classes=10)
        slices = compute_buffer_slices(model)
        # Every conv_block carries BatchNorm running_mean/var (+ tracked count)
        assert len(slices) > 0
        assert all(start < end for start, end in slices)

    def test_slices_cover_all_buffer_coords(self):
        from src.core.model import MultiHeadResNet9
        model = MultiHeadResNet9(in_channels=3, num_classes=10)
        param_keys = {k for k, _ in model.named_parameters()}
        expected = sum(v.numel() for k, v in model.state_dict().items()
                       if k not in param_keys)
        covered = sum(end - start for start, end in compute_buffer_slices(model))
        assert covered == expected

    def test_param_only_vector_has_no_slices(self):
        import torch.nn as nn
        model = nn.Linear(4, 2)  # no buffers
        assert compute_buffer_slices(model) == []


class TestFedAvgPassthrough:
    def test_delta_space_fedavg_equals_plain_fedavg(self):
        """Linear-aggregator identity: mean(w_i - ref) + ref == mean(w_i).
        Guarantees the engine's delta-space path preserves FedAvg semantics."""
        torch.manual_seed(0)
        refs = torch.randn(32)
        states = _make_states([refs + torch.randn(32) * 0.1 for _ in range(5)])
        plain = FedAvgAggregator().aggregate(states)

        deltas = [s.weights - refs for s in states]
        delta_result = refs + torch.stack(deltas).mean(dim=0)
        assert torch.allclose(plain, delta_result, atol=1e-6)
