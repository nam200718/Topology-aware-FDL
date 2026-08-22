import torch
import pytest

from src.config import ClientConfig
from src.core.updater import PyTorchLocalUpdater
from src.core.loss import compute_binomial_loss_weights
from src.core.interfaces import ClientState


class TestBinomialScheduleProperties:
    def test_partition_of_unity(self):
        for r in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            alphas = compute_binomial_loss_weights(r)[3:]
            assert sum(alphas) == pytest.approx(1.0, abs=1e-9)

    def test_extremes_match_legacy_routing(self):
        # IID -> pure root (legacy r>=0.85 hard route)
        ar, ap, al = compute_binomial_loss_weights(1.0)[3:]
        assert ar == pytest.approx(1.0) and ap == 0.0 and al == 0.0
        # Extreme skew -> local-dominant with anchored root floor
        # (raw anchor a=0.15 normalizes to a/(1+a) since sum(lambda)=1+a)
        ar, _, _ = compute_binomial_loss_weights(0.0, anchor_min=0.15)[3:]
        assert ar == pytest.approx(0.15 / 1.15)

    def test_parent_peaks_at_moderate_skew(self):
        lp_lo = compute_binomial_loss_weights(0.3)[4]
        lp_mid = compute_binomial_loss_weights(0.5)[4]
        lp_hi = compute_binomial_loss_weights(0.7)[4]
        assert lp_mid > lp_lo and lp_mid > lp_hi

    def test_updater_binomial_path_runs(self):
        from torch.utils.data import TensorDataset
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        images = torch.randn(12, 1, 28, 28)
        labels = torch.randint(0, 10, (12,))
        ds = TensorDataset(images, labels)

        cfg = ClientConfig(
            use_ensemble=True,
            hierarchical_ensemble=True,
            compute_optimization_mode="shared_backbone",
            head_training_schedule="binomial",
            total_local_steps=3,
            local_steps=1,
        )
        state = ClientState(0, torch.nn.utils.parameters_to_vector(
            updater.multihead_model.parameters()).detach())
        updated = updater.update(state, ds, cfg, None)
        assert updated.weights.shape == state.weights.shape
        assert updated.head_steps["root"] >= 0


class TestTwoStageFreeze:
    def _make_updater_and_model(self):
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        return updater, updater.multihead_model

    def test_freeze_blocks_backbone_only(self):
        _, model = self._make_updater_and_model()
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        updater._freeze_backbone(model, frozen=True)
        for name, p in model.named_parameters():
            is_head = name.startswith(("fc2_root", "fc2_parent", "fc2_local"))
            assert p.requires_grad == is_head, name
        # BatchNorm statistics paused while frozen
        bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        if bn_modules:
            assert all(not m.training for m in bn_modules)
        updater._freeze_backbone(model, frozen=False)
        assert all(p.requires_grad for p in model.parameters())
        assert model.training

    def test_two_stage_engages_on_high_cardinality_config(self):
        from torch.utils.data import TensorDataset
        updater, model = self._make_updater_and_model()
        images = torch.randn(10, 1, 28, 28)
        labels = torch.randint(0, 10, (10,))
        ds = TensorDataset(images, labels)

        cfg = ClientConfig(
            use_ensemble=True,
            hierarchical_ensemble=True,
            compute_optimization_mode="shared_backbone",
            high_cardinality_two_stage=True,
            two_stage_min_classes=5,   # force engagement on a 10-class problem
            head_training_schedule="binomial",
            total_local_steps=4,
            local_steps=1,
        )
        state = ClientState(0, torch.nn.utils.parameters_to_vector(
            model.parameters()).detach())
        updated = updater.update(state.copy(), ds, cfg, None)
        assert updated.weights is not None

    def test_two_stage_disabled_for_low_cardinality(self):
        """With the default 100-class gate, a 10-class run must not freeze."""
        from torch.utils.data import TensorDataset
        updater, model = self._make_updater_and_model()
        images = torch.randn(8, 1, 28, 28)
        labels = torch.randint(0, 10, (8,))
        ds = TensorDataset(images, labels)

        cfg = ClientConfig(
            use_ensemble=True,
            hierarchical_ensemble=True,
            compute_optimization_mode="shared_backbone",
            high_cardinality_two_stage=True,
            head_training_schedule="piecewise",
            total_local_steps=3,
            local_steps=1,
        )
        state = ClientState(0, torch.nn.utils.parameters_to_vector(
            model.parameters()).detach())
        frozen_seen = []

        original_freeze = updater._freeze_backbone
        updater._freeze_backbone = lambda m, frozen: frozen_seen.append(frozen)
        updater.update(state, ds, cfg, None)
        assert len(frozen_seen) == 0


class TestSAFRRouting:
    def _engine(self, enabled=True, window=4, tau=4.0):
        """Minimal engine shell exposing what _safr_blend_weights needs."""
        import types
        from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine

        engine = object.__new__(HierarchicalEnsembleEngine)
        engine.last_sampled_round = {0: 10}
        engine.config = types.SimpleNamespace(
            clients=ClientConfig(
                s_afr_enabled=enabled,
                s_afr_staleness_window=window,
                s_afr_fade_tau=tau,
            ),
        )
        return engine

    def test_iid_shortcut_routes_to_root_regardless(self):
        engine = self._engine()
        state = types = __import__("types").SimpleNamespace(r_skew=0.9)
        assert engine._safr_blend_weights(state, 0, 12, 0.85) == (0.0, 0.0, 1.0)

    def test_disabled_returns_dynamic(self):
        engine = self._engine(enabled=False)
        state = __import__("types").SimpleNamespace(r_skew=0.5)
        assert engine._safr_blend_weights(state, 0, 12, 0.85) is None

    def test_fresh_client_keeps_dynamic_weighting(self):
        engine = self._engine()
        state = __import__("types").SimpleNamespace(r_skew=0.5)
        # round 13 - last sampled 10 = staleness 3 <= window 4
        assert engine._safr_blend_weights(state, 0, 13, 0.85) is None

    def test_stale_client_fades_toward_root(self):
        engine = self._engine()
        state = __import__("types").SimpleNamespace(
            r_skew=0.5, ensemble_alpha=[0.3, 0.3, 0.4])
        w_l, w_p, w_r = engine._safr_blend_weights(state, 0, 18, 0.85)
        # staleness 8, tau 4 -> fade = exp(-2)
        fade = 2.718281828 ** (-2.0)
        assert w_l == pytest.approx(0.3 * fade)
        assert w_p == pytest.approx(0.3 * fade)
        assert w_r == pytest.approx(1.0 - 0.6 * fade)

    def test_very_stale_client_converges_to_root(self):
        engine = self._engine()
        state = __import__("types").SimpleNamespace(
            r_skew=0.5, ensemble_alpha=[0.3, 0.3, 0.4])
        _, _, w_r = engine._safr_blend_weights(state, 0, 60, 0.85)
        assert w_r > 0.99

    def test_never_sampled_client_is_passthrough(self):
        engine = self._engine()
        engine.last_sampled_round = {}
        state = __import__("types").SimpleNamespace(r_skew=0.5)
        assert engine._safr_blend_weights(state, 7, 18, 0.85) is None


class TestTop2Routing:
    def _engine_with_affinity(self, own_sim):
        """Build a minimal engine shell exposing only the pieces _apply_top2_routing needs."""
        from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine
        import types

        engine = object.__new__(HierarchicalEnsembleEngine)
        engine.client_cluster_affinities = {0: (own_sim, 0.1)}
        engine.cluster_sync_sim_threshold = 0.5
        engine.config = types.SimpleNamespace(
            clients=ClientConfig(top2_routing=True),
        )
        return engine

    def test_strong_affinity_suppresses_local(self):
        engine = self._engine_with_affinity(own_sim=0.8)
        w_l, w_p, w_r = engine._apply_top2_routing((0.4, 0.3, 0.3), 0)
        assert w_l == pytest.approx(0.0)
        assert (w_p + w_r) == pytest.approx(1.0)
        assert w_p == pytest.approx(0.5) and w_r == pytest.approx(0.5)

    def test_weak_affinity_suppresses_parent(self):
        engine = self._engine_with_affinity(own_sim=0.2)
        w_l, w_p, w_r = engine._apply_top2_routing((0.4, 0.3, 0.3), 0)
        assert w_p == pytest.approx(0.0)
        assert (w_l + w_r) == pytest.approx(1.0)

    def test_no_affinity_data_is_passthrough(self):
        engine = self._engine_with_affinity(own_sim=0.8)
        engine.client_cluster_affinities = {}
        assert engine._apply_top2_routing((0.4, 0.3, 0.3), 0) == (0.4, 0.3, 0.3)

    def test_disabled_flag_is_passthrough(self):
        engine = self._engine_with_affinity(own_sim=0.8)
        engine.config.clients.top2_routing = False
        assert engine._apply_top2_routing((0.4, 0.3, 0.3), 0) == (0.4, 0.3, 0.3)
