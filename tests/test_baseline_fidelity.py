"""Baseline fidelity tests: canonical implementations of Ditto, FedBABU,
FedRep (compute parity), FedALA, and CFL primitives."""

import pytest
import torch
from torch.utils.data import TensorDataset

from src.config import ClientConfig
from src.core.interfaces import ClientState
from src.core.model import vector_to_model
from src.core.updater import PyTorchLocalUpdater


def _make_ds(n=12, seed=0):
    torch.manual_seed(seed)
    images = torch.randn(n, 1, 28, 28)
    labels = torch.randint(0, 10, (n,))
    return TensorDataset(images, labels)


def _state(updater, cid=0):
    return ClientState(cid, torch.nn.utils.parameters_to_vector(
        updater.global_model.parameters()).detach())


def _vec(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()


class TestDittoCanonicalAnchor:
    def test_proximal_anchor_is_received_weights(self):
        """With a dominant proximal penalty the personalized model must land
        near the RECEIVED global w^t even though the global leg drifted away
        (the previous implementation anchored post-update weights)."""
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        ds = _make_ds(n=32)
        cfg = ClientConfig(personalization_method="ditto",
                           ditto_lambda=15.0, local_lr=0.01, local_steps=30)
        state = _state(updater)
        w_received = state.weights.clone()

        updated = updater.update(state.copy(), ds, cfg, None)

        d = (updated.weights - w_received).norm()
        assert d.item() > 1e-4, "global leg must drift from received weights"

        dist_anchor = (updated.local_weights - w_received).norm().item()
        dist_global = (updated.local_weights - updated.weights).norm().item()
        # New-code signature: w_p converges toward the received anchor.
        # Old-code signature would keep w_p near the drifted global instead.
        assert dist_anchor < 0.5 * dist_global


class TestFedRepComputeParity:
    def test_total_passes_equal_local_steps(self):
        """Head+joint phases must sum to exactly local_steps loader passes
        once a private head exists (compute parity with other baselines)."""
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        base_ds = _make_ds(n=16)

        class CountingLoader:
            def __init__(self, ds):
                self.ds = ds
                self.exhaustions = 0

            def __iter__(self):
                self.exhaustions += 1
                imgs = torch.stack([self.ds[i][0] for i in range(len(self.ds))])
                labs = torch.stack([self.ds[i][1] for i in range(len(self.ds))])
                yield imgs, labs

        cfg = ClientConfig(personalization_method="fedrep",
                           local_steps=3, total_local_steps=5,
                           fedrep_head_epochs=1)
        state = _state(updater)

        # Round 1 (cold start): joint-only schedule -> local_steps passes.
        counting = CountingLoader(base_ds)
        updater._update_split_head(state.copy(), cfg, counting,
                                   cfg.local_steps, cfg.local_lr,
                                   state.weights.clone(), "fedrep")
        assert counting.exhaustions == cfg.local_steps

        # Round 2 (private head exists): e_head + joint == local_steps.
        head_state = {k: v.clone() for k, v in updater.global_model.fc2.state_dict().items()}
        r2 = state.copy()
        r2.local_head_state = head_state
        counting = CountingLoader(base_ds)
        updater._update_split_head(r2, cfg, counting,
                                   cfg.local_steps, cfg.local_lr,
                                   state.weights.clone(), "fedrep")
        assert counting.exhaustions == cfg.local_steps


class TestFedBABUCanonical:
    def test_head_stays_at_client_init_across_rounds(self):
        """The head must remain bit-identical to the client's initial random
        head across federated rounds, surviving body-only broadcasts."""
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        ds = _make_ds()
        cfg = ClientConfig(personalization_method="fedbabu", local_steps=3)
        state = _state(updater)

        head_init = _vec(updater.global_model.fc2).clone()
        numel = _vec(updater.global_model).numel()
        body_before = _vec(updater.global_model)[:numel - head_init.numel()].clone()

        r1 = updater.update(state.copy(), ds, cfg, None)
        assert torch.equal(head_init, _vec(updater.global_model.fc2).clone()), \
            "head must stay at init during federated rounds"

        # Round 2 broadcasts the body-only upload (zeroed head coords); the
        # client's init head must be restored rather than clobbered.
        r2_state = state.copy()
        r2_state.weights = r1.weights.clone()
        r2_state.local_head_state = r1.local_head_state
        r2 = updater.update(r2_state, ds, cfg, None)

        assert torch.equal(head_init, _vec(updater.global_model.fc2).clone())
        assert not torch.allclose(body_before,
                                  _vec(updater.global_model)[:numel - head_init.numel()]), \
            "body must train"

    def test_upload_is_body_only_and_head_state_persisted(self):
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        ds = _make_ds()
        cfg = ClientConfig(personalization_method="fedbabu", local_steps=2)
        state = _state(updater)
        updated = updater.update(state.copy(), ds, cfg, None)

        mask = updater._head_param_mask(updater.global_model)
        assert torch.all(updated.weights[mask] == 0), "upload must zero head coords"
        assert updated.local_head_state is not None
        for v in updated.local_head_state.values():
            assert float(v.abs().sum()) > 0, "persisted head is the random init"


class TestFedALA:
    def test_first_round_deactivates_ala(self):
        """Official semantics: ALA is inactive on the first communication
        iteration (received global == local); the convergence phase flag
        stays armed until the first adaptive aggregation actually runs."""
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        ds = _make_ds()
        cfg = ClientConfig(personalization_method="fedala", local_steps=2)
        state = _state(updater)
        updated = updater.update(state.copy(), ds, cfg, None)

        assert updated.weights.shape == state.weights.shape
        assert updated.ala_start_phase is True
        assert updated.ala_weights is None
        assert not torch.allclose(updated.weights, state.weights)

    def test_second_round_learns_bounded_weights(self):
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        ds = _make_ds()
        cfg = ClientConfig(personalization_method="fedala", local_steps=2)
        state = _state(updater)

        r1 = updater.update(state.copy(), ds, cfg, None)
        r2_state = state.copy()
        # Simulate cross-client aggregation drift: the received global model
        # must differ from the client's previous local model for ALA to arm.
        torch.manual_seed(7)
        r2_state.weights = r1.weights + 0.01 * torch.randn_like(r1.weights)
        r2_state.local_weights = r1.local_weights.clone()
        r2 = updater.update(r2_state, ds, cfg, None)

        assert r2.ala_weights is not None and len(r2.ala_weights) > 0
        for w in r2.ala_weights:
            assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0
        assert r2.ala_start_phase is False

    def test_routes_via_dispatcher(self):
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        called = {}
        original = updater._update_fedala

        def spy(*a, **k):
            called["x"] = True
            return original(*a, **k)

        updater._update_fedala = spy
        cfg = ClientConfig(personalization_method="fedala", local_steps=1)
        updater.update(_state(updater), _make_ds(), cfg, None)
        assert called.get("x")


class TestCFLPrimitives:
    def _engine_stub(self):
        from src.core.cfl_engine import CFLEngine
        eng = CFLEngine.__new__(CFLEngine)
        eng.cfl_eps = 0.1
        eng.cfl_eps_min = 1e-8
        eng.cfl_decrease_factor = 0.1
        eng.separation_observed = False
        return eng

    def test_separates_opposing_groups(self):
        eng = self._engine_stub()
        torch.manual_seed(0)
        v = torch.randn(64)
        deltas = {
            0: v + 0.01 * torch.randn(64),
            1: -v + 0.01 * torch.randn(64),
            2: v + 0.01 * torch.randn(64),
            3: -v + 0.01 * torch.randn(64),
        }
        split = eng._try_split([0, 1, 2, 3], deltas)
        assert split is not None, "opposing updates must trigger separation"
        a, b = frozenset(split[0]), frozenset(split[1])
        expected = {frozenset({0, 2}), frozenset({1, 3})}
        assert {a, b} == expected

    def test_no_split_below_threshold(self):
        eng = self._engine_stub()
        torch.manual_seed(1)
        base = torch.randn(32) + 10 * torch.ones(32)
        deltas = {cid: base + 0.001 * torch.randn(32) for cid in range(4)}
        assert eng._try_split(list(deltas.keys()), deltas) is None

    def test_refine_removes_negatively_correlated(self):
        eng = self._engine_stub()
        v = torch.randn(32)
        deltas = {0: v, 1: v + 0.01 * torch.randn(32), 2: -v}
        refined = eng._refine([0, 1, 2], deltas)
        assert set(refined) == {0, 1}

    def test_eps_decreases_without_separation(self):
        eng = self._engine_stub()
        eps0 = eng.cfl_eps
        torch.manual_seed(3)
        base = torch.randn(32)
        deltas = {cid: base + 0.001 * torch.randn(32) for cid in range(4)}
        members = list(range(4))
        eng.cluster_members = {0: members}
        eng.cluster_models = {0: torch.zeros(4)}
        eng.server_weights = torch.zeros(4)
        eng._update_clusters(deltas)
        assert eng.separation_observed is False
        assert eng.cfl_eps < eps0
        assert set().union(*eng.cluster_members.values()) == set(members)
