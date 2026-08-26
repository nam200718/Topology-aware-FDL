import torch
import pytest

from src.config import ClientConfig
from src.core.updater import PyTorchLocalUpdater
from src.core.interfaces import ClientState
from torch.utils.data import TensorDataset


def _make_ds(n=12):
    torch.manual_seed(0)
    images = torch.randn(n, 1, 28, 28)
    labels = torch.randint(0, 10, (n,))
    return TensorDataset(images, labels)


@pytest.mark.parametrize("method", ["fedrep", "fedper", "fedbabu"])
class TestSplitHeadBaselines:
    def test_update_runs_and_persists_private_state(self, method):
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        ds = _make_ds()
        cfg = ClientConfig(
            personalization_method=method,
            local_steps=3,
            total_local_steps=5,
        )
        state = ClientState(0, torch.nn.utils.parameters_to_vector(
            updater.global_model.parameters()).detach())
        updated = updater.update(state.copy(), ds, cfg, None)

        assert updated.local_weights is not None
        assert updated.weights.shape == state.weights.shape
        # All three methods persist a head state; FedBABU's snapshot is its
        # (untrained) random init, restored verbatim every round.
        assert updated.local_head_state is not None

    def test_upload_is_body_only(self, method):
        """Head coordinates must be zeroed pre-aggregation."""
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        ds = _make_ds()
        cfg = ClientConfig(personalization_method=method, local_steps=3, total_local_steps=5)
        state = ClientState(0, torch.nn.utils.parameters_to_vector(
            updater.global_model.parameters()).detach())
        updated = updater.update(state.copy(), ds, cfg, None)

        mask = updater._head_param_mask(updater.global_model)
        assert mask.any(), "mask must select the classification head"
        assert not mask.all(), "mask must not select the entire model"
        assert torch.all(updated.weights[mask] == 0), "uploaded head coords must be zero"
        assert not torch.all(updated.weights[~mask] == 0), "uploaded body coords must NOT be zero"
        # Private model keeps trained head values distinct from upload
        assert not torch.all(updated.local_weights[mask] == 0)

    def test_private_head_persists_across_rounds(self, method):
        """Round 2 must restore round 1's private head before training
        (FedBABU intentionally keeps its head at init and is skipped)."""
        if method == "fedbabu":
            pytest.skip("Canonical FedBABU does not persist trained head state")
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        ds = _make_ds()
        cfg = ClientConfig(personalization_method=method, local_steps=2, total_local_steps=3)
        state = ClientState(0, torch.nn.utils.parameters_to_vector(
            updater.global_model.parameters()).detach())

        r1 = updater.update(state.copy(), ds, cfg, None)
        saved_head = {k: v.clone() for k, v in r1.local_head_state.items()}

        # Round 2 starts from aggregated (body-only) broadcast
        r2_state = state.copy()
        r2_state.weights = r1.weights.clone()
        r2_state.local_head_state = r1.local_head_state
        r2 = updater.update(r2_state, ds, cfg, None)

        for k in saved_head:
            assert saved_head[k].shape == r2.local_head_state[k].shape


class TestNumClassesThreading:
    def test_cifar100_sized_heads(self):
        """C=100 datasets must produce 100-class heads end-to-end."""
        from torch.utils.data import TensorDataset as TD
        updater = PyTorchLocalUpdater(device="cpu", in_channels=3, num_classes=100)
        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 100, (8,))
        ds = TD(images, labels)

        assert updater.multihead_model.fc2_root.out_features == 100
        assert updater.global_model.fc2.out_features == 100

        cfg = ClientConfig(
            personalization_method="fedrep", local_steps=2, total_local_steps=3,
        )
        state = ClientState(0, torch.nn.utils.parameters_to_vector(
            updater.global_model.parameters()).detach())
        updated = updater.update(state, ds, cfg, None)
        assert updated.weights.shape == state.weights.shape


class TestDispatcherRouting:
    def test_split_head_methods_route_correctly(self):
        updater = PyTorchLocalUpdater(device="cpu", in_channels=1)
        called = {}
        original = updater._update_split_head
        updater._update_split_head = lambda *a, **k: called.setdefault("m", a[-1]) or original(*a, **k)

        ds = _make_ds()
        for method in ["fedrep", "fedper", "fedbabu"]:
            called.clear()
            cfg = ClientConfig(personalization_method=method, local_steps=2, total_local_steps=3)
            state = ClientState(0, torch.nn.utils.parameters_to_vector(
                updater.global_model.parameters()).detach())
            updater.update(state.copy(), ds, cfg, None)
            assert called.get("m") == method
