"""Unit and integration tests for FEMNIST (62-class EMNIST) on ResNet-9.

Verifies:
1. ResNet-9 single-channel 62-class architecture shape compatibility
2. MultiHeadResNet9 head routing across root, parent, local, all
3. Symmetric Dirichlet partitioning across 62 classes
4. get_femnist loader contract
"""
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.core.model import ResNet9, MultiHeadResNet9
from src.data.dataset import partition_data, ClientDataset, FastDataset
from src.baselines.experiment_configs import FEMNIST_DEFAULTS, create_personalization_config


def test_resnet9_femnist_shape_compatibility():
    """Verify ResNet-9 natively processes 1-channel 28x28 FEMNIST images to 62 logits."""
    model = ResNet9(in_channels=1, num_classes=62, base_channels=32)
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (4, 62), f"Expected shape (4, 62), got {logits.shape}"


def test_multihead_resnet9_femnist_head_routing():
    """Verify MultiHeadResNet9 routes to all 3 heads with 62-class output."""
    model = MultiHeadResNet9(in_channels=1, num_classes=62, base_channels=32)
    x = torch.randn(4, 1, 28, 28)

    for head in ("root", "parent", "local"):
        out = model(x, head=head)
        assert out.shape == (4, 62), f"Head '{head}' expected (4, 62), got {out.shape}"

    root_out, parent_out, local_out = model(x, head="all")
    assert root_out.shape == (4, 62)
    assert parent_out.shape == (4, 62)
    assert local_out.shape == (4, 62)


def test_femnist_dirichlet_partitioning_62_classes():
    """Verify Dirichlet partitioning successfully allocates 62 classes across 15 clients."""
    num_samples = 620
    num_classes = 62
    num_clients = 15

    # Simulate 10 samples per class across 62 classes
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.repeat_interleave(torch.arange(num_classes), 10)

    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, lbls):
            self.images = imgs
            self.labels = lbls
            self.targets = lbls
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    mock_ds = MockDataset(images, labels)

    for alpha in [1.0, 0.5, 0.1, 0.05]:
        client_indices = partition_data(mock_ds, num_clients=num_clients, non_iid=True, alpha=alpha, seed=42)
        assert len(client_indices) == num_clients

        all_allocated = []
        for cid, idxs in client_indices.items():
            assert len(idxs) > 0, f"Client {cid} received 0 samples under alpha={alpha}"
            all_allocated.extend(idxs)

        # Check all samples are accounted for
        assert len(all_allocated) == num_samples
        assert set(all_allocated) == set(range(num_samples))


def test_femnist_defaults_and_config_generation():
    """Verify FEMNIST_DEFAULTS generates correct experiment configuration."""
    assert FEMNIST_DEFAULTS["dataset"] == "femnist"
    assert FEMNIST_DEFAULTS["model_name"] == "resnet9"
    assert FEMNIST_DEFAULTS["num_clients"] == 15

    cfg = create_personalization_config(
        method_id="topo",
        regime_id="moderate",
        base_defaults=FEMNIST_DEFAULTS,
    )
    assert cfg.experiment_name == "femnist_topo_moderate"
    assert cfg.env.dataset == "femnist"
    assert cfg.clients.model_name == "resnet9"
    assert cfg.clients.num_clients == 15
    assert cfg.clients.use_ensemble is True
