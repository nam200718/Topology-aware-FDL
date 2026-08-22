"""
Unit tests for the Streamlined Hierarchical Residual Classifier (HRC / Streamlined HEP).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.model import (
    HierarchicalResidualLinear,
    HierarchicalResidualResNet9,
    HierarchicalResidualMobileNetV3Small,
)
from src.core.loss import ActiveMaskedCrossEntropyLoss, compute_hierarchical_residual_penalty


def test_zero_initialization():
    """Verify that at initialization, Delta_W_cluster and Delta_W_local are identically zero."""
    layer = HierarchicalResidualLinear(in_features=64, num_classes=10)
    assert torch.all(layer.weight_cluster == 0.0)
    assert torch.all(layer.bias_cluster == 0.0)
    assert torch.all(layer.weight_local == 0.0)
    assert torch.all(layer.bias_local == 0.0)
    assert not torch.all(layer.weight_global == 0.0)

    # Verify that effective weights equal global weights at initialization
    w_eff, b_eff = layer.get_effective_weights()
    assert torch.allclose(w_eff, layer.weight_global)
    assert torch.allclose(b_eff, layer.bias_global)


def test_forward_pass_equivalence():
    """Verify that forward pass matches standard linear projection of effective weights."""
    layer = HierarchicalResidualLinear(in_features=32, num_classes=5)
    x = torch.randn(8, 32)
    out = layer(x)

    w_eff, b_eff = layer.get_effective_weights()
    expected = F.linear(x, w_eff, b_eff)
    assert torch.allclose(out, expected)


def test_aclm_gradient_isolation():
    """Verify that Active Class Logit Masking (ACLM) yields zero gradient for inactive classes."""
    loss_fn = ActiveMaskedCrossEntropyLoss()
    logits = torch.randn(4, 10, requires_grad=True)
    targets = torch.tensor([1, 2, 1, 2])

    # Only classes 1 and 2 are active
    active_mask = torch.tensor([False, True, True, False, False, False, False, False, False, False])
    loss = loss_fn(logits, targets, active_mask=active_mask)
    loss.backward()

    # Inactive classes (0, 3, 4, 5, 6, 7, 8, 9) must have exactly 0.0 gradient
    for c in range(10):
        if not active_mask[c]:
            assert torch.all(logits.grad[:, c] == 0.0)
        else:
            assert not torch.all(logits.grad[:, c] == 0.0)


def test_residual_shrinkage_penalties():
    """Verify that residual shrinkage correctly penalizes local residuals under IID and cluster under skew."""
    model = HierarchicalResidualResNet9(in_channels=3, num_classes=10, base_channels=16)
    
    # Inject non-zero weights into cluster and local residuals
    with torch.no_grad():
        model.classifier.weight_cluster.fill_(1.0)
        model.classifier.weight_local.fill_(1.0)

    # Under pure IID (r_skew = 1.0), local residual is penalized, cluster penalty is 0
    pen_iid = compute_hierarchical_residual_penalty(model, r_skew=1.0, mu=1e-2)
    # Under extreme skew (r_skew = 0.0), cluster residual is penalized, local penalty is 0
    pen_skew = compute_hierarchical_residual_penalty(model, r_skew=0.0, mu=1e-2)

    assert pen_iid > 0.0
    assert pen_skew > 0.0
    assert torch.allclose(pen_iid, pen_skew)


def test_hierarchical_resnet9_and_mobilenet_forward():
    """Verify end-to-end forward pass and feature extraction for ResNet9 and MobileNetV3."""
    x = torch.randn(2, 3, 32, 32)
    
    resnet = HierarchicalResidualResNet9(in_channels=3, num_classes=10, base_channels=16)
    f_res = resnet.extract_features(x)
    assert f_res.shape == (2, 128)
    out_res = resnet(x)
    assert out_res.shape == (2, 10)

    mobilenet = HierarchicalResidualMobileNetV3Small(in_channels=3, num_classes=10)
    f_mob = mobilenet.extract_features(x)
    assert f_mob.shape == (2, 576)
    out_mob = mobilenet(x)
    assert out_mob.shape == (2, 10)
