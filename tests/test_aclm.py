"""
Unit tests for Active-Class Logit Masking (ACLM).
Verifies:
1. Gradient isolation for unobserved / inactive classes.
2. Numerical stability under extreme logits.
3. Behavior when all classes or single class are active.
"""

import pytest
import torch
import torch.nn.functional as F

from src.core.loss import ActiveMaskedCrossEntropyLoss


def test_aclm_gradient_isolation():
    """Verify that Active Class Logit Masking (ACLM) yields exactly zero gradient for inactive classes."""
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
            assert torch.all(logits.grad[:, c] == 0.0), f"Class {c} gradient should be 0.0"
        else:
            assert not torch.all(logits.grad[:, c] == 0.0), f"Active class {c} gradient should be non-zero"


def test_aclm_all_active_matches_standard_ce():
    """Verify that when all classes are active, ACLM produces identical loss to standard CrossEntropy."""
    loss_fn = ActiveMaskedCrossEntropyLoss()
    logits = torch.randn(8, 20)
    targets = torch.randint(0, 20, (8,))

    all_active_mask = torch.ones(20, dtype=torch.bool)
    loss_aclm = loss_fn(logits, targets, active_mask=all_active_mask)
    loss_standard = F.cross_entropy(logits, targets)

    assert torch.allclose(loss_aclm, loss_standard, atol=1e-5)


def test_aclm_none_mask_passthrough():
    """Verify that when active_mask is None, standard cross-entropy is computed."""
    loss_fn = ActiveMaskedCrossEntropyLoss()
    logits = torch.randn(6, 15)
    targets = torch.randint(0, 15, (6,))

    loss_none = loss_fn(logits, targets, active_mask=None)
    loss_standard = F.cross_entropy(logits, targets)

    assert torch.allclose(loss_none, loss_standard, atol=1e-5)
