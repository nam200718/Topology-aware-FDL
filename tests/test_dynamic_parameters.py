import math
import pytest
import torch
from src.core.loss import compute_hill_number_r_skew, compute_dynamic_binomial_loss_weights


def test_hill_number_r_skew_dirac():
    # 1-class Dirac distribution: H = 0 -> R_skew = 0.0
    probs = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    r_skew = compute_hill_number_r_skew(probs, num_classes=10)
    assert math.isclose(r_skew, 0.0, abs_tol=1e-5), f"Expected 0.0 for Dirac, got {r_skew}"


def test_hill_number_r_skew_uniform_c():
    # Uniform 10-class distribution: H = ln(10) -> R_skew = 1.0
    probs = torch.full((10,), 0.1)
    r_skew = compute_hill_number_r_skew(probs, num_classes=10)
    assert math.isclose(r_skew, 1.0, abs_tol=1e-5), f"Expected 1.0 for Uniform C, got {r_skew}"


def test_hill_number_r_skew_uniform_k():
    # Uniform 2-class distribution: H = ln(2) -> R_skew = (2-1)/(10-1) = 1/9 ~ 0.1111
    probs = torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    r_skew = compute_hill_number_r_skew(probs, num_classes=10)
    assert math.isclose(r_skew, 1.0 / 9.0, abs_tol=1e-5), f"Expected 1/9, got {r_skew}"


def test_dynamic_binomial_loss_weights():
    # Extreme skew: R_skew = 0, local_classes = 2, num_classes = 10, num_clusters = 3
    lr, lp, ll, ar, ap, al = compute_dynamic_binomial_loss_weights(
        r_skew=0.0, num_classes=10, local_classes=2, num_clusters=3
    )
    # Sum of alphas should be 1.0
    assert math.isclose(ar + ap + al, 1.0, abs_tol=1e-5)
    # Under R_skew = 0, lp = 0, ll = 1, lr = max(1/6, 2/10) = 0.20
    assert math.isclose(lp, 0.0, abs_tol=1e-5)
    assert math.isclose(ll, 1.0, abs_tol=1e-5)
    assert math.isclose(lr, 0.20, abs_tol=1e-5)

    # IID: R_skew = 1.0
    lr, lp, ll, ar, ap, al = compute_dynamic_binomial_loss_weights(
        r_skew=1.0, num_classes=10, local_classes=10, num_clusters=3
    )
    assert math.isclose(ar + ap + al, 1.0, abs_tol=1e-5)
    assert math.isclose(ar, 1.0, abs_tol=1e-5)
    assert math.isclose(ap, 0.0, abs_tol=1e-5)
    assert math.isclose(al, 0.0, abs_tol=1e-5)
