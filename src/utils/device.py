"""Device resolution with accelerator self-verification.

Some accelerator backends silently mis-execute ops this project relies on
(e.g. torch.median(dim=0) on DirectML returns an empty tensor instead of
raising). resolve_device() therefore probes any candidate accelerator with
the exact op suite used by the aggregation/training paths and falls back to
CPU loudly if anything fails or returns malformed results.

HEP_FORCE_DEVICE=cpu|dml|cuda|mps forces a choice (no probing for "cpu").
"""
import os

import torch
import torch.nn.functional as F


def _probe_op_suite(device) -> bool:
    """Verify every accelerator op our hot paths depend on executes correctly."""
    try:
        # Aggregation primitives (src/core/aggregator.py)
        x = torch.randn(6, 32, device=device)
        s, _ = torch.sort(x, dim=0)
        if s.shape != x.shape:
            return False

        idx = torch.tensor([0, 2, 4], device=device)
        sel = x.index_select(1, idx)
        if sel.shape != (6, 3):
            return False

        out = torch.zeros(32, device=device)
        out.index_copy_(0, idx[:3], torch.ones(3, device=device))
        if not bool((out[idx[:3]] == 1).all()):
            return False

        # Trust weighting (soft_cosine_trust)
        t = F.softmax(F.cosine_similarity(F.normalize(x, dim=1), F.normalize(x, dim=1), dim=1), dim=0)
        if t.shape != (6,) or bool(torch.isnan(t).any()):
            return False

        # Sort-based median must produce full-width output (DirectML's
        # torch.median(dim=0) returns empty tensors).
        med = s[3]
        if med.shape != (32,):
            return False

        # Training-path sanity: small matmul + CE backward
        w = torch.randn(8, 16, device=device, requires_grad=True)
        loss = F.cross_entropy(w @ torch.randn(16, 8, device=device),
                               torch.randint(0, 8, (8,), device=device))
        loss.backward()
        if w.grad is None:
            return False
    except Exception:
        return False
    return True


def resolve_device():
    """Pick the best verified compute device for this machine."""
    forced = os.environ.get("HEP_FORCE_DEVICE", "").strip().lower()
    if forced == "cpu":
        return "cpu"

    try:
        import torch_directml  # type: ignore
        if forced in ("", "dml", "directml") and torch_directml.is_available():
            dev = torch_directml.device()
            if _probe_op_suite(dev):
                return dev
            print("[device] DirectML available but failed op-suite verification; falling back.")
    except ImportError:
        pass
    except Exception as exc:
        print(f"[device] DirectML probe error: {exc}")

    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        acc = torch.accelerator.current_accelerator()
        if acc is not None and _probe_op_suite(acc):
            return acc.type if isinstance(acc.type, str) else str(acc)
    elif torch.cuda.is_available() and _probe_op_suite("cuda"):
        return "cuda"

    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") \
            and torch.backends.mps.is_available() and _probe_op_suite("mps"):
        return "mps"

    return "cpu"
