"""Per-batch latency probe for FedALA and CFL client paths (ResNet-9).

Measures forward+backward+optimizer step time per batch (after warmup),
matching the Table III 'Latency/Batch' convention. Run WITHOUT the main GPU
queue to avoid contention.

Usage:
    python scripts/profile_fedala_cfl_latency.py
"""

import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from src.config import ClientConfig
from src.core.interfaces import ClientState
from src.core.updater import PyTorchLocalUpdater


def _ds(n=512):
    torch.manual_seed(0)
    return torch.utils.data.TensorDataset(
        torch.randn(n, 3, 32, 32), torch.randint(0, 10, (n,)))


def _state(updater):
    return ClientState(0, torch.nn.utils.parameters_to_vector(
        updater.global_model.parameters()).detach())


def _bench(step_fn, batches=60, warmup=8):
    times = []
    loader = list(torch.utils.data.DataLoader(ds_global, batch_size=32))
    n = len(loader)
    for i in range(warmup + batches):
        x, y = loader[i % n]
        t0 = time.perf_counter()
        step_fn(x, y)
        times.append((time.perf_counter() - t0) * 1000.0)
    tail = times[warmup:]
    return sum(tail) / len(tail)


def main():
    global ds_global
    ds_global = _ds()
    device = "cpu"
    try:
        import torch_directml  # noqa: F401
        if torch_directml.is_available():
            device = "privateuseone:0"
    except Exception:
        pass

    updater = PyTorchLocalUpdater(device=device, in_channels=3, num_classes=10)
    cfg = ClientConfig(personalization_method="fedala", local_steps=1,
                       ala_max_epochs=3)

    state = _state(updater)

    def fedala_step(x, y):
        sub = torch.utils.data.TensorDataset(x, y)
        updater._update_fedala(state, cfg, sub,
                               [(x, y)], 1, cfg.local_lr, state.weights.clone())

    print(f"device={device}")
    ms = _bench(fedala_step)
    print(f"FedALA train-step: {ms:.2f} ms/batch (incl. ALA bookkeeping)")

    # CFL shares the standard FedAvg client path; measure it as reference.
    cfg_std = ClientConfig(personalization_method="none", local_steps=1)

    def std_step(x, y):
        updater._update_standard(state, cfg_std, [(x, y)], 1,
                                 cfg.local_lr, state.weights.clone())

    ms2 = _bench(std_step)
    print(f"Standard (CFL client path) train-step: {ms2:.2f} ms/batch")


if __name__ == "__main__":
    main()
