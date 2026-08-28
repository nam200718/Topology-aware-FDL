"""
Breakthrough Benchmark: Post-Convergence Local Head Calibration (P-LHC) + S-AOC
Tests whether 1-epoch convex linear head alignment pushes:
- CIFAR-100 Extreme Skew from 42.45% -> 55%+ (matching/beating FedBABU & Ditto)
- CIFAR-10 Severe Skew from 82.71% -> 85%+
- Byzantine Sign-Flipping from 28.30% -> 35%+
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.model import MultiHeadResNet9
from src.core.loss import ClassFrequencyBalancedMaskedLoss
from src.data.dataset import get_cifar10, get_cifar100, partition_data, ClientDataset, get_fast_dataloader, FastDataset
from src.experiments.builder import detect_device


def run_breakthrough_experiment():
    device = detect_device()
    print(f"Device: {device}")
    
    # -------------------------------------------------------------
    # 1. CIFAR-100 Extreme Skew (alpha=0.05) with 1-Epoch Linear Calibration
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("Testing CIFAR-100 Extreme Skew (alpha=0.05) with Linear Calibration (P-LHC)...")
    print("="*70)

    train100_raw, test100_raw = get_cifar100(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train100_fast = FastDataset(train100_raw, device=device)
    test100_fast = FastDataset(test100_raw, device=device)

    num_clients = 15
    train100_splits = partition_data(train100_fast, num_clients=num_clients, non_iid=True, alpha=0.05, seed=42)
    test100_splits = partition_data(test100_fast, num_clients=num_clients, non_iid=True, alpha=0.05, seed=42)

    global100 = MultiHeadResNet9(in_channels=3, num_classes=100).to(device)
    cluster_heads = [global100.fc2_root.weight.data.clone() for _ in range(5)]
    cluster_heads_b = [global100.fc2_root.bias.data.clone() for _ in range(5)]

    num_rounds = 25
    batch_size = 32
    loss_fn = ClassFrequencyBalancedMaskedLoss(gamma=0.5)

    client_active_masks = []
    client_counts_tensors = []
    for cid in range(num_clients):
        c_train = ClientDataset(train100_fast, train100_splits[cid])
        y_targets = [int(y.item()) if isinstance(y, torch.Tensor) else int(y) for _, y in c_train]
        counts = np.bincount(y_targets, minlength=100)
        active = torch.zeros(100, dtype=torch.bool, device=device)
        active[np.where(counts > 0)[0]] = True
        client_active_masks.append(active)
        client_counts_tensors.append(torch.tensor(counts, dtype=torch.float32, device=device))

    t0 = time.time()
    for r in range(num_rounds):
        lr_t = 0.001 + (0.05 - 0.001) * 0.5 * (1.0 + np.cos(np.pi * r / num_rounds))

        client_deltas_bb = []
        client_deltas_rw = []
        client_deltas_rb = []
        client_deltas_pw = []
        client_deltas_pb = []

        for cid in range(num_clients):
            c_train = ClientDataset(train100_fast, train100_splits[cid])
            loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)

            local_m = MultiHeadResNet9(in_channels=3, num_classes=100).to(device)
            local_m.load_state_dict(global100.state_dict())

            k_assigned = cid % 5
            local_m.fc2_parent.weight.data.copy_(cluster_heads[k_assigned])
            local_m.fc2_parent.bias.data.copy_(cluster_heads_b[k_assigned])

            opt = torch.optim.SGD(local_m.parameters(), lr=lr_t, momentum=0.9, weight_decay=1e-4, foreach=False)
            mask = client_active_masks[cid]
            counts_t = client_counts_tensors[cid]

            local_m.train()
            for _ in range(3):
                for x, y in loader:
                    opt.zero_grad(set_to_none=True)
                    zr, zp, zl = local_m(x, head="all")
                    loss_r = F.cross_entropy(zr, y)
                    loss_p = F.cross_entropy(zp, y)
                    zl_masked = zl.masked_fill(~mask.unsqueeze(0), -1e9)
                    loss_l = loss_fn(zl_masked, y, active_mask=mask, class_counts=counts_t)
                    total_loss = 0.5 * loss_r + 0.3 * loss_p + 0.2 * loss_l
                    total_loss.backward()
                    opt.step()

            delta_bb = {k: (local_m.state_dict()[k] - global100.state_dict()[k]).float() for k in global100.state_dict() if 'fc2' not in k}
            d_rw = (local_m.fc2_root.weight.data - global100.fc2_root.weight.data).float()
            d_rb = (local_m.fc2_root.bias.data - global100.fc2_root.bias.data).float()
            d_pw = (local_m.fc2_parent.weight.data - cluster_heads[k_assigned]).float()
            d_pb = (local_m.fc2_parent.bias.data - cluster_heads_b[k_assigned]).float()

            client_deltas_bb.append(delta_bb)
            client_deltas_rw.append(d_rw)
            client_deltas_rb.append(d_rb)
            client_deltas_pw.append((k_assigned, d_pw))
            client_deltas_pb.append((k_assigned, d_pb))

        # Server Global Aggregation
        avg_bb = {}
        for k in client_deltas_bb[0].keys():
            avg_bb[k] = global100.state_dict()[k] + torch.stack([client_deltas_bb[i][k] for i in range(num_clients)]).mean(dim=0)
        avg_rw = global100.fc2_root.weight.data + torch.stack(client_deltas_rw).mean(dim=0)
        avg_rb = global100.fc2_root.bias.data + torch.stack(client_deltas_rb).mean(dim=0)

        global100.load_state_dict(avg_bb, strict=False)
        global100.fc2_root.weight.data.copy_(avg_rw)
        global100.fc2_root.bias.data.copy_(avg_rb)

        # Cluster heads
        for k in range(5):
            pw_k = [d for k_id, d in client_deltas_pw if k_id == k]
            pb_k = [d for k_id, d in client_deltas_pb if k_id == k]
            if pw_k:
                cluster_heads[k].add_(torch.stack(pw_k).mean(dim=0))
                cluster_heads_b[k].add_(torch.stack(pb_k).mean(dim=0))

    elapsed = round(time.time() - t0, 2)

    # Post-Convergence Local Head Calibration (P-LHC):
    # Each client does 3 quick epochs on ONLY fc2_local (body completely frozen!)
    print(f"Federated training completed in {elapsed}s. Running Post-Convergence Local Head Calibration...")
    calibrated_accs = []
    for cid in range(num_clients):
        c_train = ClientDataset(train100_fast, train100_splits[cid])
        train_loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)

        local_m = MultiHeadResNet9(in_channels=3, num_classes=100).to(device)
        local_m.load_state_dict(global100.state_dict())

        # Freeze entire backbone
        for param in local_m.parameters():
            param.requires_grad = False
        for param in local_m.fc2_local.parameters():
            param.requires_grad = True

        head_opt = torch.optim.Adam(local_m.fc2_local.parameters(), lr=0.01)
        mask = client_active_masks[cid]
        counts_t = client_counts_tensors[cid]

        local_m.train()
        for _ in range(5):
            for x, y in train_loader:
                head_opt.zero_grad()
                with torch.no_grad():
                    h = local_m.extract_features(x)
                zl = local_m.fc2_local(h)
                zl_masked = zl.masked_fill(~mask.unsqueeze(0), -1e9)
                loss = loss_fn(zl_masked, y, active_mask=mask, class_counts=counts_t)
                loss.backward()
                head_opt.step()

        # Evaluate on client test dataset
        c_test = ClientDataset(test100_fast, test100_splits[cid])
        test_loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)

        local_m.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                h = local_m.extract_features(x)
                zl = local_m.fc2_local(h)
                zl_masked = zl.masked_fill(~mask.unsqueeze(0), -1e9)
                pred = zl_masked.argmax(dim=1)
                corr += (pred == y).sum().item()
                tot += y.size(0)
        c_acc = (corr / tot * 100.0) if tot > 0 else 0.0
        calibrated_accs.append(c_acc)

    mean_acc = round(float(np.mean(calibrated_accs)), 2)
    bot10 = round(float(np.percentile(calibrated_accs, 10)), 2)
    print(f"\n>>> [CIFAR-100 Extreme Skew w/ P-LHC Result]: Mean Top-1 = {mean_acc}%, Bottom-10% Fairness = {bot10}%")


if __name__ == "__main__":
    run_breakthrough_experiment()
