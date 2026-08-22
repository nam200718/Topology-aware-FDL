"""
Verification of P-LHC across all 5 Dirichlet regimes on CIFAR-10.
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
from src.data.dataset import get_cifar10, partition_data, ClientDataset, get_fast_dataloader, FastDataset
from src.experiments.builder import detect_device


def evaluate_cifar10_p_lhc(alpha=0.05, num_rounds=20, num_clients=15, batch_size=32, device="cpu"):
    train_raw, test_raw = get_cifar10(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)

    is_non_iid = alpha is not None
    train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=is_non_iid, alpha=(alpha or 0.5), seed=42)
    test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=is_non_iid, alpha=(alpha or 0.5), seed=42)

    client_r_skew = []
    client_active_masks = []
    client_counts_tensors = []

    for cid in range(num_clients):
        c_train = ClientDataset(train_fast, train_splits[cid])
        y_targets = [int(y.item()) if isinstance(y, torch.Tensor) else int(y) for _, y in c_train]
        counts = np.bincount(y_targets, minlength=10)
        probs = counts / (counts.sum() + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        r_sk = float(entropy / np.log(10)) if is_non_iid else 1.0
        client_r_skew.append(r_sk)

        active = torch.zeros(10, dtype=torch.bool, device=device)
        active[np.where(counts > 0)[0]] = True
        client_active_masks.append(active)
        client_counts_tensors.append(torch.tensor(counts, dtype=torch.float32, device=device))

    global_model = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
    cluster_heads = [global_model.fc2_root.weight.data.clone() for _ in range(5)]
    cluster_heads_b = [global_model.fc2_root.bias.data.clone() for _ in range(5)]

    loss_fn = ClassFrequencyBalancedMaskedLoss(gamma=0.5)

    for r in range(num_rounds):
        lr_t = 0.001 + (0.05 - 0.001) * 0.5 * (1.0 + np.cos(np.pi * r / num_rounds))

        client_deltas_bb = []
        client_deltas_rw = []
        client_deltas_rb = []
        client_deltas_pw = []
        client_deltas_pb = []

        for cid in range(num_clients):
            c_train = ClientDataset(train_fast, train_splits[cid])
            loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)

            local_m = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
            local_m.load_state_dict(global_model.state_dict())

            k_assigned = cid % 5
            local_m.fc2_parent.weight.data.copy_(cluster_heads[k_assigned])
            local_m.fc2_parent.bias.data.copy_(cluster_heads_b[k_assigned])

            opt = torch.optim.SGD(local_m.parameters(), lr=lr_t, momentum=0.9, weight_decay=1e-4, foreach=False)
            mask = client_active_masks[cid]
            counts_t = client_counts_tensors[cid]
            r_sk = client_r_skew[cid]

            local_m.train()
            for _ in range(3):
                for x, y in loader:
                    opt.zero_grad(set_to_none=True)
                    zr, zp, zl = local_m(x, head="all")
                    loss_r = F.cross_entropy(zr, y)
                    loss_p = F.cross_entropy(zp, y) if r_sk < 0.85 else 0.0
                    zl_masked = zl.masked_fill(~mask.unsqueeze(0), -1e9)
                    loss_l = loss_fn(zl_masked, y, active_mask=mask, class_counts=counts_t) if r_sk < 0.85 else 0.0

                    total_loss = 0.5 * loss_r + 0.3 * loss_p + 0.2 * loss_l
                    total_loss.backward()
                    opt.step()

            delta_bb = {k: (local_m.state_dict()[k] - global_model.state_dict()[k]).float() for k in global_model.state_dict() if 'fc2' not in k}
            d_rw = (local_m.fc2_root.weight.data - global_model.fc2_root.weight.data).float()
            d_rb = (local_m.fc2_root.bias.data - global_model.fc2_root.bias.data).float()
            d_pw = (local_m.fc2_parent.weight.data - cluster_heads[k_assigned]).float()
            d_pb = (local_m.fc2_parent.bias.data - cluster_heads_b[k_assigned]).float()

            client_deltas_bb.append(delta_bb)
            client_deltas_rw.append(d_rw)
            client_deltas_rb.append(d_rb)
            client_deltas_pw.append((k_assigned, d_pw))
            client_deltas_pb.append((k_assigned, d_pb))

        avg_bb = {}
        for k in client_deltas_bb[0].keys():
            avg_bb[k] = global_model.state_dict()[k] + torch.stack([client_deltas_bb[i][k] for i in range(num_clients)]).mean(dim=0)
        avg_rw = global_model.fc2_root.weight.data + torch.stack(client_deltas_rw).mean(dim=0)
        avg_rb = global_model.fc2_root.bias.data + torch.stack(client_deltas_rb).mean(dim=0)

        global_model.load_state_dict(avg_bb, strict=False)
        global_model.fc2_root.weight.data.copy_(avg_rw)
        global_model.fc2_root.bias.data.copy_(avg_rb)

        for k in range(5):
            pw_k = [d for k_id, d in client_deltas_pw if k_id == k]
            pb_k = [d for k_id, d in client_deltas_pb if k_id == k]
            if pw_k:
                cluster_heads[k].add_(torch.stack(pw_k).mean(dim=0))
                cluster_heads_b[k].add_(torch.stack(pb_k).mean(dim=0))

    # P-LHC Linear Calibration
    calibrated_accs = []
    for cid in range(num_clients):
        c_train = ClientDataset(train_fast, train_splits[cid])
        train_loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)

        local_m = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
        local_m.load_state_dict(global_model.state_dict())

        r_sk = client_r_skew[cid]
        if r_sk < 0.85:
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

        c_test = ClientDataset(test_fast, test_splits[cid])
        test_loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)

        local_m.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                if r_sk >= 0.85:
                    pred = local_m(x, head="root").argmax(dim=1)
                else:
                    h = local_m.extract_features(x)
                    zl = local_m.fc2_local(h)
                    mask = client_active_masks[cid]
                    zl_masked = zl.masked_fill(~mask.unsqueeze(0), -1e9)
                    pred = zl_masked.argmax(dim=1)
                corr += (pred == y).sum().item()
                tot += y.size(0)
        c_acc = (corr / tot * 100.0) if tot > 0 else 0.0
        calibrated_accs.append(c_acc)

    mean_acc = round(float(np.mean(calibrated_accs)), 2)
    bot10 = round(float(np.percentile(calibrated_accs, 10)), 2)
    return mean_acc, bot10


if __name__ == "__main__":
    device = detect_device()
    scenarios = [
        ("IID (inf)", None),
        ("Mild (1.0)", 1.0),
        ("Moderate (0.5)", 0.5),
        ("Severe (0.1)", 0.1),
        ("Extreme (0.05)", 0.05),
    ]

    print("="*70)
    print("CIFAR-10 FULL SPECTRUM BENCHMARK WITH P-LHC")
    print("="*70)

    for label, alpha_val in scenarios:
        m, b = evaluate_cifar10_p_lhc(alpha=alpha_val, num_rounds=20, num_clients=15, batch_size=32, device=device)
        print(f"Scenario [{label:15s}]: Top-1 Mean = {m:5.2f}% | Bottom-10% Fairness = {b:5.2f}%")
