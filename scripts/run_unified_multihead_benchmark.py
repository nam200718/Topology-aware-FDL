"""
Benchmark Script for the Unified Single-Pass Multi-Head HEP with ACLM.
Uses:
- Isolated parameter heads (fc2_root, fc2_parent, fc2_local)
- Unified Single-Pass Multi-Task Loss: L_total = lambda_r * L_r + lambda_p * L_p + lambda_l * L_l (ACLM)
- Continuous entropy weighting: lambda_r = 0.2 + 0.8 * R_skew, lambda_p = 2 * R_skew * (1 - R_skew), lambda_l = 1.0 - R_skew
- Single optimizer step per batch
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
from src.core.loss import ActiveMaskedCrossEntropyLoss
from src.data.dataset import get_cifar10, get_cifar100, partition_data, ClientDataset, get_fast_dataloader, FastDataset
from src.experiments.builder import detect_device


def compute_loss_weights(r_skew: float):
    # Continuous differentiable weights:
    # IID (R_skew=1.0) -> lambda_r=1.0, lambda_p=0.0, lambda_l=0.0 (Pure FedAvg)
    # Extreme (R_skew=0.0) -> lambda_r=0.20 (anchor), lambda_p=0.0, lambda_l=1.0 (Specialized)
    # Moderate (R_skew=0.5) -> lambda_r=0.60, lambda_p=0.50, lambda_l=0.50
    lr = 0.20 + 0.80 * r_skew
    lp = 2.0 * r_skew * (1.0 - r_skew)
    ll = 1.0 - r_skew
    total = lr + lp + ll
    alpha_r = lr / total
    alpha_p = lp / total
    alpha_l = ll / total
    return lr, lp, ll, alpha_r, alpha_p, alpha_l


def run_unified_multihead_benchmark(num_clients: int = 15, num_rounds: int = 20, local_epochs: int = 5, batch_size: int = 64):
    device = detect_device()
    print(f"Target Hardware Device: {device}")

    train_raw, test_raw = get_cifar10(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)

    alphas = [None, 1.0, 0.5, 0.1, 0.05]
    alpha_labels = ["IID (inf)", "Mild (1.0)", "Moderate (0.5)", "Severe (0.1)", "Extreme (0.05)"]
    cifar10_results = {}

    loss_fn_aclm = ActiveMaskedCrossEntropyLoss()

    for alpha_val, label in zip(alphas, alpha_labels):
        print(f"\n" + "="*70)
        print(f"Evaluating Unified Single-Pass Multi-Head HEP on CIFAR-10: {label}")
        print("="*70)

        train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=(alpha_val is not None), alpha=(alpha_val or 0.5), seed=42)
        test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=(alpha_val is not None), alpha=(alpha_val or 0.5), seed=42)

        client_r_skew = []
        client_active_masks = []
        client_weights = []
        for cid in range(num_clients):
            c_train = ClientDataset(train_fast, train_splits[cid])
            y_targets = [int(y.item()) if isinstance(y, torch.Tensor) else int(y) for _, y in c_train]
            counts = np.bincount(y_targets, minlength=10)
            probs = counts / (counts.sum() + 1e-8)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            r_sk = float(entropy / np.log(10)) if alpha_val is not None else 1.0
            client_r_skew.append(r_sk)

            active = torch.zeros(10, dtype=torch.bool, device=device)
            active[np.where(counts > 0)[0]] = True
            client_active_masks.append(active)

            lr, lp, ll, ar, ap, al = compute_loss_weights(r_sk)
            client_weights.append((lr, lp, ll, ar, ap, al))

        # Global Model
        global_model = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)

        # Cluster heads (5 clusters)
        cluster_heads_w = [torch.zeros(10, 256, device=device) for _ in range(5)]
        cluster_heads_b = [torch.zeros(10, device=device) for _ in range(5)]
        for k in range(5):
            cluster_heads_w[k].copy_(global_model.fc2_root.weight.data)
            cluster_heads_b[k].copy_(global_model.fc2_root.bias.data)

        # Local heads (15 clients)
        local_heads_w = [torch.zeros(10, 256, device=device) for _ in range(num_clients)]
        local_heads_b = [torch.zeros(10, device=device) for _ in range(num_clients)]
        for cid in range(num_clients):
            local_heads_w[cid].copy_(global_model.fc2_root.weight.data)
            local_heads_b[cid].copy_(global_model.fc2_root.bias.data)

        cluster_assignments = [i % 5 for i in range(num_clients)]

        t0 = time.time()
        for r in range(num_rounds):
            client_deltas_backbone = []
            client_deltas_root_w = []
            client_deltas_root_b = []
            client_deltas_parent_w = []
            client_deltas_parent_b = []

            for cid in range(num_clients):
                c_train = ClientDataset(train_fast, train_splits[cid])
                loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)

                local_m = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
                local_m.load_state_dict(global_model.state_dict())

                k_assigned = cluster_assignments[cid]
                local_m.fc2_parent.weight.data.copy_(cluster_heads_w[k_assigned])
                local_m.fc2_parent.bias.data.copy_(cluster_heads_b[k_assigned])
                local_m.fc2_local.weight.data.copy_(local_heads_w[cid])
                local_m.fc2_local.bias.data.copy_(local_heads_b[cid])

                opt = torch.optim.SGD(local_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
                lr, lp, ll, _, _, _ = client_weights[cid]
                mask = client_active_masks[cid]

                local_m.train()
                for _ in range(local_epochs):
                    for x, y in loader:
                        opt.zero_grad(set_to_none=True)
                        # Unified Single Forward Pass for all 3 heads
                        zr, zp, zl = local_m(x, head="all")

                        loss_r = F.cross_entropy(zr, y)
                        loss_p = F.cross_entropy(zp, y)
                        loss_l = loss_fn_aclm(zl, y, active_mask=mask)

                        total_loss = lr * loss_r + lp * loss_p + ll * loss_l
                        total_loss.backward()
                        opt.step()

                # Save updated local head
                local_heads_w[cid].copy_(local_m.fc2_local.weight.data)
                local_heads_b[cid].copy_(local_m.fc2_local.bias.data)

                # Compute deltas for backbone, root head, and parent head
                delta_bb = {k: local_m.state_dict()[k] - global_model.state_dict()[k] for k in global_model.state_dict() if 'fc2' not in k}
                d_rw = local_m.fc2_root.weight.data - global_model.fc2_root.weight.data
                d_rb = local_m.fc2_root.bias.data - global_model.fc2_root.bias.data
                d_pw = local_m.fc2_parent.weight.data - cluster_heads_w[k_assigned]
                d_pb = local_m.fc2_parent.bias.data - cluster_heads_b[k_assigned]

                client_deltas_backbone.append(delta_bb)
                client_deltas_root_w.append(d_rw)
                client_deltas_root_b.append(d_rb)
                client_deltas_parent_w.append((k_assigned, d_pw))
                client_deltas_parent_b.append((k_assigned, d_pb))

            # 1. Global Server Aggregation: Backbone & Root Head
            avg_bb = {}
            for k in client_deltas_backbone[0].keys():
                avg_bb[k] = global_model.state_dict()[k] + torch.stack([client_deltas_backbone[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
            avg_rw = global_model.fc2_root.weight.data + torch.stack(client_deltas_root_w, dim=0).mean(dim=0)
            avg_rb = global_model.fc2_root.bias.data + torch.stack(client_deltas_root_b, dim=0).mean(dim=0)

            global_model.load_state_dict(avg_bb, strict=False)
            global_model.fc2_root.weight.data.copy_(avg_rw)
            global_model.fc2_root.bias.data.copy_(avg_rb)

            # 2. Cluster Aggregation: Parent Heads
            for k in range(5):
                pw_k = [d for k_id, d in client_deltas_parent_w if k_id == k]
                pb_k = [d for k_id, d in client_deltas_parent_b if k_id == k]
                if pw_k:
                    cluster_heads_w[k].add_(torch.stack(pw_k, dim=0).mean(dim=0))
                    cluster_heads_b[k].add_(torch.stack(pb_k, dim=0).mean(dim=0))

        total_time = round(time.time() - t0, 2)

        # Evaluation
        accs = []
        global_model.eval()
        for cid in range(num_clients):
            c_test = ClientDataset(test_fast, test_splits[cid])
            loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
            local_m = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
            local_m.load_state_dict(global_model.state_dict())
            k_assigned = cluster_assignments[cid]
            local_m.fc2_parent.weight.data.copy_(cluster_heads_w[k_assigned])
            local_m.fc2_parent.bias.data.copy_(cluster_heads_b[k_assigned])
            local_m.fc2_local.weight.data.copy_(local_heads_w[cid])
            local_m.fc2_local.bias.data.copy_(local_heads_b[cid])

            _, _, _, ar, ap, al = client_weights[cid]

            c_total, c_corr = 0, 0
            with torch.no_grad():
                for x, y in loader:
                    zr, zp, zl = local_m(x, head="all")
                    # Gated inference blending
                    z_blend = ar * zr + ap * zp + al * zl
                    pred = z_blend.argmax(dim=1)
                    c_corr += (pred == y).sum().item()
                    c_total += y.size(0)
            accs.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

        mean_acc = round(float(np.mean(accs)), 2)
        bot10 = round(float(np.percentile(accs, 10)), 2)
        cifar10_results[label] = {
            "mean_acc": mean_acc,
            "bottom_10": bot10,
            "total_time_s": total_time,
        }
        print(f"Result [{label}]: Mean Top-1 = {mean_acc}%, Bottom-10% = {bot10}%, Time = {total_time}s")

    # -------------------------------------------------------------
    # CIFAR-100 Evaluation with ACLM
    # -------------------------------------------------------------
    print(f"\n" + "="*70)
    print("Evaluating Unified Single-Pass Multi-Head HEP on CIFAR-100 (Extreme Skew alpha=0.05)")
    print("="*70)

    train100_raw, test100_raw = get_cifar100(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train100_fast = FastDataset(train100_raw, device=device)
    test100_fast = FastDataset(test100_raw, device=device)

    train100_splits = partition_data(train100_fast, num_clients=num_clients, non_iid=True, alpha=0.05, seed=42)
    test100_splits = partition_data(test100_fast, num_clients=num_clients, non_iid=True, alpha=0.05, seed=42)

    client_weights100 = []
    client_active_masks100 = []
    for cid in range(num_clients):
        c_train = ClientDataset(train100_fast, train100_splits[cid])
        y_targets = [int(y.item()) if isinstance(y, torch.Tensor) else int(y) for _, y in c_train]
        counts = np.bincount(y_targets, minlength=100)
        probs = counts / (counts.sum() + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        r_sk = float(entropy / np.log(100))
        client_weights100.append(compute_loss_weights(r_sk))

        active = torch.zeros(100, dtype=torch.bool, device=device)
        active[np.where(counts > 0)[0]] = True
        client_active_masks100.append(active)

    global100 = MultiHeadResNet9(in_channels=3, num_classes=100).to(device)
    cluster_heads100_w = [torch.zeros(100, 256, device=device) for _ in range(5)]
    cluster_heads100_b = [torch.zeros(100, device=device) for _ in range(5)]
    local_heads100_w = [torch.zeros(100, 256, device=device) for _ in range(num_clients)]
    local_heads100_b = [torch.zeros(100, device=device) for _ in range(num_clients)]

    for k in range(5):
        cluster_heads100_w[k].copy_(global100.fc2_root.weight.data)
        cluster_heads100_b[k].copy_(global100.fc2_root.bias.data)
    for cid in range(num_clients):
        local_heads100_w[cid].copy_(global100.fc2_root.weight.data)
        local_heads100_b[cid].copy_(global100.fc2_root.bias.data)

    t0 = time.time()
    for r in range(num_rounds):
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
            local_m.fc2_parent.weight.data.copy_(cluster_heads100_w[k_assigned])
            local_m.fc2_parent.bias.data.copy_(cluster_heads100_b[k_assigned])
            local_m.fc2_local.weight.data.copy_(local_heads100_w[cid])
            local_m.fc2_local.bias.data.copy_(local_heads100_b[cid])

            opt = torch.optim.SGD(local_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
            lr, lp, ll, _, _, _ = client_weights100[cid]
            mask = client_active_masks100[cid]

            local_m.train()
            for _ in range(local_epochs):
                for x, y in loader:
                    opt.zero_grad(set_to_none=True)
                    zr, zp, zl = local_m(x, head="all")

                    loss_r = F.cross_entropy(zr, y)
                    loss_p = F.cross_entropy(zp, y)
                    loss_l = loss_fn_aclm(zl, y, active_mask=mask)

                    total_loss = lr * loss_r + lp * loss_p + ll * loss_l
                    total_loss.backward()
                    opt.step()

            local_heads100_w[cid].copy_(local_m.fc2_local.weight.data)
            local_heads100_b[cid].copy_(local_m.fc2_local.bias.data)

            delta_bb = {k: local_m.state_dict()[k] - global100.state_dict()[k] for k in global100.state_dict() if 'fc2' not in k}
            d_rw = local_m.fc2_root.weight.data - global100.fc2_root.weight.data
            d_rb = local_m.fc2_root.bias.data - global100.fc2_root.bias.data
            d_pw = local_m.fc2_parent.weight.data - cluster_heads100_w[k_assigned]
            d_pb = local_m.fc2_parent.bias.data - cluster_heads100_b[k_assigned]

            client_deltas_bb.append(delta_bb)
            client_deltas_rw.append(d_rw)
            client_deltas_rb.append(d_rb)
            client_deltas_pw.append((k_assigned, d_pw))
            client_deltas_pb.append((k_assigned, d_pb))

        avg_bb = {}
        for k in client_deltas_bb[0].keys():
            avg_bb[k] = global100.state_dict()[k] + torch.stack([client_deltas_bb[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
        avg_rw = global100.fc2_root.weight.data + torch.stack(client_deltas_rw, dim=0).mean(dim=0)
        avg_rb = global100.fc2_root.bias.data + torch.stack(client_deltas_rb, dim=0).mean(dim=0)

        global100.load_state_dict(avg_bb, strict=False)
        global100.fc2_root.weight.data.copy_(avg_rw)
        global100.fc2_root.bias.data.copy_(avg_rb)

        for k in range(5):
            pw_k = [d for k_id, d in client_deltas_pw if k_id == k]
            pb_k = [d for k_id, d in client_deltas_pb if k_id == k]
            if pw_k:
                cluster_heads100_w[k].add_(torch.stack(pw_k, dim=0).mean(dim=0))
                cluster_heads100_b[k].add_(torch.stack(pb_k, dim=0).mean(dim=0))

    accs100 = []
    global100.eval()
    for cid in range(num_clients):
        c_test = ClientDataset(test100_fast, test100_splits[cid])
        loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
        local_m = MultiHeadResNet9(in_channels=3, num_classes=100).to(device)
        local_m.load_state_dict(global100.state_dict())
        k_assigned = cid % 5
        local_m.fc2_parent.weight.data.copy_(cluster_heads100_w[k_assigned])
        local_m.fc2_parent.bias.data.copy_(cluster_heads100_b[k_assigned])
        local_m.fc2_local.weight.data.copy_(local_heads100_w[cid])
        local_m.fc2_local.bias.data.copy_(local_heads100_b[cid])

        _, _, _, ar, ap, al = client_weights100[cid]

        c_total, c_corr = 0, 0
        with torch.no_grad():
            for x, y in loader:
                zr, zp, zl = local_m(x, head="all")
                z_blend = ar * zr + ap * zp + al * zl
                pred = z_blend.argmax(dim=1)
                c_corr += (pred == y).sum().item()
                c_total += y.size(0)
        accs100.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

    mean_acc100 = round(float(np.mean(accs100)), 2)
    bot10_100 = round(float(np.percentile(accs100, 10)), 2)
    print(f"CIFAR-100 ACLM Multi-Head Result: Mean Top-1 = {mean_acc100}%, Bottom-10% = {bot10_100}%")

    final_results = {
        "cifar10_unified_multihead": cifar10_results,
        "cifar100_aclm_multihead": {
            "mean_acc": mean_acc100,
            "bottom_10": bot10_100,
        }
    }

    out_file = os.path.join(_project_root, "outputs", "unified_multihead_benchmark.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nUnified Multi-Head Benchmark results saved to: {out_file}")
    return final_results


if __name__ == "__main__":
    run_unified_multihead_benchmark()
