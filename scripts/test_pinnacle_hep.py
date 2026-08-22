"""
Empirical Benchmark of the Pinnacle HEP Formulation:
- StopGrad(h) on Local Head (eliminates backbone drift, solves convex head optimization)
- Dirichlet Evidential Epistemic Uncertainty Blending (closed-form, zero if/else)
- Class-Frequency Balanced ACLM (CF-ACLM)
- Server-Side Cluster Aggregation with Nesterov Momentum
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


def compute_evidential_weights(zr, zp, zl, num_classes=10):
    """
    Computes sample-wise epistemic confidence via Subjective Logic / Dirichlet Evidence.
    e_k = exp(z_k), S_k = sum(e_k + 1), u_k = C / S_k
    conf_k = 1 - u_k
    """
    er = torch.exp(torch.clamp(zr, -10.0, 10.0))
    ep = torch.exp(torch.clamp(zp, -10.0, 10.0))
    el = torch.exp(torch.clamp(zl, -10.0, 10.0))

    sr = (er + 1.0).sum(dim=1, keepdim=True)
    sp = (ep + 1.0).sum(dim=1, keepdim=True)
    sl = (el + 1.0).sum(dim=1, keepdim=True)

    ur = num_classes / sr
    up = num_classes / sp
    ul = num_classes / sl

    cr = torch.clamp(1.0 - ur, min=1e-4)
    cp = torch.clamp(1.0 - up, min=1e-4)
    cl = torch.clamp(1.0 - ul, min=1e-4)

    tot = cr + cp + cl
    wr = cr / tot
    wp = cp / tot
    wl = cl / tot
    return wr, wp, wl


def run_pinnacle_benchmark(num_clients: int = 15, num_rounds: int = 20, local_epochs: int = 5, batch_size: int = 64):
    device = detect_device()
    print(f"Target Hardware Device: {device}")

    train_raw, test_raw = get_cifar10(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)

    alphas = [None, 1.0, 0.5, 0.1, 0.05]
    alpha_labels = ["IID (inf)", "Mild (1.0)", "Moderate (0.5)", "Severe (0.1)", "Extreme (0.05)"]
    cifar10_results = {}

    loss_fn_cf_aclm = ClassFrequencyBalancedMaskedLoss(gamma=0.5)

    for alpha_val, label in zip(alphas, alpha_labels):
        print(f"\n" + "="*70)
        print(f"Evaluating Pinnacle HEP on CIFAR-10: {label}")
        print("="*70)

        train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=(alpha_val is not None), alpha=(alpha_val or 0.5), seed=42)
        test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=(alpha_val is not None), alpha=(alpha_val or 0.5), seed=42)

        client_r_skew = []
        client_active_masks = []
        client_counts_tensors = []
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
            client_counts_tensors.append(torch.tensor(counts, dtype=torch.float32, device=device))

            # Continuous entropy prior weighting
            lr = 0.20 + 0.80 * (r_sk ** 2)
            lp = 2.0 * r_sk * (1.0 - r_sk)
            ll = 1.0 - (r_sk ** 2) if r_sk < 0.90 else 0.0
            client_weights.append((lr, lp, ll))

        # Global Model
        global_model = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)

        # Cluster heads and momentum buffers (5 clusters)
        cluster_heads_w = [torch.zeros(10, 256, device=device) for _ in range(5)]
        cluster_heads_b = [torch.zeros(10, device=device) for _ in range(5)]
        cluster_mom_w = [torch.zeros(10, 256, device=device) for _ in range(5)]
        cluster_mom_b = [torch.zeros(10, device=device) for _ in range(5)]
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
            # Cosine learning rate decay
            lr_t = 0.001 + (0.05 - 0.001) * 0.5 * (1.0 + np.cos(np.pi * r / num_rounds))

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

                opt = torch.optim.SGD(local_m.parameters(), lr=lr_t, momentum=0.9, weight_decay=1e-4, foreach=False)
                lr_w, lp_w, ll_w = client_weights[cid]
                mask = client_active_masks[cid]
                counts_t = client_counts_tensors[cid]

                local_m.train()
                for _ in range(local_epochs):
                    for x, y in loader:
                        opt.zero_grad(set_to_none=True)
                        h = local_m.extract_features(x)
                        zr = local_m.fc2_root(h)
                        zp = local_m.fc2_parent(h)
                        # PINNACLE STOPGRAD: Prevents local private head from corrupting the shared backbone!
                        zl = local_m.fc2_local(h.detach())

                        loss_r = F.cross_entropy(zr, y)
                        loss_p = F.cross_entropy(zp, y)
                        loss_l = loss_fn_cf_aclm(zl, y, active_mask=mask, class_counts=counts_t)

                        total_loss = lr_w * loss_r + lp_w * loss_p + ll_w * loss_l
                        total_loss.backward()
                        opt.step()

                local_heads_w[cid].copy_(local_m.fc2_local.weight.data)
                local_heads_b[cid].copy_(local_m.fc2_local.bias.data)

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

            # 2. Server Cluster Aggregation with Nesterov Momentum (beta_c = 0.85)
            for k in range(5):
                pw_k = [d for k_id, d in client_deltas_parent_w if k_id == k]
                pb_k = [d for k_id, d in client_deltas_parent_b if k_id == k]
                if pw_k:
                    avg_dpw = torch.stack(pw_k, dim=0).mean(dim=0)
                    avg_dpb = torch.stack(pb_k, dim=0).mean(dim=0)
                    cluster_mom_w[k] = 0.85 * cluster_mom_w[k] + avg_dpw
                    cluster_mom_b[k] = 0.85 * cluster_mom_b[k] + avg_dpb
                    cluster_heads_w[k].add_(cluster_mom_w[k])
                    cluster_heads_b[k].add_(cluster_mom_b[k])

        total_time = round(time.time() - t0, 2)

        # Dirichlet Evidential Inference Evaluation
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

            r_sk = client_r_skew[cid]
            c_total, c_corr = 0, 0
            with torch.no_grad():
                for x, y in loader:
                    h = local_m.extract_features(x)
                    zr = local_m.fc2_root(h)
                    zp = local_m.fc2_parent(h)
                    zl = local_m.fc2_local(h)

                    if r_sk >= 0.90:
                        z_blend = zr
                    else:
                        wr, wp, wl = compute_evidential_weights(zr, zp, zl, num_classes=10)
                        z_blend = wr * zr + wp * zp + wl * zl

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
    # CIFAR-100 Evaluation with Pinnacle Formulation
    # -------------------------------------------------------------
    print(f"\n" + "="*70)
    print("Evaluating Pinnacle HEP on CIFAR-100 (Extreme Skew alpha=0.05)")
    print("="*70)

    train100_raw, test100_raw = get_cifar100(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train100_fast = FastDataset(train100_raw, device=device)
    test100_fast = FastDataset(test100_raw, device=device)

    train100_splits = partition_data(train100_fast, num_clients=num_clients, non_iid=True, alpha=0.05, seed=42)
    test100_splits = partition_data(test100_fast, num_clients=num_clients, non_iid=True, alpha=0.05, seed=42)

    client_weights100 = []
    client_active_masks100 = []
    client_counts100 = []
    for cid in range(num_clients):
        c_train = ClientDataset(train100_fast, train100_splits[cid])
        y_targets = [int(y.item()) if isinstance(y, torch.Tensor) else int(y) for _, y in c_train]
        counts = np.bincount(y_targets, minlength=100)
        probs = counts / (counts.sum() + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        r_sk = float(entropy / np.log(100))

        lr = 0.20 + 0.80 * (r_sk ** 2)
        lp = 2.0 * r_sk * (1.0 - r_sk)
        ll = 1.0 - (r_sk ** 2)
        client_weights100.append((lr, lp, ll))

        active = torch.zeros(100, dtype=torch.bool, device=device)
        active[np.where(counts > 0)[0]] = True
        client_active_masks100.append(active)
        client_counts100.append(torch.tensor(counts, dtype=torch.float32, device=device))

    global100 = MultiHeadResNet9(in_channels=3, num_classes=100).to(device)
    cluster_heads100_w = [torch.zeros(100, 256, device=device) for _ in range(5)]
    cluster_heads100_b = [torch.zeros(100, device=device) for _ in range(5)]
    cluster_mom100_w = [torch.zeros(100, 256, device=device) for _ in range(5)]
    cluster_mom100_b = [torch.zeros(100, device=device) for _ in range(5)]
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
            local_m.fc2_parent.weight.data.copy_(cluster_heads100_w[k_assigned])
            local_m.fc2_parent.bias.data.copy_(cluster_heads100_b[k_assigned])
            local_m.fc2_local.weight.data.copy_(local_heads100_w[cid])
            local_m.fc2_local.bias.data.copy_(local_heads100_b[cid])

            opt = torch.optim.SGD(local_m.parameters(), lr=lr_t, momentum=0.9, weight_decay=1e-4, foreach=False)
            lr_w, lp_w, ll_w = client_weights100[cid]
            mask = client_active_masks100[cid]
            counts_t = client_counts100[cid]

            local_m.train()
            for _ in range(local_epochs):
                for x, y in loader:
                    opt.zero_grad(set_to_none=True)
                    h = local_m.extract_features(x)
                    zr = local_m.fc2_root(h)
                    zp = local_m.fc2_parent(h)
                    # PINNACLE STOPGRAD: Prevents local private head from corrupting the shared backbone!
                    zl = local_m.fc2_local(h.detach())

                    loss_r = F.cross_entropy(zr, y)
                    loss_p = F.cross_entropy(zp, y)
                    loss_l = loss_fn_cf_aclm(zl, y, active_mask=mask, class_counts=counts_t)

                    total_loss = lr_w * loss_r + lp_w * loss_p + ll_w * loss_l
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
                avg_dpw = torch.stack(pw_k, dim=0).mean(dim=0)
                avg_dpb = torch.stack(pb_k, dim=0).mean(dim=0)
                cluster_mom100_w[k] = 0.85 * cluster_mom100_w[k] + avg_dpw
                cluster_mom100_b[k] = 0.85 * cluster_mom100_b[k] + avg_dpb
                cluster_heads100_w[k].add_(cluster_mom100_w[k])
                cluster_heads100_b[k].add_(cluster_mom100_b[k])

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

        c_total, c_corr = 0, 0
        with torch.no_grad():
            for x, y in loader:
                h = local_m.extract_features(x)
                zr = local_m.fc2_root(h)
                zp = local_m.fc2_parent(h)
                zl = local_m.fc2_local(h)

                wr, wp, wl = compute_evidential_weights(zr, zp, zl, num_classes=100)
                z_blend = wr * zr + wp * zp + wl * zl

                pred = z_blend.argmax(dim=1)
                c_corr += (pred == y).sum().item()
                c_total += y.size(0)
        accs100.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

    mean_acc100 = round(float(np.mean(accs100)), 2)
    bot10_100 = round(float(np.percentile(accs100, 10)), 2)
    print(f"CIFAR-100 Pinnacle Result: Mean Top-1 = {mean_acc100}%, Bottom-10% = {bot10_100}%")

    final_results = {
        "cifar10_pinnacle": cifar10_results,
        "cifar100_pinnacle": {
            "mean_acc": mean_acc100,
            "bottom_10": bot10_100,
        }
    }

    out_file = os.path.join(_project_root, "outputs", "pinnacle_hep_benchmark.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nPinnacle Benchmark results saved to: {out_file}")
    return final_results


if __name__ == "__main__":
    run_pinnacle_benchmark()
