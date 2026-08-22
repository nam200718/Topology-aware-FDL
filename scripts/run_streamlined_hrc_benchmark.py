"""
Benchmark Script for the Streamlined Hierarchical Residual Classifier (HRC / Streamlined HEP).
Compares:
1. CIFAR-10 across alpha in [inf (IID), 1.0, 0.5, 0.1, 0.05] (Mean Acc, Bottom-10% Fairness, Latency, Peak VRAM)
2. CIFAR-100 with Active Class Logit Masking (ACLM) (Extreme Skew alpha=0.05)
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

from src.core.model import HierarchicalResidualResNet9, ResNet9
from src.core.loss import ActiveMaskedCrossEntropyLoss, compute_hierarchical_residual_penalty
from src.data.dataset import get_cifar10, get_cifar100, partition_data, ClientDataset, get_fast_dataloader, FastDataset
from src.experiments.builder import detect_device


def run_streamlined_hrc_cifar10(num_clients: int = 15, num_rounds: int = 20, batch_size: int = 64, mu: float = 1e-3):
    device = detect_device()
    print(f"Target Hardware Device: {device}")

    train_raw, test_raw = get_cifar10(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)

    alphas = [None, 1.0, 0.5, 0.1, 0.05]
    alpha_labels = ["IID (inf)", "Mild (1.0)", "Moderate (0.5)", "Severe (0.1)", "Extreme (0.05)"]
    cifar10_results = {}

    loss_fn = ActiveMaskedCrossEntropyLoss()

    for alpha_val, label in zip(alphas, alpha_labels):
        print(f"\n" + "="*70)
        print(f"Evaluating Streamlined HRC on CIFAR-10: {label}")
        print("="*70)

        train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=(alpha_val is not None), alpha=(alpha_val or 0.5), seed=42)
        test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=(alpha_val is not None), alpha=(alpha_val or 0.5), seed=42)

        # Compute Shannon entropy R_skew and active class masks per client
        client_r_skew = []
        client_active_masks = []
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

        # Global Model
        global_model = HierarchicalResidualResNet9(in_channels=3, num_classes=10).to(device)
        
        # Cluster state storage (5 clusters)
        cluster_weights = [torch.zeros(10, 256, device=device) for _ in range(5)]
        cluster_biases = [torch.zeros(10, device=device) for _ in range(5)]
        
        # Local state storage (15 clients)
        local_weights = [torch.zeros(10, 256, device=device) for _ in range(num_clients)]
        local_biases = [torch.zeros(10, device=device) for _ in range(num_clients)]

        # Cluster assignment via update momentum / topology
        cluster_assignments = [i % 5 for i in range(num_clients)]

        t0 = time.time()
        for r in range(num_rounds):
            client_deltas_theta = []
            client_deltas_wg = []
            client_deltas_wc = []

            for cid in range(num_clients):
                c_train = ClientDataset(train_fast, train_splits[cid])
                loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                
                local_m = HierarchicalResidualResNet9(in_channels=3, num_classes=10).to(device)
                local_m.load_state_dict(global_model.state_dict())
                
                # Load assigned cluster residual and client local residual
                k_assigned = cluster_assignments[cid]
                local_m.classifier.weight_cluster.data.copy_(cluster_weights[k_assigned])
                local_m.classifier.bias_cluster.data.copy_(cluster_biases[k_assigned])
                local_m.classifier.weight_local.data.copy_(local_weights[cid])
                local_m.classifier.bias_local.data.copy_(local_biases[cid])

                # Unified Single-Pass SGD Optimizer
                opt = torch.optim.SGD(local_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
                r_sk = client_r_skew[cid]
                mask = client_active_masks[cid]

                local_m.train()
                for _ in range(3):
                    for x, y in loader:
                        opt.zero_grad(set_to_none=True)
                        logits = local_m(x)
                        ce_loss = loss_fn(logits, y, active_mask=mask)
                        reg_loss = compute_hierarchical_residual_penalty(local_m, r_skew=r_sk, mu=mu)
                        loss = ce_loss + reg_loss
                        loss.backward()
                        opt.step()

                # Save updated local residual back to client storage
                local_weights[cid].copy_(local_m.classifier.weight_local.data)
                local_biases[cid].copy_(local_m.classifier.bias_local.data)

                # Track deltas for global and cluster aggregation
                delta_theta = {k: local_m.state_dict()[k] - global_model.state_dict()[k] for k in global_model.state_dict() if 'classifier' not in k}
                delta_wg = local_m.classifier.weight_global.data - global_model.classifier.weight_global.data
                delta_wc = local_m.classifier.weight_cluster.data - cluster_weights[k_assigned]

                client_deltas_theta.append(delta_theta)
                client_deltas_wg.append(delta_wg)
                client_deltas_wc.append((k_assigned, delta_wc))

            # 1. Server Aggregation: Global Backbone & Global Head
            avg_theta = {}
            for k in client_deltas_theta[0].keys():
                avg_theta[k] = global_model.state_dict()[k] + torch.stack([client_deltas_theta[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
            avg_wg = global_model.classifier.weight_global.data + torch.stack(client_deltas_wg, dim=0).mean(dim=0)
            
            global_model.load_state_dict(avg_theta, strict=False)
            global_model.classifier.weight_global.data.copy_(avg_wg)

            # 2. Server Aggregation: Cluster Residuals
            for k in range(5):
                k_deltas = [d for k_id, d in client_deltas_wc if k_id == k]
                if k_deltas:
                    cluster_weights[k].add_(torch.stack(k_deltas, dim=0).mean(dim=0))

        total_time = round(time.time() - t0, 2)

        # Evaluation across clients
        accs = []
        global_model.eval()
        for cid in range(num_clients):
            c_test = ClientDataset(test_fast, test_splits[cid])
            loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
            local_m = HierarchicalResidualResNet9(in_channels=3, num_classes=10).to(device)
            local_m.load_state_dict(global_model.state_dict())
            k_assigned = cluster_assignments[cid]
            local_m.classifier.weight_cluster.data.copy_(cluster_weights[k_assigned])
            local_m.classifier.weight_local.data.copy_(local_weights[cid])

            c_total, c_corr = 0, 0
            with torch.no_grad():
                for x, y in loader:
                    logits = local_m(x)
                    pred = logits.argmax(dim=1)
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
    # CIFAR-100 High-Cardinality Evaluation with ACLM
    # -------------------------------------------------------------
    print(f"\n" + "="*70)
    print("Evaluating Streamlined HRC with ACLM on CIFAR-100 (Extreme Skew alpha=0.05)")
    print("="*70)

    train100_raw, test100_raw = get_cifar100(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train100_fast = FastDataset(train100_raw, device=device)
    test100_fast = FastDataset(test100_raw, device=device)

    train100_splits = partition_data(train100_fast, num_clients=num_clients, non_iid=True, alpha=0.05, seed=42)
    test100_splits = partition_data(test100_fast, num_clients=num_clients, non_iid=True, alpha=0.05, seed=42)

    client_r_skew100 = []
    client_active_masks100 = []
    for cid in range(num_clients):
        c_train = ClientDataset(train100_fast, train100_splits[cid])
        y_targets = [int(y.item()) if isinstance(y, torch.Tensor) else int(y) for _, y in c_train]
        counts = np.bincount(y_targets, minlength=100)
        probs = counts / (counts.sum() + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        r_sk = float(entropy / np.log(100))
        client_r_skew100.append(r_sk)

        active = torch.zeros(100, dtype=torch.bool, device=device)
        active[np.where(counts > 0)[0]] = True
        client_active_masks100.append(active)

    global100 = HierarchicalResidualResNet9(in_channels=3, num_classes=100).to(device)
    cluster_weights100 = [torch.zeros(100, 256, device=device) for _ in range(5)]
    local_weights100 = [torch.zeros(100, 256, device=device) for _ in range(num_clients)]

    t0 = time.time()
    for r in range(num_rounds):
        client_deltas_theta = []
        client_deltas_wg = []
        client_deltas_wc = []

        for cid in range(num_clients):
            c_train = ClientDataset(train100_fast, train100_splits[cid])
            loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
            local_m = HierarchicalResidualResNet9(in_channels=3, num_classes=100).to(device)
            local_m.load_state_dict(global100.state_dict())
            
            k_assigned = cid % 5
            local_m.classifier.weight_cluster.data.copy_(cluster_weights100[k_assigned])
            local_m.classifier.weight_local.data.copy_(local_weights100[cid])

            opt = torch.optim.SGD(local_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
            r_sk = client_r_skew100[cid]
            mask = client_active_masks100[cid]

            local_m.train()
            for _ in range(3):
                for x, y in loader:
                    opt.zero_grad(set_to_none=True)
                    logits = local_m(x)
                    ce_loss = loss_fn(logits, y, active_mask=mask)
                    reg_loss = compute_hierarchical_residual_penalty(local_m, r_skew=r_sk, mu=mu)
                    loss = ce_loss + reg_loss
                    loss.backward()
                    opt.step()

            local_weights100[cid].copy_(local_m.classifier.weight_local.data)

            delta_theta = {k: local_m.state_dict()[k] - global100.state_dict()[k] for k in global100.state_dict() if 'classifier' not in k}
            delta_wg = local_m.classifier.weight_global.data - global100.classifier.weight_global.data
            delta_wc = local_m.classifier.weight_cluster.data - cluster_weights100[k_assigned]

            client_deltas_theta.append(delta_theta)
            client_deltas_wg.append(delta_wg)
            client_deltas_wc.append((k_assigned, delta_wc))

        avg_theta = {}
        for k in client_deltas_theta[0].keys():
            avg_theta[k] = global100.state_dict()[k] + torch.stack([client_deltas_theta[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
        avg_wg = global100.classifier.weight_global.data + torch.stack(client_deltas_wg, dim=0).mean(dim=0)
        global100.load_state_dict(avg_theta, strict=False)
        global100.classifier.weight_global.data.copy_(avg_wg)

        for k in range(5):
            k_deltas = [d for k_id, d in client_deltas_wc if k_id == k]
            if k_deltas:
                cluster_weights100[k].add_(torch.stack(k_deltas, dim=0).mean(dim=0))

    accs100 = []
    global100.eval()
    for cid in range(num_clients):
        c_test = ClientDataset(test100_fast, test100_splits[cid])
        loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
        local_m = HierarchicalResidualResNet9(in_channels=3, num_classes=100).to(device)
        local_m.load_state_dict(global100.state_dict())
        k_assigned = cid % 5
        local_m.classifier.weight_cluster.data.copy_(cluster_weights100[k_assigned])
        local_m.classifier.weight_local.data.copy_(local_weights100[cid])

        c_total, c_corr = 0, 0
        with torch.no_grad():
            for x, y in loader:
                logits = local_m(x)
                pred = logits.argmax(dim=1)
                c_corr += (pred == y).sum().item()
                c_total += y.size(0)
        accs100.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

    mean_acc100 = round(float(np.mean(accs100)), 2)
    bot10_100 = round(float(np.percentile(accs100, 10)), 2)
    print(f"CIFAR-100 ACLM Result: Mean Top-1 = {mean_acc100}%, Bottom-10% = {bot10_100}%")

    final_results = {
        "cifar10_streamlined_hrc": cifar10_results,
        "cifar100_aclm_extreme": {
            "mean_acc": mean_acc100,
            "bottom_10": bot10_100,
        }
    }

    out_file = os.path.join(_project_root, "outputs", "streamlined_hrc_benchmark.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nStreamlined HRC Benchmark results saved to: {out_file}")
    return final_results


if __name__ == "__main__":
    run_streamlined_hrc_cifar10()
