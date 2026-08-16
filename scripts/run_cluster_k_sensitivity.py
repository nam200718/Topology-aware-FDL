"""
Cluster Count Sensitivity Experiment (K in {1, 2, 3, 5, 8}).

Evaluates the impact of cluster count K on personalization accuracy and stability
under Extreme Non-IID skew (alpha = 0.05) on CIFAR-10.

Usage:
    python scripts/run_cluster_k_sensitivity.py
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

from src.core.model import ResNet9
from src.data.dataset import get_cifar10, partition_data, ClientDataset, get_fast_dataloader, FastDataset
from src.experiments.builder import detect_device


def run_k_sensitivity(num_clients: int = 15, num_rounds: int = 20, batch_size: int = 64):
    device = detect_device()
    print(f"Target Hardware Device: {device}")

    train_raw, test_raw = get_cifar10(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)

    train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=True, alpha=0.05, seed=42)
    test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=True, alpha=0.05, seed=42)

    k_values = [1, 2, 3, 5, 8]
    results = {}
    crit = nn.CrossEntropyLoss()

    print(f"\n{'='*70}\nRunning Cluster Count Sensitivity Sweep (K in {k_values}) on alpha=0.05\n{'='*70}")

    for K in k_values:
        print(f"\nEvaluating K = {K} Clusters...")
        global_backbone = ResNet9(in_channels=3, num_classes=10).to(device)
        global_root_head = nn.Linear(256, 10).to(device)
        cluster_heads = [nn.Linear(256, 10).to(device) for _ in range(K)]
        local_heads = [nn.Linear(256, 10).to(device) for _ in range(num_clients)]
        client_alphas = [torch.tensor([0.33, 0.33, 0.34], device=device) for _ in range(num_clients)]
        client_clusters = [i % K for i in range(num_clients)]

        entropy_priors = []
        for cid in range(num_clients):
            c_labels = train_fast.labels[train_splits[cid]].cpu().numpy()
            counts = np.bincount(c_labels, minlength=10)
            probs = counts / (counts.sum() + 1e-8)
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            r_skew = float(np.clip(entropy / np.log(10), 0.0, 1.0))
            pi_r = r_skew ** 2.0
            pi_l = (1.0 - pi_r) * (1.0 - r_skew)
            pi_p = max(0.0, 1.0 - pi_r - pi_l)
            entropy_priors.append(torch.tensor([pi_l, pi_p, pi_r], device=device))

        for r in range(num_rounds):
            client_bb_states = []
            client_root_states = []
            cluster_updates = {k: [] for k in range(K)}

            for cid in range(num_clients):
                c_train = ClientDataset(train_fast, train_splits[cid])
                loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                k_idx = client_clusters[cid]

                l_bb = ResNet9(in_channels=3, num_classes=10).to(device)
                l_bb.load_state_dict(global_backbone.state_dict())
                l_root = nn.Linear(256, 10).to(device)
                l_root.load_state_dict(global_root_head.state_dict())
                l_parent = cluster_heads[k_idx]
                l_local = local_heads[cid]

                params = list(l_bb.parameters()) + list(l_root.parameters()) + list(l_parent.parameters()) + list(l_local.parameters())
                opt = torch.optim.SGD(params, lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

                l_bb.train(); l_root.train(); l_parent.train(); l_local.train()

                e_r, e_p, e_l = 5, 3, 2
                for ep in range(1, max(e_r, e_p, e_l) + 1):
                    for x, y in loader:
                        opt.zero_grad(set_to_none=True)
                        feats = l_bb.extract_features(x)
                        loss = 0.0
                        if ep <= e_r: loss += crit(l_root(feats), y)
                        if ep <= e_p: loss += crit(l_parent(feats), y)
                        if ep <= e_l: loss += crit(l_local(feats), y)
                        loss.backward()
                        opt.step()

                client_bb_states.append(l_bb.state_dict())
                client_root_states.append(l_root.state_dict())
                cluster_updates[k_idx].append(l_parent.state_dict())

                l_bb.eval()
                with torch.no_grad():
                    for x, y in loader:
                        feats = l_bb.extract_features(x)
                        z_r, z_p, z_l = l_root(feats), l_parent(feats), l_local(feats)
                        acc_r = (z_r.argmax(dim=1) == y).float().mean()
                        acc_p = (z_p.argmax(dim=1) == y).float().mean()
                        acc_l = (z_l.argmax(dim=1) == y).float().mean()
                        grad_a = torch.tensor([acc_l, acc_p, acc_r], device=device)
                        client_alphas[cid] = F.softmax(0.7 * grad_a + 0.3 * entropy_priors[cid], dim=0)
                        break

            # Server aggregation
            avg_bb = {}
            for key in global_backbone.state_dict().keys():
                avg_bb[key] = torch.stack([client_bb_states[i][key].float() for i in range(num_clients)], dim=0).mean(dim=0)
            global_backbone.load_state_dict(avg_bb)

            avg_root = {}
            for key in global_root_head.state_dict().keys():
                avg_root[key] = torch.stack([client_root_states[i][key].float() for i in range(num_clients)], dim=0).mean(dim=0)
            global_root_head.load_state_dict(avg_root)

            for k in range(K):
                if cluster_updates[k]:
                    avg_p = {}
                    for key in cluster_heads[k].state_dict().keys():
                        avg_p[key] = torch.stack([cluster_updates[k][i][key].float() for i in range(len(cluster_updates[k]))], dim=0).mean(dim=0)
                    cluster_heads[k].load_state_dict(avg_p)

        # Eval
        accs = []
        global_backbone.eval(); global_root_head.eval()
        with torch.no_grad():
            for cid in range(num_clients):
                k_idx = client_clusters[cid]
                cluster_heads[k_idx].eval()
                local_heads[cid].eval()
                a = client_alphas[cid]

                c_test = ClientDataset(test_fast, test_splits[cid])
                loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
                c_total, c_corr = 0, 0
                for x, y in loader:
                    feats = global_backbone.extract_features(x)
                    z_r = global_root_head(feats)
                    z_p = cluster_heads[k_idx](feats)
                    z_l = local_heads[cid](feats)
                    z_blend = a[0] * z_l + a[1] * z_p + a[2] * z_r
                    pred = z_blend.argmax(dim=1)
                    c_corr += (pred == y).sum().item()
                    c_total += y.size(0)
                accs.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

        results[f"K={K}"] = {
            "mean": round(float(np.mean(accs)), 2),
            "bottom10": round(float(np.mean(sorted(accs)[:2])), 2),
            "top10": round(float(np.mean(sorted(accs)[-2:])), 2),
            "std": round(float(np.std(accs)), 2)
        }
        print(f"  K={K} Result: Mean = {results[f'K={K}']['mean']:.2f}% | Bottom 10% = {results[f'K={K}']['bottom10']:.2f}% | Std = {results[f'K={K}']['std']:.2f}%")

    out_path = os.path.join(_project_root, "outputs", "cluster_k_sensitivity.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCluster sensitivity sweep completed! Saved to: {out_path}")
    return results


if __name__ == "__main__":
    run_k_sensitivity()
