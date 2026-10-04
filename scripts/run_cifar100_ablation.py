"""
Streamlined, High-Throughput CIFAR-100 Ablation Study Runner (Table VI).

Evaluates the 4 architectural ablation variants of FedHEP on CIFAR-100 ResNet-9:
  1. topo_no_aclm   : w/o Active-Class Logit Masking (ACLM)
  2. topo_no_parent : w/o Collaborative Parent Head (2-Tier Bipartite: Root + Local)
  3. topo_k1        : K=1 Grand Coalition (Single Cluster, No Sub-Coalitions)
  4. topo_oracle_k3 : K=3 Oracle Label-Distribution Clustering Bound

Optimized for Cloud GPUs (RTX 4090):
  * Preloads CIFAR-100 into GPU memory (FastDataset) -- zero PCIe/DataLoader latency.
  * In-place model state updates on CUDA -- zero CPU-GPU deepcopy overhead.
  * Evaluates final round test metrics with bottom-10% fairness.
  * Checkpoints atomic JSON after each completed method & regime.
  * Compatible with scripts/generate_all_tables.py -> outputs/tables/table6_ablation.tex.

Usage:
    python scripts/run_cifar100_ablation.py --rounds 25 --seeds 42 123 7
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.model import ResNet9
from src.data.dataset import (
    get_cifar10,
    get_cifar100,
    partition_data,
    ClientDataset,
    get_fast_dataloader,
    FastDataset,
)
from src.experiments.builder import detect_device

OUT_DIR = os.path.join(_project_root, "outputs", "baselines", "ablation")

REGIME_CONFIGS = {
    "iid": {"alpha": None, "non_iid": False, "label": "IID (alpha=inf)"},
    "mild": {"alpha": 1.0, "non_iid": True, "label": "Mild (alpha=1.0)"},
    "moderate": {"alpha": 0.5, "non_iid": True, "label": "Moderate (alpha=0.5)"},
    "severe": {"alpha": 0.1, "non_iid": True, "label": "Severe (alpha=0.1)"},
    "extreme": {"alpha": 0.05, "non_iid": True, "label": "Extreme (alpha=0.05)"},
}

ABLATION_METHODS = [
    "topo_no_aclm",
    "topo_no_parent",
    "topo_k1",
    "topo_oracle_k3",
]


def _build_oracle_clusters(
    num_clients: int,
    train_fast: FastDataset,
    train_splits: Dict[int, List[int]],
    num_clusters: int = 3,
    seed: int = 42,
) -> List[int]:
    """Clusters clients by ground-truth label distribution (K-Means on class histograms)."""
    rng = np.random.RandomState(seed)
    num_classes = 100
    client_vectors = np.zeros((num_clients, num_classes), dtype=np.float32)
    for cid in range(num_clients):
        c_labels = train_fast.labels[train_splits[cid]].cpu().numpy()
        counts = np.bincount(c_labels, minlength=num_classes)
        norm = np.linalg.norm(counts)
        client_vectors[cid] = counts / max(norm, 1e-8)

    num_clusters = min(num_clusters, num_clients)
    init_indices = rng.choice(num_clients, num_clusters, replace=False)
    centroids = client_vectors[init_indices].copy()

    assignments = np.zeros(num_clients, dtype=int)
    for _ in range(10):
        dists = np.linalg.norm(
            client_vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )
        assignments = np.argmin(dists, axis=1)
        for k in range(num_clusters):
            members = client_vectors[assignments == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
    return assignments.tolist()


def run_single_simulation(
    method_id: str,
    regime_id: str,
    seed: int,
    device: torch.device,
    train_fast: FastDataset,
    test_fast: FastDataset,
    num_clients: int = 15,
    num_rounds: int = 25,
    local_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 0.05,
) -> Tuple[float, float]:
    """Runs one training simulation and returns (test_accuracy, bottom10_fairness)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    num_classes = 100

    reg_info = REGIME_CONFIGS[regime_id]
    train_splits = partition_data(
        train_fast,
        num_clients=num_clients,
        non_iid=reg_info["non_iid"],
        alpha=reg_info["alpha"] if reg_info["non_iid"] else 1.0,
        seed=seed,
    )
    test_splits = partition_data(
        test_fast,
        num_clients=num_clients,
        non_iid=reg_info["non_iid"],
        alpha=reg_info["alpha"] if reg_info["non_iid"] else 1.0,
        seed=seed,
    )

    # Active class masks for ACLM
    client_masks = []
    client_entropy_priors = []
    for cid in range(num_clients):
        c_labels = train_fast.labels[train_splits[cid]].cpu().numpy()
        mask = np.zeros(num_classes, dtype=bool)
        mask[np.unique(c_labels)] = True
        client_masks.append(torch.tensor(mask, device=device))

        counts = np.bincount(c_labels, minlength=num_classes)
        p = counts / (counts.sum() + 1e-8)
        entropy = -np.sum(p * np.log(p + 1e-12))
        r_skew = float(np.clip(entropy / np.log(num_classes), 0.0, 1.0))
        pi_r = r_skew ** 2.0
        pi_l = (1.0 - pi_r) * (1.0 - r_skew)
        pi_p = max(0.0, 1.0 - pi_r - pi_l)
        client_entropy_priors.append(
            torch.tensor([pi_l, pi_p, pi_r], device=device)
        )

    # Configuration switches based on ablation variant
    use_aclm = method_id != "topo_no_aclm"
    enable_parent = method_id != "topo_no_parent"

    if method_id == "topo_k1":
        num_clusters = 1
        client_clusters = [0] * num_clients
    elif method_id == "topo_oracle_k3":
        num_clusters = 3
        client_clusters = _build_oracle_clusters(
            num_clients, train_fast, train_splits, num_clusters=3, seed=seed
        )
    else:
        num_clusters = 3
        client_clusters = [i % num_clusters for i in range(num_clients)]

    # Model Heads & Shared Backbone
    global_backbone = ResNet9(in_channels=3, num_classes=num_classes).to(device)
    global_root_head = nn.Linear(256, num_classes).to(device)
    cluster_heads = [
        nn.Linear(256, num_classes).to(device) for _ in range(num_clusters)
    ]
    local_heads = [
        nn.Linear(256, num_classes).to(device) for _ in range(num_clients)
    ]

    for cid in range(num_clients):
        local_heads[cid].load_state_dict(global_root_head.state_dict())

    client_alphas = [
        torch.tensor([0.33, 0.33, 0.34], device=device)
        for _ in range(num_clients)
    ]

    crit = nn.CrossEntropyLoss()
    beta_c = 0.70  # Asynchronous cluster momentum

    e_r = local_epochs
    e_p = local_epochs if enable_parent else 0
    e_l = local_epochs

    # Training rounds
    for r in range(num_rounds):
        client_bb_states = []
        client_root_states = []
        cluster_updates = {k: [] for k in range(num_clusters)}

        for cid in range(num_clients):
            c_train = ClientDataset(train_fast, train_splits[cid])
            loader = get_fast_dataloader(
                c_train, batch_size=batch_size, shuffle=True
            )
            k_idx = client_clusters[cid]
            mask = client_masks[cid]

            l_bb = copy.deepcopy(global_backbone).train()
            l_root = copy.deepcopy(global_root_head).train()
            l_parent = copy.deepcopy(cluster_heads[k_idx]).train()
            l_local = local_heads[cid].train()

            params = (
                list(l_bb.parameters())
                + list(l_root.parameters())
                + list(l_local.parameters())
            )
            if enable_parent:
                params += list(l_parent.parameters())

            opt = torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=1e-4, foreach=False
            )

            max_e = max(e_r, e_p, e_l)
            for ep in range(1, max_e + 1):
                for x, y in loader:
                    opt.zero_grad(set_to_none=True)
                    feats = l_bb.extract_features(x)
                    loss = 0.0

                    if ep <= e_r:
                        loss = loss + crit(l_root(feats), y)

                    if enable_parent and ep <= e_p:
                        z_p = l_parent(feats)
                        if use_aclm:
                            z_p = z_p.masked_fill(~mask.unsqueeze(0), -1e9)
                        loss = loss + crit(z_p, y)

                    if ep <= e_l:
                        z_l = l_local(feats)
                        if use_aclm:
                            z_l = z_l.masked_fill(~mask.unsqueeze(0), -1e9)
                        loss = loss + crit(z_l, y)

                    loss.backward()
                    opt.step()

            client_bb_states.append(l_bb.state_dict())
            client_root_states.append(l_root.state_dict())
            if enable_parent:
                cluster_updates[k_idx].append(l_parent.state_dict())

            # Dynamic alpha calibration
            l_bb.eval()
            with torch.no_grad():
                x_b, y_b = next(iter(loader))
                feats_b = l_bb.extract_features(x_b)
                z_rb = l_root(feats_b)
                z_lb = l_local(feats_b)
                if use_aclm:
                    z_lb = z_lb.masked_fill(~mask.unsqueeze(0), -1e9)

                acc_r = (z_rb.argmax(dim=1) == y_b).float().mean()
                acc_l = (z_lb.argmax(dim=1) == y_b).float().mean()

                if enable_parent:
                    z_pb = l_parent(feats_b)
                    if use_aclm:
                        z_pb = z_pb.masked_fill(~mask.unsqueeze(0), -1e9)
                    acc_p = (z_pb.argmax(dim=1) == y_b).float().mean()
                    grad_a = torch.tensor([acc_l, acc_p, acc_r], device=device)
                    client_alphas[cid] = F.softmax(
                        0.7 * grad_a + 0.3 * client_entropy_priors[cid], dim=0
                    )
                else:
                    client_alphas[cid] = torch.tensor(
                        [0.60, 0.0, 0.40], device=device
                    )

        # Server backbone and root aggregation
        avg_bb = {
            k: torch.stack([s[k].float() for s in client_bb_states], dim=0).mean(
                dim=0
            )
            for k in global_backbone.state_dict().keys()
        }
        global_backbone.load_state_dict(avg_bb)

        avg_root = {
            k: torch.stack(
                [s[k].float() for s in client_root_states], dim=0
            ).mean(dim=0)
            for k in global_root_head.state_dict().keys()
        }
        global_root_head.load_state_dict(avg_root)

        # Cluster head momentum aggregation
        if enable_parent:
            for k in range(num_clusters):
                if cluster_updates[k]:
                    avg_p = {
                        key: torch.stack(
                            [s[key].float() for s in cluster_updates[k]], dim=0
                        ).mean(dim=0)
                        for key in cluster_heads[k].state_dict().keys()
                    }
                    curr_p = cluster_heads[k].state_dict()
                    updated_p = {
                        key: beta_c * curr_p[key].float()
                        + (1.0 - beta_c) * avg_p[key]
                        for key in avg_p.keys()
                    }
                    cluster_heads[k].load_state_dict(updated_p)

    # Final evaluation across all clients
    accs = []
    global_backbone.eval()
    global_root_head.eval()
    for h in cluster_heads:
        h.eval()
    for h in local_heads:
        h.eval()

    with torch.no_grad():
        for cid in range(num_clients):
            c_test = ClientDataset(test_fast, test_splits[cid])
            loader = get_fast_dataloader(
                c_test, batch_size=batch_size, shuffle=False
            )
            k_idx = client_clusters[cid]
            mask = client_masks[cid]
            a = client_alphas[cid]

            c_total, c_corr = 0, 0
            for x, y in loader:
                feats = global_backbone.extract_features(x)
                z_r = global_root_head(feats)
                z_l = local_heads[cid](feats)

                if enable_parent:
                    z_p = cluster_heads[k_idx](feats)
                    z_blend = a[0] * z_l + a[1] * z_p + a[2] * z_r
                else:
                    z_blend = a[0] * z_l + a[2] * z_r

                if use_aclm:
                    z_blend = z_blend.masked_fill(~mask.unsqueeze(0), -1e9)

                pred = z_blend.argmax(dim=1)
                c_corr += (pred == y).sum().item()
                c_total += y.size(0)

            accs.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

    mean_acc = round(float(np.mean(accs)), 2)
    k_b10 = max(1, int(np.ceil(0.1 * len(accs))))
    b10_acc = round(float(np.mean(sorted(accs)[:k_b10])), 2)
    return mean_acc, b10_acc


def run_ablation_study(
    methods: List[str] = None,
    regimes: List[str] = None,
    seeds: List[int] = None,
    num_rounds: int = 25,
    local_epochs: int = 3,
    batch_size: int = 64,
    device: str = None,
    data_dir: str = "./data",
    train_subset: int = 10000,
    test_subset: int = 3000,
    force: bool = False,
):
    target_methods = methods or ABLATION_METHODS
    target_regimes = regimes or list(REGIME_CONFIGS.keys())
    target_seeds = seeds or [42, 123, 7]

    dev = torch.device(device) if device else detect_device()
    print(f"\n=======================================================")
    print(f"CIFAR-100 ABLATION STUDY RUNNER (Table VI)")
    print(f"Hardware Device: {dev}")
    print(f"Methods: {target_methods}")
    print(f"Regimes: {target_regimes}")
    print(f"Seeds: {target_seeds} | Rounds: {num_rounds} | Local Epochs: {local_epochs}")
    print(f"=======================================================\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "results_ablation.json")
    csv_path = os.path.join(OUT_DIR, "results_ablation.csv")

    results_table = []
    completed_keys = set()
    if os.path.exists(json_path) and not force:
        try:
            with open(json_path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    results_table = loaded
                    for item in results_table:
                        m = item.get("method")
                        r = item.get("regime")
                        if m and r:
                            completed_keys.add((m, r))
            print(f"Loaded {len(results_table)} completed entries from: {json_path}")
        except Exception as e:
            print(f"Warning: Failed to read checkpoint {json_path}: {e}")

    # Preload dataset once to GPU memory
    print(f"Preloading CIFAR-100 ({train_subset} train, {test_subset} test) to {dev} memory...")
    tr_raw, te_raw = get_cifar100(
        data_dir=data_dir,
        train_subset=train_subset,
        test_subset=test_subset,
        seed=42,
    )
    train_fast = FastDataset(tr_raw, device=dev)
    test_fast = FastDataset(te_raw, device=dev)
    print("Dataset preloaded successfully.\n")

    for m_id in target_methods:
        for r_id in target_regimes:
            key = (m_id, r_id)
            if key in completed_keys and not force:
                print(f">>> Skipping {m_id.upper()} on {r_id.upper()} (Already Completed)")
                continue

            print(f"\n>>> Running Ablation: {m_id} | Regime: {r_id} ({REGIME_CONFIGS[r_id]['label']}) <<<")
            t0 = time.time()
            per_seed_accs = []
            per_seed_b10s = []

            for seed in target_seeds:
                s_t0 = time.time()
                acc, b10 = run_single_simulation(
                    method_id=m_id,
                    regime_id=r_id,
                    seed=seed,
                    device=dev,
                    train_fast=train_fast,
                    test_fast=test_fast,
                    num_clients=15,
                    num_rounds=num_rounds,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                )
                per_seed_accs.append(acc)
                per_seed_b10s.append(b10)
                print(f"  Seed {seed}: Acc = {acc:.2f}% | Bottom-10% = {b10:.2f}% ({time.time() - s_t0:.1f}s)")

            elapsed = round(time.time() - t0, 2)
            mean_acc = round(float(np.mean(per_seed_accs)), 2)
            std_acc = round(float(np.std(per_seed_accs)), 2)
            mean_b10 = round(float(np.mean(per_seed_b10s)), 2)
            std_b10 = round(float(np.std(per_seed_b10s)), 2)

            entry = {
                "method": m_id,
                "regime": r_id,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_b10": mean_b10,
                "std_b10": std_b10,
                "per_seed_acc": per_seed_accs,
                "per_seed_b10": per_seed_b10s,
                "elapsed_s": elapsed,
            }

            if force:
                results_table = [
                    item for item in results_table
                    if not (item.get("method") == m_id and item.get("regime") == r_id)
                ]
            results_table.append(entry)
            completed_keys.add(key)

            # Atomic save
            with open(json_path, "w") as f:
                json.dump(results_table, f, indent=2)
            df = pd.DataFrame(results_table)
            df.to_csv(csv_path, index=False)

            print(f"  Summary for {m_id} ({r_id}): {mean_acc:.2f} +- {std_acc:.2f}% (B10: {mean_b10:.2f}%) in {elapsed:.1f}s\n")

    print("\n=======================================================")
    print("Ablation Study Complete!")
    print(f"Results saved to:\n  - {json_path}\n  - {csv_path}")
    print("=======================================================\n")
    return results_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-100 Streamlined Ablation Study")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to run")
    parser.add_argument("--regimes", nargs="+", default=None, help="Regimes to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7], help="Random seeds")
    parser.add_argument("--rounds", type=int, default=25, help="Communication rounds")
    parser.add_argument("--local-epochs", type=int, default=3, help="Local epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--train-subset", type=int, default=10000, help="Train subset")
    parser.add_argument("--test-subset", type=int, default=3000, help="Test subset")
    parser.add_argument("--force", action="store_true", help="Force re-run")
    args = parser.parse_args()

    run_ablation_study(
        methods=args.methods,
        regimes=args.regimes,
        seeds=args.seeds,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        device=args.device,
        data_dir=args.data_dir,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        force=args.force,
    )
