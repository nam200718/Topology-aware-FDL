"""
Client Population & Participation Scalability Experiment (N = 50, Cp = 0.2).

Evaluates FedAvg, FedRep, Ditto, and HEP with Staleness-Aware Fallback Routing (S-AFR)
and Asynchronous Cluster Momentum on CIFAR-10 with 50 edge clients,
where only 10 clients (20% participation fraction) are sampled per round.

Usage:
    python scripts/run_scale_50clients.py
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

from src.core.model import ResNet9, MultiHeadResNet9
from src.data.dataset import get_cifar10, partition_data, ClientDataset, get_fast_dataloader, FastDataset
from src.experiments.builder import detect_device


def run_50clients_scaling(num_clients: int = 50, clients_per_round: int = 10, num_rounds: int = 20, batch_size: int = 64):
    device = detect_device()
    print(f"Target Hardware Device: {device}")

    # Load CIFAR-10 preloaded to GPU memory
    train_raw, test_raw = get_cifar10(data_dir="./data", train_subset=15000, test_subset=3000, seed=42)
    print("Preloading CIFAR-10 to GPU memory for 50-client scaling...")
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)

    scenarios = [
        ("Moderate (alpha=0.5)", 0.5),
        ("Severe (alpha=0.1)", 0.1),
    ]

    all_results = {}

    for sc_name, alpha in scenarios:
        print(f"\n{'='*70}\nRunning 50-Client Scalability Scenario: {sc_name} (Cp = {clients_per_round/num_clients:.1f})\n{'='*70}")
        train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=True, alpha=alpha, seed=42)
        test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=True, alpha=alpha, seed=42)

        scenario_res = {}
        crit = nn.CrossEntropyLoss()
        rng = np.random.RandomState(42)

        # -------------------------------------------------------------
        # 1. FedAvg (50 clients, 10 active/round)
        # -------------------------------------------------------------
        print("\n[1/4] Training FedAvg (N=50, Cp=0.2)...")
        global_model = ResNet9(in_channels=3, num_classes=10).to(device)

        for r in range(num_rounds):
            active_clients = rng.choice(num_clients, clients_per_round, replace=False)
            client_states = []
            for cid in active_clients:
                c_train = ClientDataset(train_fast, train_splits[cid])
                loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                local_m = ResNet9(in_channels=3, num_classes=10).to(device)
                local_m.load_state_dict(global_model.state_dict())
                opt = torch.optim.SGD(local_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

                local_m.train()
                for _ in range(2):
                    for x, y in loader:
                        opt.zero_grad(set_to_none=True)
                        loss = crit(local_m(x), y)
                        loss.backward()
                        opt.step()
                client_states.append(local_m.state_dict())

            avg_state = {}
            for k in global_model.state_dict().keys():
                avg_state[k] = torch.stack([client_states[i][k].float() for i in range(len(active_clients))], dim=0).mean(dim=0)
            global_model.load_state_dict(avg_state)

        # Evaluate across all 50 clients
        accs = []
        global_model.eval()
        with torch.no_grad():
            for cid in range(num_clients):
                c_test = ClientDataset(test_fast, test_splits[cid])
                loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
                c_total, c_corr = 0, 0
                for x, y in loader:
                    pred = global_model(x).argmax(dim=1)
                    c_corr += (pred == y).sum().item()
                    c_total += y.size(0)
                accs.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

        scenario_res["FedAvg"] = {
            "mean": round(float(np.mean(accs)), 2),
            "bottom10": round(float(np.mean(sorted(accs)[:5])), 2)
        }
        print(f"  FedAvg: Mean = {scenario_res['FedAvg']['mean']:.2f}% | Bottom 10% = {scenario_res['FedAvg']['bottom10']:.2f}%")

        # -------------------------------------------------------------
        # 2. FedRep (50 clients, 10 active/round)
        # -------------------------------------------------------------
        print("\n[2/4] Training FedRep (N=50, Cp=0.2)...")
        global_bb = ResNet9(in_channels=3, num_classes=10).to(device)
        local_heads = [nn.Linear(256, 10).to(device) for _ in range(num_clients)]

        for r in range(num_rounds):
            active_clients = rng.choice(num_clients, clients_per_round, replace=False)
            client_bb_states = []
            for cid in active_clients:
                c_train = ClientDataset(train_fast, train_splits[cid])
                loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                local_bb = ResNet9(in_channels=3, num_classes=10).to(device)
                local_bb.load_state_dict(global_bb.state_dict())
                l_head = local_heads[cid]

                opt_head = torch.optim.SGD(l_head.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
                opt_bb = torch.optim.SGD(local_bb.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

                # 1. Train head on detached features
                for _ in range(2):
                    for x, y in loader:
                        opt_head.zero_grad(set_to_none=True)
                        feats = local_bb.extract_features(x).detach()
                        loss = crit(l_head(feats), y)
                        loss.backward()
                        opt_head.step()

                # 2. Train body representation
                for _ in range(2):
                    for x, y in loader:
                        opt_bb.zero_grad(set_to_none=True)
                        feats = local_bb.extract_features(x)
                        loss = crit(l_head(feats), y)
                        loss.backward()
                        opt_bb.step()

                client_bb_states.append(local_bb.state_dict())

            avg_bb = {}
            for k in global_bb.state_dict().keys():
                avg_bb[k] = torch.stack([client_bb_states[i][k].float() for i in range(len(active_clients))], dim=0).mean(dim=0)
            global_bb.load_state_dict(avg_bb)

        # Eval FedRep
        accs = []
        global_bb.eval()
        with torch.no_grad():
            for cid in range(num_clients):
                local_heads[cid].eval()
                c_test = ClientDataset(test_fast, test_splits[cid])
                loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
                c_total, c_corr = 0, 0
                for x, y in loader:
                    pred = local_heads[cid](global_bb.extract_features(x)).argmax(dim=1)
                    c_corr += (pred == y).sum().item()
                    c_total += y.size(0)
                accs.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

        scenario_res["FedRep"] = {
            "mean": round(float(np.mean(accs)), 2),
            "bottom10": round(float(np.mean(sorted(accs)[:5])), 2)
        }
        print(f"  FedRep: Mean = {scenario_res['FedRep']['mean']:.2f}% | Bottom 10% = {scenario_res['FedRep']['bottom10']:.2f}%")

        # -------------------------------------------------------------
        # 3. Ditto (50 clients, 10 active/round)
        # -------------------------------------------------------------
        print("\n[3/4] Training Ditto (N=50, Cp=0.2)...")
        global_m = ResNet9(in_channels=3, num_classes=10).to(device)
        local_models = [ResNet9(in_channels=3, num_classes=10).to(device) for _ in range(num_clients)]

        for r in range(num_rounds):
            active_clients = rng.choice(num_clients, clients_per_round, replace=False)
            client_states = []
            for cid in active_clients:
                c_train = ClientDataset(train_fast, train_splits[cid])
                loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                local_g = ResNet9(in_channels=3, num_classes=10).to(device)
                local_g.load_state_dict(global_m.state_dict())
                opt_g = torch.optim.SGD(local_g.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

                local_g.train()
                for _ in range(2):
                    for x, y in loader:
                        opt_g.zero_grad(set_to_none=True)
                        loss = crit(local_g(x), y)
                        loss.backward()
                        opt_g.step()
                client_states.append(local_g.state_dict())

                # Personalized model update with proximal term
                p_mod = local_models[cid]
                opt_p = torch.optim.SGD(p_mod.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
                w_g_vec = torch.nn.utils.parameters_to_vector(local_g.parameters()).detach()

                p_mod.train()
                for _ in range(2):
                    for x, y in loader:
                        opt_p.zero_grad(set_to_none=True)
                        w_p_vec = torch.nn.utils.parameters_to_vector(p_mod.parameters())
                        loss = crit(p_mod(x), y) + 0.5 * 0.1 * torch.sum((w_p_vec - w_g_vec) ** 2)
                        loss.backward()
                        opt_p.step()

            avg_state = {}
            for k in global_m.state_dict().keys():
                avg_state[k] = torch.stack([client_states[i][k].float() for i in range(len(active_clients))], dim=0).mean(dim=0)
            global_m.load_state_dict(avg_state)

        # Eval Ditto
        accs = []
        with torch.no_grad():
            for cid in range(num_clients):
                local_models[cid].eval()
                c_test = ClientDataset(test_fast, test_splits[cid])
                loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
                c_total, c_corr = 0, 0
                for x, y in loader:
                    pred = local_models[cid](x).argmax(dim=1)
                    c_corr += (pred == y).sum().item()
                    c_total += y.size(0)
                accs.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

        scenario_res["Ditto"] = {
            "mean": round(float(np.mean(accs)), 2),
            "bottom10": round(float(np.mean(sorted(accs)[:5])), 2)
        }
        print(f"  Ditto: Mean = {scenario_res['Ditto']['mean']:.2f}% | Bottom 10% = {scenario_res['Ditto']['bottom10']:.2f}%")

        # -------------------------------------------------------------
        # 4. HEP with Staleness-Aware Fallback Routing (S-AFR) & Cluster Momentum
        # -------------------------------------------------------------
        print("\n[4/4] Training HEP w/ S-AFR & Cluster Momentum (N=50, K=5, Cp=0.2)...")
        num_clusters = 5
        global_backbone = ResNet9(in_channels=3, num_classes=10).to(device)
        global_root_head = nn.Linear(256, 10).to(device)
        cluster_heads = [nn.Linear(256, 10).to(device) for _ in range(num_clusters)]
        local_heads = [nn.Linear(256, 10).to(device) for _ in range(num_clients)]
        client_alphas = [torch.tensor([0.33, 0.33, 0.34], device=device) for _ in range(num_clients)]
        client_clusters = [i % num_clusters for i in range(num_clients)]
        sample_counts = np.zeros(num_clients, dtype=int)
        last_sampled_round = np.zeros(num_clients, dtype=int)

        # Initialize local heads from global root
        for cid in range(num_clients):
            local_heads[cid].load_state_dict(global_root_head.state_dict())

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

        beta_c = 0.70  # Asynchronous cluster momentum factor

        for r in range(num_rounds):
            active_clients = rng.choice(num_clients, clients_per_round, replace=False)
            client_bb_states = []
            client_root_states = []
            cluster_updates = {k: [] for k in range(num_clusters)}

            for cid in active_clients:
                sample_counts[cid] += 1
                last_sampled_round[cid] = r
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

                e_r, e_p, e_l = 4, 2, 2
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

            # Server aggregation for backbone and root head
            avg_bb = {}
            for k in global_backbone.state_dict().keys():
                avg_bb[k] = torch.stack([client_bb_states[i][k].float() for i in range(len(active_clients))], dim=0).mean(dim=0)
            global_backbone.load_state_dict(avg_bb)

            avg_root = {}
            for k in global_root_head.state_dict().keys():
                avg_root[k] = torch.stack([client_root_states[i][k].float() for i in range(len(active_clients))], dim=0).mean(dim=0)
            global_root_head.load_state_dict(avg_root)

            # Asynchronous Cluster Momentum aggregation
            for k in range(num_clusters):
                if cluster_updates[k]:
                    avg_p = {}
                    for key in cluster_heads[k].state_dict().keys():
                        batch_p = torch.stack([cluster_updates[k][i][key].float() for i in range(len(cluster_updates[k]))], dim=0).mean(dim=0)
                        curr_p = cluster_heads[k].state_dict()[key].float()
                        avg_p[key] = beta_c * curr_p + (1.0 - beta_c) * batch_p
                    cluster_heads[k].load_state_dict(avg_p)

        # Eval HEP with Staleness-Aware Fallback Routing (S-AFR)
        accs = []
        global_backbone.eval(); global_root_head.eval()
        with torch.no_grad():
            for cid in range(num_clients):
                k_idx = client_clusters[cid]
                cluster_heads[k_idx].eval()
                local_heads[cid].eval()
                a = client_alphas[cid].clone()

                # S-AFR: Fall back to Root head if client was never sampled or has high staleness
                if sample_counts[cid] == 0:
                    a = torch.tensor([0.0, 0.0, 1.0], device=device)
                else:
                    staleness = (num_rounds - 1) - last_sampled_round[cid]
                    if staleness > 4:
                        fade = float(np.exp(-staleness / 4.0))
                        a[0] *= fade
                        a[1] *= fade
                        a[2] = 1.0 - (a[0] + a[1])

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

        scenario_res["HEP (Ours)"] = {
            "mean": round(float(np.mean(accs)), 2),
            "bottom10": round(float(np.mean(sorted(accs)[:5])), 2)
        }
        print(f"  HEP w/ S-AFR: Mean = {scenario_res['HEP (Ours)']['mean']:.2f}% | Bottom 10% = {scenario_res['HEP (Ours)']['bottom10']:.2f}%")

        all_results[sc_name] = scenario_res

    out_path = os.path.join(_project_root, "outputs", "scale_50clients_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n50-Client Scalability Benchmark complete! Saved to: {out_path}")
    return all_results


if __name__ == "__main__":
    run_50clients_scaling()
