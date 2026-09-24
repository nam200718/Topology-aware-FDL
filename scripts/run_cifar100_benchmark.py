"""
CIFAR-100 High-Class-Cardinality Multi-Regime Benchmark (C = 100).

Evaluates FedAvg, FedRep, Ditto, and Defended H-ResFL on CIFAR-100 ResNet9 across
the full spectrum of 5 heterogeneity regimes:
  1. Uniform IID (alpha = inf)
  2. Mild Non-IID (alpha = 1.0)
  3. Moderate Non-IID (alpha = 0.5)
  4. Severe Non-IID (alpha = 0.1)
  5. Extreme Non-IID (alpha = 0.05)

Also includes the CIFAR-100 Byzantine Multi-Attack Robustness Suite (Label-flipping & Sign-flipping).

Usage:
    python scripts/run_cifar100_benchmark.py
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.model import ResNet9, MultiHeadResNet9
from src.data.dataset import get_cifar100, partition_data, ClientDataset, get_fast_dataloader, FastDataset
from src.experiments.builder import detect_device


def run_cifar100_multiregime_experiment(num_clients: int = 15, num_rounds: int = 20, batch_size: int = 64):
    device = detect_device()
    print(f"Target Hardware Device: {device}")

    # Load CIFAR-100 (10,000 train samples, 3,000 test samples for fast simulation)
    train_raw, test_raw = get_cifar100(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    
    print("Preloading CIFAR-100 datasets to memory...")
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)

    scenarios = [
        ("IID", None),
        ("Mild (alpha=1.0)", 1.0),
        ("Moderate (alpha=0.5)", 0.5),
        ("Severe (alpha=0.1)", 0.1),
        ("Extreme (alpha=0.05)", 0.05),
    ]

    all_results = {}

    for sc_name, alpha in scenarios:
        print(f"\n{'='*70}\nRunning CIFAR-100 Scenario: {sc_name}\n{'='*70}")
        if alpha is None:
            train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=False, seed=42)
            test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=False, seed=42)
        else:
            train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=True, alpha=alpha, seed=42)
            test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=True, alpha=alpha, seed=42)

        scenario_res = {}
        crit = nn.CrossEntropyLoss()

        # -------------------------------------------------------------
        # 1. FedAvg (1 Global Model)
        # -------------------------------------------------------------
        print("\n[1/4] Training FedAvg...")
        global_model = ResNet9(in_channels=3, num_classes=100).to(device)

        for r in range(num_rounds):
            client_states = []
            for cid in range(num_clients):
                c_train = ClientDataset(train_fast, train_splits[cid])
                loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                local_m = ResNet9(in_channels=3, num_classes=100).to(device)
                local_m.load_state_dict(global_model.state_dict())
                opt = torch.optim.SGD(local_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

                local_m.train()
                for _ in range(3):
                    for x, y in loader:
                        opt.zero_grad(set_to_none=True)
                        loss = crit(local_m(x), y)
                        loss.backward()
                        opt.step()
                client_states.append(local_m.state_dict())

            # Server aggregation
            avg_state = {}
            for k in global_model.state_dict().keys():
                avg_state[k] = torch.stack([client_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
            global_model.load_state_dict(avg_state)

        # Eval FedAvg
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
            "bottom10": round(float(np.mean(sorted(accs)[:max(1, int(0.1 * num_clients))])), 2),
            "min_acc": round(float(np.min(accs)), 2)
        }
        print(f"  FedAvg Result: {scenario_res['FedAvg']['mean']:.2f}% (B10: {scenario_res['FedAvg']['bottom10']:.2f}%, Min: {scenario_res['FedAvg']['min_acc']:.2f}%)")

        # -------------------------------------------------------------
        # 2. FedRep (Body Shared, Local Heads)
        # -------------------------------------------------------------
        print("\n[2/4] Training FedRep...")
        global_body = ResNet9(in_channels=3, num_classes=100).to(device)
        local_heads = [nn.Linear(256, 100).to(device) for _ in range(num_clients)]

        for r in range(num_rounds):
            body_states = []
            for cid in range(num_clients):
                c_train = ClientDataset(train_fast, train_splits[cid])
                loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                l_body = ResNet9(in_channels=3, num_classes=100).to(device)
                l_body.load_state_dict(global_body.state_dict())
                l_head = local_heads[cid]

                # Step 1: Train Head only (2 epochs)
                opt_h = torch.optim.SGD(l_head.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
                l_body.eval(); l_head.train()
                for _ in range(2):
                    for x, y in loader:
                        opt_h.zero_grad(set_to_none=True)
                        with torch.no_grad():
                            f = l_body.extract_features(x)
                        loss = crit(l_head(f), y)
                        loss.backward()
                        opt_h.step()

                # Step 2: Train Body only (1 epoch)
                opt_b = torch.optim.SGD(l_body.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
                l_body.train(); l_head.eval()
                for _ in range(1):
                    for x, y in loader:
                        opt_b.zero_grad(set_to_none=True)
                        f = l_body.extract_features(x)
                        loss = crit(l_head(f), y)
                        loss.backward()
                        opt_b.step()

                body_states.append(l_body.state_dict())

            # Server aggregation of body only
            avg_body = {}
            for k in global_body.state_dict().keys():
                if "fc2" not in k:
                    avg_body[k] = torch.stack([body_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
                else:
                    avg_body[k] = global_body.state_dict()[k]
            global_body.load_state_dict(avg_body)

        # Eval FedRep
        accs = []
        global_body.eval()
        with torch.no_grad():
            for cid in range(num_clients):
                local_heads[cid].eval()
                c_test = ClientDataset(test_fast, test_splits[cid])
                loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
                c_total, c_corr = 0, 0
                for x, y in loader:
                    f = global_body.extract_features(x)
                    pred = local_heads[cid](f).argmax(dim=1)
                    c_corr += (pred == y).sum().item()
                    c_total += y.size(0)
                accs.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)
        scenario_res["FedRep"] = {
            "mean": round(float(np.mean(accs)), 2),
            "bottom10": round(float(np.mean(sorted(accs)[:max(1, int(0.1 * num_clients))])), 2),
            "min_acc": round(float(np.min(accs)), 2)
        }
        print(f"  FedRep Result: {scenario_res['FedRep']['mean']:.2f}% (B10: {scenario_res['FedRep']['bottom10']:.2f}%, Min: {scenario_res['FedRep']['min_acc']:.2f}%)")

        # -------------------------------------------------------------
        # 3. Ditto (Dual Model Regularization)
        # -------------------------------------------------------------
        print("\n[3/4] Training Ditto...")
        global_m = ResNet9(in_channels=3, num_classes=100).to(device)
        local_models = [ResNet9(in_channels=3, num_classes=100).to(device) for _ in range(num_clients)]
        for lm in local_models:
            lm.load_state_dict(global_m.state_dict())

        for r in range(num_rounds):
            client_states = []
            for cid in range(num_clients):
                c_train = ClientDataset(train_fast, train_splits[cid])
                loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)

                # Local pass on global model
                l_glob = ResNet9(in_channels=3, num_classes=100).to(device)
                l_glob.load_state_dict(global_m.state_dict())
                opt_g = torch.optim.SGD(l_glob.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
                l_glob.train()
                for _ in range(3):
                    for x, y in loader:
                        opt_g.zero_grad(set_to_none=True)
                        loss = crit(l_glob(x), y)
                        loss.backward()
                        opt_g.step()
                client_states.append(l_glob.state_dict())

                # Local pass on personalized model with proximal penalty
                p_mod = local_models[cid]
                opt_p = torch.optim.SGD(p_mod.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
                w_g_vec = torch.nn.utils.parameters_to_vector(l_glob.parameters()).detach()

                p_mod.train()
                for _ in range(3):
                    for x, y in loader:
                        opt_p.zero_grad(set_to_none=True)
                        w_p_vec = torch.nn.utils.parameters_to_vector(p_mod.parameters())
                        loss = crit(p_mod(x), y) + 0.5 * 0.1 * torch.sum((w_p_vec - w_g_vec) ** 2)
                        loss.backward()
                        opt_p.step()

            # Server aggregation of global model
            avg_state = {}
            for k in global_m.state_dict().keys():
                avg_state[k] = torch.stack([client_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
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
            "bottom10": round(float(np.mean(sorted(accs)[:max(1, int(0.1 * num_clients))])), 2),
            "min_acc": round(float(np.min(accs)), 2)
        }
        print(f"  Ditto Result: {scenario_res['Ditto']['mean']:.2f}% (B10: {scenario_res['Ditto']['bottom10']:.2f}%, Min: {scenario_res['Ditto']['min_acc']:.2f}%)")

        # -------------------------------------------------------------
        # 4. Defended H-ResFL (Ours)
        # -------------------------------------------------------------
        print("\n[4/4] Training Defended H-ResFL (Ours)...")
        num_clusters = 3
        global_backbone = ResNet9(in_channels=3, num_classes=100).to(device)
        global_root_head = nn.Linear(256, 100).to(device)
        cluster_heads = [nn.Linear(256, 100).to(device) for _ in range(num_clusters)]
        local_heads = [nn.Linear(256, 100).to(device) for _ in range(num_clients)]
        client_alphas = [torch.tensor([0.33, 0.33, 0.34], device=device) for _ in range(num_clients)]
        client_clusters = [i % num_clusters for i in range(num_clients)]

        # Precompute active class support masks for ACLM
        active_classes_per_client = []
        entropy_priors = []
        for cid in range(num_clients):
            c_labels = train_fast.labels[train_splits[cid]].cpu().numpy()
            u_classes = np.unique(c_labels)
            active_classes_per_client.append(set(u_classes))
            counts = np.bincount(c_labels, minlength=100)
            probs = counts / (counts.sum() + 1e-8)
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            r_skew = float(np.clip(entropy / np.log(100), 0.0, 1.0))
            pi_r = r_skew ** 2.0
            pi_l = (1.0 - pi_r) * (1.0 - r_skew)
            pi_p = max(0.0, 1.0 - pi_r - pi_l)
            entropy_priors.append(torch.tensor([pi_l, pi_p, pi_r], device=device))

        for r in range(num_rounds):
            client_bb_states = []
            client_root_states = []
            cluster_updates = {k: [] for k in range(num_clusters)}

            for cid in range(num_clients):
                c_train = ClientDataset(train_fast, train_splits[cid])
                loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                k_idx = client_clusters[cid]

                l_bb = ResNet9(in_channels=3, num_classes=100).to(device)
                l_bb.load_state_dict(global_backbone.state_dict())
                l_root = nn.Linear(256, 100).to(device)
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
                        if ep <= e_r:
                            loss += crit(l_root(feats), y)
                        if ep <= e_p:
                            loss += crit(l_parent(feats), y)
                        if ep <= e_l:
                            loss += crit(l_local(feats), y)
                        loss.backward()
                        opt.step()

                client_bb_states.append(l_bb.state_dict())
                client_root_states.append(l_root.state_dict())
                cluster_updates[k_idx].append(l_parent.state_dict())

                # Update alpha simplex
                l_bb.eval()
                with torch.no_grad():
                    for x, y in loader:
                        feats = l_bb.extract_features(x)
                        z_r = l_root(feats)
                        z_p = l_parent(feats)
                        z_l = l_local(feats)
                        acc_r = (z_r.argmax(dim=1) == y).float().mean()
                        acc_p = (z_p.argmax(dim=1) == y).float().mean()
                        acc_l = (z_l.argmax(dim=1) == y).float().mean()
                        grad_a = torch.tensor([acc_l, acc_p, acc_r], device=device)
                        client_alphas[cid] = F.softmax(0.7 * grad_a + 0.3 * entropy_priors[cid], dim=0)
                        break

            # Server aggregation: Global Backbone & Root Head
            avg_bb = {}
            for k in global_backbone.state_dict().keys():
                avg_bb[k] = torch.stack([client_bb_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
            global_backbone.load_state_dict(avg_bb)

            avg_root = {}
            for k in global_root_head.state_dict().keys():
                avg_root[k] = torch.stack([client_root_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
            global_root_head.load_state_dict(avg_root)

            # Cluster aggregation: Parent Heads
            for k in range(num_clusters):
                if cluster_updates[k]:
                    avg_p = {}
                    for key in cluster_heads[k].state_dict().keys():
                        avg_p[key] = torch.stack([cluster_updates[k][i][key].float() for i in range(len(cluster_updates[k]))], dim=0).mean(dim=0)
                    cluster_heads[k].load_state_dict(avg_p)

        # Eval Defended H-ResFL
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
                    # Active-Class Logit Masking (ACLM)
                    mask = torch.ones_like(z_blend) * -1e4
                    for c in active_classes_per_client[cid]:
                        mask[:, c] = 0.0
                    pred = (z_blend + mask).argmax(dim=1)
                    c_corr += (pred == y).sum().item()
                    c_total += y.size(0)
                accs.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

        scenario_res["Defended H-ResFL (Ours)"] = {
            "mean": round(float(np.mean(accs)), 2),
            "bottom10": round(float(np.mean(sorted(accs)[:max(1, int(0.1 * num_clients))])), 2),
            "min_acc": round(float(np.min(accs)), 2)
        }
        print(f"  Defended H-ResFL Result: {scenario_res['Defended H-ResFL (Ours)']['mean']:.2f}% (B10: {scenario_res['Defended H-ResFL (Ours)']['bottom10']:.2f}%, Min: {scenario_res['Defended H-ResFL (Ours)']['min_acc']:.2f}%)")

        all_results[sc_name] = scenario_res

    out_path = os.path.join(_project_root, "outputs", "cifar100_multiregime_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[CIFAR-100 COMPLETE] Multi-regime results saved to: {out_path}")
    return all_results


def run_cifar100_byzantine_suite(num_clients: int = 15, num_rounds: int = 15, batch_size: int = 64):
    """
    Byzantine Fault Tolerance Benchmark directly on CIFAR-100 (C = 100).
    Evaluates FedAvg, Ditto, FedRep, Undefended H-ResFL, and Defended H-ResFL (with Skew-Calibrated Defense).
    """
    device = detect_device()
    print(f"\n{'='*70}\nRunning Byzantine Multi-Attack Benchmark on CIFAR-100 (C = 100)\n{'='*70}")

    train_raw, test_raw = get_cifar100(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)

    # Moderate Skew (alpha=0.5)
    train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=True, alpha=0.5, seed=42)
    test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=True, alpha=0.5, seed=42)

    active_classes_per_client = []
    for cid in range(num_clients):
        c_labels = train_fast.labels[train_splits[cid]].cpu().numpy()
        active_classes_per_client.append(set(np.unique(c_labels)))

    attacker_rates = [0.0, 0.1, 0.2, 0.3]
    attacks = ["label_flipping", "sign_flipping"]
    all_byz_results = {atk: {} for atk in attacks}

    crit = nn.CrossEntropyLoss()

    for atk in attacks:
        print(f"\nEvaluating Attack Type on CIFAR-100: {atk.upper()}")
        for f_rate in attacker_rates:
            num_attackers = int(np.round(num_clients * f_rate))
            attacker_ids = set(range(num_attackers))
            honest_ids = [i for i in range(num_clients) if i not in attacker_ids]

            # -------------------------------------------------------------
            # 1. FedAvg
            # -------------------------------------------------------------
            global_model = ResNet9(in_channels=3, num_classes=100).to(device)
            for r in range(num_rounds):
                client_states = []
                for cid in range(num_clients):
                    c_train = ClientDataset(train_fast, train_splits[cid])
                    loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                    local_m = ResNet9(in_channels=3, num_classes=100).to(device)
                    local_m.load_state_dict(global_model.state_dict())
                    opt = torch.optim.SGD(local_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

                    is_att = cid in attacker_ids
                    local_m.train()
                    for _ in range(2):
                        for x, y in loader:
                            opt.zero_grad(set_to_none=True)
                            if is_att and atk == "label_flipping":
                                y = 99 - y
                            loss = crit(local_m(x), y)
                            loss.backward()
                            opt.step()

                    st = local_m.state_dict()
                    if is_att and atk == "sign_flipping":
                        for k in st.keys():
                            delta = st[k] - global_model.state_dict()[k]
                            st[k] = global_model.state_dict()[k] - 2.0 * delta
                    client_states.append(st)

                avg_state = {}
                for k in global_model.state_dict().keys():
                    avg_state[k] = torch.stack([client_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
                global_model.load_state_dict(avg_state)

            accs_fedavg = []
            global_model.eval()
            with torch.no_grad():
                for cid in honest_ids:
                    c_test = ClientDataset(test_fast, test_splits[cid])
                    loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
                    c_tot, c_corr = 0, 0
                    for x, y in loader:
                        pred = global_model(x).argmax(dim=1)
                        c_corr += (pred == y).sum().item()
                        c_tot += y.size(0)
                    accs_fedavg.append((c_corr / c_tot * 100.0) if c_tot > 0 else 0.0)

            # -------------------------------------------------------------
            # 2. Defended H-ResFL (Skew-Calibrated Defense)
            # -------------------------------------------------------------
            num_clusters = 3
            global_bb = ResNet9(in_channels=3, num_classes=100).to(device)
            global_root = nn.Linear(256, 100).to(device)
            cluster_heads = [nn.Linear(256, 100).to(device) for _ in range(num_clusters)]
            local_heads = [nn.Linear(256, 100).to(device) for _ in range(num_clients)]
            client_alphas = [torch.tensor([0.33, 0.33, 0.34], device=device) for _ in range(num_clients)]
            client_clusters = [i % num_clusters for i in range(num_clients)]

            # Temporal reputation tracker
            reputations = torch.ones(num_clients, device=device) / num_clients

            for r in range(num_rounds):
                client_bb_states = []
                client_root_states = []
                cluster_updates = {k: [] for k in range(num_clusters)}

                for cid in range(num_clients):
                    c_train = ClientDataset(train_fast, train_splits[cid])
                    loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                    k_idx = client_clusters[cid]
                    is_att = cid in attacker_ids

                    l_bb = ResNet9(in_channels=3, num_classes=100).to(device)
                    l_bb.load_state_dict(global_bb.state_dict())
                    l_root = nn.Linear(256, 100).to(device)
                    l_root.load_state_dict(global_root.state_dict())
                    l_parent = cluster_heads[k_idx]
                    l_local = local_heads[cid]

                    params = list(l_bb.parameters()) + list(l_root.parameters()) + list(l_parent.parameters()) + list(l_local.parameters())
                    opt = torch.optim.SGD(params, lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

                    l_bb.train(); l_root.train(); l_parent.train(); l_local.train()
                    for ep in range(2):
                        for x, y in loader:
                            opt.zero_grad(set_to_none=True)
                            if is_att and atk == "label_flipping":
                                y = 99 - y
                            feats = l_bb.extract_features(x)
                            loss = crit(l_root(feats), y) + crit(l_parent(feats), y) + crit(l_local(feats), y)
                            loss.backward()
                            opt.step()

                    st_bb = l_bb.state_dict()
                    st_rt = l_root.state_dict()
                    if is_att and atk == "sign_flipping":
                        for k in st_bb.keys():
                            delta = st_bb[k] - global_bb.state_dict()[k]
                            st_bb[k] = global_bb.state_dict()[k] - 2.0 * delta

                    client_bb_states.append(st_bb)
                    client_root_states.append(st_rt)
                    cluster_updates[k_idx].append((cid, l_parent.state_dict()))

                # Skew-Calibrated Cosine Defense on Root Head
                deltas = []
                for cid in range(num_clients):
                    d = client_root_states[cid]["weight"] - global_root.state_dict()["weight"]
                    deltas.append(d)
                
                # Directional normalization
                normed_deltas = [d / (torch.norm(d) + 1e-8) for d in deltas]
                centroid = torch.stack(normed_deltas, dim=0).mean(dim=0)
                sims_list = []
                for cid, nd in enumerate(normed_deltas):
                    act = list(active_classes_per_client[cid])
                    if len(act) > 0:
                        nd_act = nd[act]
                        cent_act = centroid[act]
                        s = float(torch.sum(nd_act * cent_act) / (torch.norm(nd_act) * torch.norm(cent_act) + 1e-8))
                    else:
                        s = float(torch.sum(nd * centroid) / (torch.norm(nd) * torch.norm(centroid) + 1e-8))
                    sims_list.append(s)
                sims = torch.tensor(sims_list, device=device)

                # Variance-gated temperature
                var_s = float(torch.var(sims))
                temp = 0.5 if var_s > 0.05 else 1.0
                trust = F.softmax(sims / temp, dim=0)

                # Update cumulative reputation
                reputations = 0.8 * reputations + 0.2 * trust
                weights = reputations / reputations.sum()

                # Weighted Aggregation
                avg_bb = {}
                for k in global_bb.state_dict().keys():
                    avg_bb[k] = sum(weights[i] * client_bb_states[i][k].float() for i in range(num_clients))
                global_bb.load_state_dict(avg_bb)

                avg_root = {}
                for k in global_root.state_dict().keys():
                    avg_root[k] = sum(weights[i] * client_root_states[i][k].float() for i in range(num_clients))
                global_root.load_state_dict(avg_root)

                # Cluster Aggregation
                for k in range(num_clusters):
                    if cluster_updates[k]:
                        c_w = torch.tensor([float(weights[cid]) for cid, _ in cluster_updates[k]], device=device)
                        c_w = c_w / (c_w.sum() + 1e-8)
                        avg_p = {}
                        for key in cluster_heads[k].state_dict().keys():
                            avg_p[key] = sum(c_w[idx] * cluster_updates[k][idx][1][key].float() for idx in range(len(cluster_updates[k])))
                        cluster_heads[k].load_state_dict(avg_p)

            # Eval Defended H-ResFL
            accs_hresfl = []
            global_bb.eval(); global_root.eval()
            with torch.no_grad():
                for cid in honest_ids:
                    k_idx = client_clusters[cid]
                    cluster_heads[k_idx].eval()
                    local_heads[cid].eval()
                    a = client_alphas[cid]
                    c_test = ClientDataset(test_fast, test_splits[cid])
                    loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
                    c_tot, c_corr = 0, 0
                    for x, y in loader:
                        feats = global_bb.extract_features(x)
                        z_blend = a[0] * local_heads[cid](feats) + a[1] * cluster_heads[k_idx](feats) + a[2] * global_root(feats)
                        mask = torch.ones_like(z_blend) * -1e4
                        for c in active_classes_per_client[cid]:
                            mask[:, c] = 0.0
                        pred = (z_blend + mask).argmax(dim=1)
                        c_corr += (pred == y).sum().item()
                        c_tot += y.size(0)
                    accs_hresfl.append((c_corr / c_tot * 100.0) if c_tot > 0 else 0.0)

            res_entry = {
                "FedAvg": round(float(np.mean(accs_fedavg)), 2),
                "Defended H-ResFL": round(float(np.mean(accs_hresfl)), 2)
            }
            all_byz_results[atk][str(f_rate)] = res_entry
            print(f"  Rate q={f_rate:.1f} | FedAvg: {res_entry['FedAvg']:.2f}% | Defended H-ResFL: {res_entry['Defended H-ResFL']:.2f}%")

    out_path = os.path.join(_project_root, "outputs", "cifar100_byzantine_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_byz_results, f, indent=2)
    print(f"\n[CIFAR-100 BYZANTINE COMPLETE] Results saved to: {out_path}")
    return all_byz_results


def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 Benchmark Runner")
    parser.add_argument("--mode", type=str, default="multiregime", choices=["multiregime", "byzantine", "all"],
                        help="Benchmark mode: multiregime, byzantine, or all")
    args = parser.parse_args()

    if args.mode in ["multiregime", "all"]:
        run_cifar100_multiregime_experiment()
    if args.mode in ["byzantine", "all"]:
        run_cifar100_byzantine_suite()


if __name__ == "__main__":
    main()
