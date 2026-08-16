"""
Advanced Multi-Attack Byzantine Robustness Benchmark.

Evaluates FedAvg, Ditto, and HEP against:
1. Deterministic Label Inversion Attack (y -> C - 1 - y)
2. Gradient Sign-Flipping Attack (Delta -> -2.0 * Delta)
3. Gaussian Noise Model Poisoning (Delta ~ N(0, sigma^2))

across attacker fractions f in {0%, 10%, 20%, 30%, 40%}.

Usage:
    python scripts/run_multi_attack_byzantine.py
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


def run_multi_attack_byzantine_suite(num_clients: int = 15, num_rounds: int = 15, batch_size: int = 64):
    device = detect_device()
    print(f"Target Hardware Device: {device}")

    train_raw, test_raw = get_cifar10(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)

    train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=True, alpha=0.5, seed=42)
    test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=True, alpha=0.5, seed=42)

    attacker_rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    attacks = ["label_flipping", "sign_flipping", "gaussian_noise"]

    all_results = {atk: {} for atk in attacks}
    crit = nn.CrossEntropyLoss()

    print(f"\n{'='*70}\nRunning Advanced Multi-Attack Byzantine Benchmark\n{'='*70}")

    for atk in attacks:
        print(f"\nEvaluating Attack Type: {atk.upper()}")
        all_results[atk] = {}

        for f_rate in attacker_rates:
            num_attackers = int(np.round(num_clients * f_rate))
            attacker_ids = set(range(num_attackers))
            honest_ids = [i for i in range(num_clients) if i not in attacker_ids]

            # ---------------------------------------------------------
            # 1. FedAvg
            # ---------------------------------------------------------
            global_model = ResNet9(in_channels=3, num_classes=10).to(device)

            for r in range(num_rounds):
                client_states = []
                for cid in range(num_clients):
                    c_train = ClientDataset(train_fast, train_splits[cid])
                    loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                    local_m = ResNet9(in_channels=3, num_classes=10).to(device)
                    local_m.load_state_dict(global_model.state_dict())
                    opt = torch.optim.SGD(local_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

                    is_att = cid in attacker_ids
                    local_m.train()
                    for _ in range(2):
                        for x, y in loader:
                            opt.zero_grad(set_to_none=True)
                            if is_att and atk == "label_flipping":
                                y = 9 - y
                            loss = crit(local_m(x), y)
                            loss.backward()
                            opt.step()

                    # Post-training poisoning for sign flipping / noise
                    state = local_m.state_dict()
                    if is_att:
                        if atk == "sign_flipping":
                            for k in state.keys():
                                delta = state[k] - global_model.state_dict()[k]
                                state[k] = global_model.state_dict()[k] - 2.0 * delta
                        elif atk == "gaussian_noise":
                            for k in state.keys():
                                if state[k].is_floating_point():
                                    state[k] = state[k] + torch.randn_like(state[k]) * 0.5

                    client_states.append(state)

                avg_state = {}
                for k in global_model.state_dict().keys():
                    avg_state[k] = torch.stack([client_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
                global_model.load_state_dict(avg_state)

            # Evaluate on honest clients only
            accs_fedavg = []
            global_model.eval()
            with torch.no_grad():
                for cid in honest_ids:
                    c_test = ClientDataset(test_fast, test_splits[cid])
                    loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
                    c_total, c_corr = 0, 0
                    for x, y in loader:
                        pred = global_model(x).argmax(dim=1)
                        c_corr += (pred == y).sum().item()
                        c_total += y.size(0)
                    accs_fedavg.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

            # ---------------------------------------------------------
            # 2. Ditto
            # ---------------------------------------------------------
            global_m = ResNet9(in_channels=3, num_classes=10).to(device)
            local_models = [ResNet9(in_channels=3, num_classes=10).to(device) for _ in range(num_clients)]

            for r in range(num_rounds):
                client_states = []
                for cid in range(num_clients):
                    c_train = ClientDataset(train_fast, train_splits[cid])
                    loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                    is_att = cid in attacker_ids

                    l_glob = ResNet9(in_channels=3, num_classes=10).to(device)
                    l_glob.load_state_dict(global_m.state_dict())
                    opt_g = torch.optim.SGD(l_glob.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

                    l_glob.train()
                    for _ in range(2):
                        for x, y in loader:
                            opt_g.zero_grad(set_to_none=True)
                            if is_att and atk == "label_flipping": y = 9 - y
                            loss = crit(l_glob(x), y)
                            loss.backward()
                            opt_g.step()

                    state_g = l_glob.state_dict()
                    if is_att:
                        if atk == "sign_flipping":
                            for k in state_g.keys():
                                delta = state_g[k] - global_m.state_dict()[k]
                                state_g[k] = global_m.state_dict()[k] - 2.0 * delta
                        elif atk == "gaussian_noise":
                            for k in state_g.keys():
                                if state_g[k].is_floating_point():
                                    state_g[k] = state_g[k] + torch.randn_like(state_g[k]) * 0.5
                    client_states.append(state_g)

                    if not is_att:
                        p_mod = local_models[cid]
                        opt_p = torch.optim.SGD(p_mod.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
                        w_g_vec = torch.nn.utils.parameters_to_vector(l_glob.parameters()).detach()
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
                    avg_state[k] = torch.stack([client_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
                global_m.load_state_dict(avg_state)

            accs_ditto = []
            with torch.no_grad():
                for cid in honest_ids:
                    local_models[cid].eval()
                    c_test = ClientDataset(test_fast, test_splits[cid])
                    loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
                    c_total, c_corr = 0, 0
                    for x, y in loader:
                        pred = local_models[cid](x).argmax(dim=1)
                        c_corr += (pred == y).sum().item()
                        c_total += y.size(0)
                    accs_ditto.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

            # ---------------------------------------------------------
            # 3. HEP (Ours)
            # ---------------------------------------------------------
            num_clusters = 3
            global_backbone = ResNet9(in_channels=3, num_classes=10).to(device)
            global_root_head = nn.Linear(256, 10).to(device)
            cluster_heads = [nn.Linear(256, 10).to(device) for _ in range(num_clusters)]
            local_heads = [nn.Linear(256, 10).to(device) for _ in range(num_clients)]
            client_alphas = [torch.tensor([0.33, 0.33, 0.34], device=device) for _ in range(num_clients)]
            client_clusters = [i % num_clusters for i in range(num_clients)]

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
                cluster_updates = {k: [] for k in range(num_clusters)}

                for cid in range(num_clients):
                    c_train = ClientDataset(train_fast, train_splits[cid])
                    loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                    k_idx = client_clusters[cid]
                    is_att = cid in attacker_ids

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
                            if is_att and atk == "label_flipping": y = 9 - y
                            feats = l_bb.extract_features(x)
                            loss = 0.0
                            if ep <= e_r: loss += crit(l_root(feats), y)
                            if ep <= e_p: loss += crit(l_parent(feats), y)
                            if ep <= e_l: loss += crit(l_local(feats), y)
                            loss.backward()
                            opt.step()

                    st_bb = l_bb.state_dict()
                    st_rt = l_root.state_dict()
                    if is_att:
                        if atk == "sign_flipping":
                            for k in st_bb.keys():
                                delta = st_bb[k] - global_backbone.state_dict()[k]
                                st_bb[k] = global_backbone.state_dict()[k] - 2.0 * delta
                        elif atk == "gaussian_noise":
                            for k in st_bb.keys():
                                if st_bb[k].is_floating_point():
                                    st_bb[k] = st_bb[k] + torch.randn_like(st_bb[k]) * 0.5

                    client_bb_states.append(st_bb)
                    client_root_states.append(st_rt)
                    cluster_updates[k_idx].append(l_parent.state_dict())

                    if not is_att:
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

                for k in range(num_clusters):
                    if cluster_updates[k]:
                        avg_p = {}
                        for key in cluster_heads[k].state_dict().keys():
                            avg_p[key] = torch.stack([cluster_updates[k][i][key].float() for i in range(len(cluster_updates[k]))], dim=0).mean(dim=0)
                        cluster_heads[k].load_state_dict(avg_p)

            # Eval HEP
            accs_hep = []
            global_backbone.eval(); global_root_head.eval()
            with torch.no_grad():
                for cid in honest_ids:
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
                    accs_hep.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

            res_entry = {
                "FedAvg": round(float(np.mean(accs_fedavg)), 2),
                "Ditto": round(float(np.mean(accs_ditto)), 2),
                "HEP (Ours)": round(float(np.mean(accs_hep)), 2),
            }
            all_results[atk][f"{int(f_rate*100)}%"] = res_entry
            print(f"  [{atk.upper()}] f={int(f_rate*100)}% | FedAvg: {res_entry['FedAvg']:.2f}% | Ditto: {res_entry['Ditto']:.2f}% | HEP: {res_entry['HEP (Ours)']:.2f}%")

    out_path = os.path.join(_project_root, "outputs", "byzantine_multi_attack_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMulti-attack Byzantine evaluation completed! Saved to: {out_path}")
    return all_results


if __name__ == "__main__":
    run_multi_attack_byzantine_suite()
