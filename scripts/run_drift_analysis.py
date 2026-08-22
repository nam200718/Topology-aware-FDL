"""
Backbone Representation Drift and Alignment Analysis.
Evaluates Linear CKA (Centered Kernel Alignment) and Cosine Similarity of penultimate
feature representations h = f_theta(x) across clients on a shared probe dataset,
comparing FedAvg, FedPer (naive split-head), FedRep, FedBABU, Ditto, and HEP.
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


def linear_cka(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """
    Computes Linear Centered Kernel Alignment (CKA) between two feature matrices (N x D).
    """
    x = features_x - features_x.mean(dim=0, keepdim=True)
    y = features_y - features_y.mean(dim=0, keepdim=True)
    
    dot_xy = torch.norm(torch.mm(x.t(), y)) ** 2
    dot_xx = torch.norm(torch.mm(x.t(), x)) ** 2
    dot_yy = torch.norm(torch.mm(y.t(), y)) ** 2
    
    denom = torch.sqrt(dot_xx * dot_yy) + 1e-10
    return float((dot_xy / denom).item())


def mean_cross_client_cka(feature_list: list[torch.Tensor]) -> float:
    """Computes average pairwise CKA across all distinct pairs of clients."""
    num_clients = len(feature_list)
    cka_vals = []
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            cka_vals.append(linear_cka(feature_list[i], feature_list[j]))
    return float(np.mean(cka_vals)) if cka_vals else 1.0


def mean_cross_client_cosine(feature_list: list[torch.Tensor]) -> float:
    """Computes average pairwise cosine similarity of mean representation vectors."""
    num_clients = len(feature_list)
    mean_vecs = [f.mean(dim=0) for f in feature_list]
    cos_vals = []
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            v_i = F.normalize(mean_vecs[i], p=2, dim=0)
            v_j = F.normalize(mean_vecs[j], p=2, dim=0)
            cos_vals.append(float(torch.dot(v_i, v_j).item()))
    return float(np.mean(cos_vals)) if cos_vals else 1.0


def run_drift_experiment(num_clients: int = 15, num_rounds: int = 20, batch_size: int = 64, alpha: float = 0.05):
    device = detect_device()
    print(f"Target Device: {device}")
    crit = nn.CrossEntropyLoss()

    train_raw, test_raw = get_cifar10(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)

    # Shared probe dataset (500 samples) to evaluate feature geometry across all client backbones
    probe_x = test_fast.images[:500]
    probe_y = test_fast.labels[:500]


    train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=True, alpha=alpha, seed=42)
    test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=True, alpha=alpha, seed=42)

    results = {}

    # -------------------------------------------------------------
    # 1. FedAvg
    # -------------------------------------------------------------
    print("\n[1/6] Evaluating FedAvg Representation Alignment...")
    global_model = ResNet9(in_channels=3, num_classes=10).to(device)
    fedavg_cka_trajectory = []
    fedavg_cos_trajectory = []

    for r in range(num_rounds):
        client_states = []
        client_features = []
        for cid in range(num_clients):
            c_train = ClientDataset(train_fast, train_splits[cid])
            loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
            local_m = ResNet9(in_channels=3, num_classes=10).to(device)
            local_m.load_state_dict(global_model.state_dict())
            opt = torch.optim.SGD(local_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

            local_m.train()
            for _ in range(3):
                for x, y in loader:
                    opt.zero_grad(set_to_none=True)
                    loss = crit(local_m(x), y)
                    loss.backward()
                    opt.step()

            local_m.eval()
            with torch.no_grad():
                feats = local_m.extract_features(probe_x)
            client_features.append(feats)
            client_states.append(local_m.state_dict())

        # Measure alignment at round r
        r_cka = mean_cross_client_cka(client_features)
        r_cos = mean_cross_client_cosine(client_features)
        fedavg_cka_trajectory.append(r_cka)
        fedavg_cos_trajectory.append(r_cos)

        avg_state = {}
        for k in global_model.state_dict().keys():
            avg_state[k] = torch.stack([client_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
        global_model.load_state_dict(avg_state)

    results["FedAvg"] = {
        "final_cka": round(fedavg_cka_trajectory[-1], 4),
        "final_cos": round(fedavg_cos_trajectory[-1], 4),
        "cka_trajectory": [round(v, 4) for v in fedavg_cka_trajectory[::4]],
    }
    print(f"FedAvg: Final Linear CKA = {results['FedAvg']['final_cka']:.4f}, Cosine = {results['FedAvg']['final_cos']:.4f}")

    # -------------------------------------------------------------
    # 2. FedPer (Naive Split-Head: Body shared, Head isolated)
    # -------------------------------------------------------------
    print("\n[2/6] Evaluating FedPer (Naive Split-Head)...")
    global_body = ResNet9(in_channels=3, num_classes=10).to(device)
    local_heads = [nn.Linear(256, 10).to(device) for _ in range(num_clients)]
    fedper_cka_trajectory = []
    fedper_cos_trajectory = []

    for r in range(num_rounds):
        body_states = []
        client_features = []
        for cid in range(num_clients):
            c_train = ClientDataset(train_fast, train_splits[cid])
            loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
            local_body = ResNet9(in_channels=3, num_classes=10).to(device)
            local_body.load_state_dict(global_body.state_dict())
            head = local_heads[cid]
            params = list(local_body.parameters()) + list(head.parameters())
            opt = torch.optim.SGD(params, lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

            local_body.train()
            head.train()
            for _ in range(3):
                for x, y in loader:
                    opt.zero_grad(set_to_none=True)
                    f = local_body.extract_features(x)
                    loss = crit(head(f), y)
                    loss.backward()
                    opt.step()

            local_body.eval()
            with torch.no_grad():
                feats = local_body.extract_features(probe_x)
            client_features.append(feats)
            body_states.append(local_body.state_dict())

        r_cka = mean_cross_client_cka(client_features)
        r_cos = mean_cross_client_cosine(client_features)
        fedper_cka_trajectory.append(r_cka)
        fedper_cos_trajectory.append(r_cos)

        avg_body = {}
        for k in global_body.state_dict().keys():
            avg_body[k] = torch.stack([body_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
        global_body.load_state_dict(avg_body)

    results["FedPer"] = {
        "final_cka": round(fedper_cka_trajectory[-1], 4),
        "final_cos": round(fedper_cos_trajectory[-1], 4),
        "cka_trajectory": [round(v, 4) for v in fedper_cka_trajectory[::4]],
    }
    print(f"FedPer: Final Linear CKA = {results['FedPer']['final_cka']:.4f}, Cosine = {results['FedPer']['final_cos']:.4f}")

    # -------------------------------------------------------------
    # 3. FedRep (Decoupled Head & Body updates)
    # -------------------------------------------------------------
    print("\n[3/6] Evaluating FedRep...")
    global_body = ResNet9(in_channels=3, num_classes=10).to(device)
    local_heads = [nn.Linear(256, 10).to(device) for _ in range(num_clients)]
    head_opts = [torch.optim.SGD(h.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False) for h in local_heads]
    fedrep_cka_trajectory = []
    fedrep_cos_trajectory = []

    for r in range(num_rounds):
        body_states = []
        client_features = []
        for cid in range(num_clients):
            c_train = ClientDataset(train_fast, train_splits[cid])
            loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
            local_body = ResNet9(in_channels=3, num_classes=10).to(device)
            local_body.load_state_dict(global_body.state_dict())
            head = local_heads[cid]
            h_opt = head_opts[cid]

            # Head update (freeze body)
            for p in local_body.parameters(): p.requires_grad = False
            head.train()
            for _ in range(3):
                for x, y in loader:
                    h_opt.zero_grad(set_to_none=True)
                    f = local_body.extract_features(x)
                    loss = crit(head(f), y)
                    loss.backward()
                    h_opt.step()

            # Body update (freeze head)
            for p in local_body.parameters(): p.requires_grad = True
            for p in head.parameters(): p.requires_grad = False
            b_opt = torch.optim.SGD(local_body.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
            local_body.train()
            for _ in range(1):
                for x, y in loader:
                    b_opt.zero_grad(set_to_none=True)
                    f = local_body.extract_features(x)
                    loss = crit(head(f), y)
                    loss.backward()
                    b_opt.step()

            for p in head.parameters(): p.requires_grad = True

            local_body.eval()
            with torch.no_grad():
                feats = local_body.extract_features(probe_x)
            client_features.append(feats)
            body_states.append(local_body.state_dict())

        r_cka = mean_cross_client_cka(client_features)
        r_cos = mean_cross_client_cosine(client_features)
        fedrep_cka_trajectory.append(r_cka)
        fedrep_cos_trajectory.append(r_cos)

        avg_body = {}
        for k in global_body.state_dict().keys():
            avg_body[k] = torch.stack([body_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
        global_body.load_state_dict(avg_body)

    results["FedRep"] = {
        "final_cka": round(fedrep_cka_trajectory[-1], 4),
        "final_cos": round(fedrep_cos_trajectory[-1], 4),
        "cka_trajectory": [round(v, 4) for v in fedrep_cka_trajectory[::4]],
    }
    print(f"FedRep: Final Linear CKA = {results['FedRep']['final_cka']:.4f}, Cosine = {results['FedRep']['final_cos']:.4f}")

    # -------------------------------------------------------------
    # 4. FedBABU (Frozen Body Head Tuning)
    # -------------------------------------------------------------
    print("\n[4/6] Evaluating FedBABU...")
    global_model = ResNet9(in_channels=3, num_classes=10).to(device)
    fedbabu_cka_trajectory = []
    fedbabu_cos_trajectory = []

    for r in range(num_rounds):
        client_states = []
        client_features = []
        for cid in range(num_clients):
            c_train = ClientDataset(train_fast, train_splits[cid])
            loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
            local_m = ResNet9(in_channels=3, num_classes=10).to(device)
            local_m.load_state_dict(global_model.state_dict())
            for param in local_m.fc2.parameters():
                param.requires_grad = False
            opt = torch.optim.SGD([p for p in local_m.parameters() if p.requires_grad], lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

            local_m.train()
            for _ in range(3):
                for x, y in loader:
                    opt.zero_grad(set_to_none=True)
                    loss = crit(local_m(x), y)
                    loss.backward()
                    opt.step()

            local_m.eval()
            with torch.no_grad():
                feats = local_m.extract_features(probe_x)
            client_features.append(feats)
            client_states.append(local_m.state_dict())

        r_cka = mean_cross_client_cka(client_features)
        r_cos = mean_cross_client_cosine(client_features)
        fedbabu_cka_trajectory.append(r_cka)
        fedbabu_cos_trajectory.append(r_cos)

        avg_state = {}
        for k in global_model.state_dict().keys():
            avg_state[k] = torch.stack([client_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
        global_model.load_state_dict(avg_state)

    results["FedBABU"] = {
        "final_cka": round(fedbabu_cka_trajectory[-1], 4),
        "final_cos": round(fedbabu_cos_trajectory[-1], 4),
        "cka_trajectory": [round(v, 4) for v in fedbabu_cka_trajectory[::4]],
    }
    print(f"FedBABU: Final Linear CKA = {results['FedBABU']['final_cka']:.4f}, Cosine = {results['FedBABU']['final_cos']:.4f}")

    # -------------------------------------------------------------
    # 5. Ditto (Dual-Model)
    # -------------------------------------------------------------
    print("\n[5/6] Evaluating Ditto...")
    global_model = ResNet9(in_channels=3, num_classes=10).to(device)
    pers_models = [ResNet9(in_channels=3, num_classes=10).to(device) for _ in range(num_clients)]
    for m in pers_models: m.load_state_dict(global_model.state_dict())
    ditto_cka_trajectory = []
    ditto_cos_trajectory = []

    for r in range(num_rounds):
        client_states = []
        client_features = []
        for cid in range(num_clients):
            c_train = ClientDataset(train_fast, train_splits[cid])
            loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
            local_g = ResNet9(in_channels=3, num_classes=10).to(device)
            local_g.load_state_dict(global_model.state_dict())
            opt_g = torch.optim.SGD(local_g.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

            # Global update
            local_g.train()
            for _ in range(3):
                for x, y in loader:
                    opt_g.zero_grad(set_to_none=True)
                    loss = crit(local_g(x), y)
                    loss.backward()
                    opt_g.step()

            # Personalized update w/ proximal penalty
            pers_m = pers_models[cid]
            opt_p = torch.optim.SGD(pers_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
            pers_m.train()
            for _ in range(3):
                for x, y in loader:
                    opt_p.zero_grad(set_to_none=True)
                    loss_ce = crit(pers_m(x), y)
                    loss_prox = 0.0
                    for p_p, p_g in zip(pers_m.parameters(), local_g.parameters()):
                        loss_prox += 0.5 * 0.1 * torch.norm(p_p - p_g.detach()) ** 2
                    loss = loss_ce + loss_prox
                    loss.backward()
                    opt_p.step()

            pers_m.eval()
            with torch.no_grad():
                feats = pers_m.extract_features(probe_x)
            client_features.append(feats)
            client_states.append(local_g.state_dict())

        r_cka = mean_cross_client_cka(client_features)
        r_cos = mean_cross_client_cosine(client_features)
        ditto_cka_trajectory.append(r_cka)
        ditto_cos_trajectory.append(r_cos)

        avg_state = {}
        for k in global_model.state_dict().keys():
            avg_state[k] = torch.stack([client_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
        global_model.load_state_dict(avg_state)

    results["Ditto"] = {
        "final_cka": round(ditto_cka_trajectory[-1], 4),
        "final_cos": round(ditto_cos_trajectory[-1], 4),
        "cka_trajectory": [round(v, 4) for v in ditto_cka_trajectory[::4]],
    }
    print(f"Ditto: Final Linear CKA = {results['Ditto']['final_cka']:.4f}, Cosine = {results['Ditto']['final_cos']:.4f}")

    # -------------------------------------------------------------
    # 6. HEP (MultiHead with Entropy Gated Budgets and Root Anchor)
    # -------------------------------------------------------------
    print("\n[6/6] Evaluating HEP (Proposed Framework)...")
    global_model = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
    cluster_heads = [nn.Linear(256, 10).to(device) for _ in range(5)]
    local_heads = [nn.Linear(256, 10).to(device) for _ in range(num_clients)]
    
    # Compute client label entropy R_skew
    client_r_skew = []
    for cid in range(num_clients):
        c_train = ClientDataset(train_fast, train_splits[cid])
        y_targets = [y for _, y in c_train]
        counts = torch.bincount(torch.tensor(y_targets, device=device), minlength=10).float()
        probs = counts / counts.sum()
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        r_skew = entropy / np.log(10)
        client_r_skew.append(r_skew)

    hep_cka_trajectory = []
    hep_cos_trajectory = []

    for r in range(num_rounds):
        client_states = []
        client_features = []
        for cid in range(num_clients):
            c_train = ClientDataset(train_fast, train_splits[cid])
            loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
            local_m = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
            local_m.load_state_dict(global_model.state_dict())
            local_m.fc2_parent.load_state_dict(cluster_heads[cid % 5].state_dict())
            local_m.fc2_local.load_state_dict(local_heads[cid].state_dict())

            r_sk = client_r_skew[cid]
            if r_sk > 0.70: e_r, e_p, e_l = 5, 3, 2
            elif r_sk >= 0.30: e_r, e_p, e_l = 4, 3, 3
            else: e_r, e_p, e_l = 2, 3, 5

            opt = torch.optim.SGD(local_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
            local_m.train()
            max_e = max(e_r, e_p, e_l)
            for epoch in range(1, max_e + 1):
                for x, y in loader:
                    opt.zero_grad(set_to_none=True)
                    f = local_m.extract_features(x)
                    loss = 0.0
                    if epoch <= e_r: loss += crit(local_m.fc2_root(f), y)
                    if epoch <= e_p: loss += crit(local_m.fc2_parent(f), y)
                    if epoch <= e_l: loss += crit(local_m.fc2_local(f), y)
                    loss.backward()
                    opt.step()

            local_heads[cid].load_state_dict(local_m.fc2_local.state_dict())
            local_m.eval()
            with torch.no_grad():
                feats = local_m.extract_features(probe_x)
            client_features.append(feats)
            client_states.append(local_m.state_dict())

        r_cka = mean_cross_client_cka(client_features)
        r_cos = mean_cross_client_cosine(client_features)
        hep_cka_trajectory.append(r_cka)
        hep_cos_trajectory.append(r_cos)

        avg_state = {}
        for k in global_model.state_dict().keys():
            if "fc2_parent" not in k and "fc2_local" not in k:
                avg_state[k] = torch.stack([client_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
            else:
                avg_state[k] = global_model.state_dict()[k]
        global_model.load_state_dict(avg_state)

    results["HEP"] = {
        "final_cka": round(hep_cka_trajectory[-1], 4),
        "final_cos": round(hep_cos_trajectory[-1], 4),
        "cka_trajectory": [round(v, 4) for v in hep_cka_trajectory[::4]],
    }
    print(f"HEP: Final Linear CKA = {results['HEP']['final_cka']:.4f}, Cosine = {results['HEP']['final_cos']:.4f}")

    out_file = os.path.join(_project_root, "outputs", "representation_drift_analysis.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRepresentation Drift & Alignment results saved to: {out_file}")
    return results


if __name__ == "__main__":
    run_drift_experiment()
