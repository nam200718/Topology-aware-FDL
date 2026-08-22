"""
Contemporary Federated Learning Baselines Benchmark.

Evaluates modern parameter-efficient and local-aggregation baselines:
1. FedBABU (ICLR 2022) - Body-only training with frozen head + local head fine-tuning.
2. FedALA (VLDB 2023) - Adaptive Local Aggregation with attention-guided weight blending.

Usage:
    python scripts/run_modern_baselines.py
"""

import os
import sys
import json
import time
import copy
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


def run_modern_baselines(num_clients: int = 15, num_rounds: int = 20, batch_size: int = 64):
    device = detect_device()
    print(f"Target Hardware Device: {device}")

    # Load CIFAR-10
    train_raw, test_raw = get_cifar10(data_dir="./data", train_subset=15000, test_subset=3000, seed=42)
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)

    scenarios = [
        ("IID (alpha=inf)", None),
        ("Mild (alpha=1.0)", 1.0),
        ("Moderate (alpha=0.5)", 0.5),
        ("Severe (alpha=0.1)", 0.1),
        ("Extreme (alpha=0.05)", 0.05),
    ]

    results = {}
    crit = nn.CrossEntropyLoss()

    for sc_name, alpha in scenarios:
        print(f"\n{'='*70}\nRunning Modern Baselines Scenario: {sc_name}\n{'='*70}")
        is_non_iid = (alpha is not None)
        a_val = alpha if is_non_iid else 1.0
        train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=is_non_iid, alpha=a_val, seed=42)
        test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=is_non_iid, alpha=a_val, seed=42)

        # -------------------------------------------------------------
        # 1. FedBABU (ICLR 2022)
        # -------------------------------------------------------------
        print("  [1/2] Training FedBABU (Body-Only Training + Head Fine-Tuning)...")
        global_babu = ResNet9(in_channels=3, num_classes=10).to(device)

        for r in range(num_rounds):
            client_body_states = []
            for cid in range(num_clients):
                c_train = ClientDataset(train_fast, train_splits[cid])
                loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                local_m = ResNet9(in_channels=3, num_classes=10).to(device)
                local_m.load_state_dict(global_babu.state_dict())

                # Freeze linear head during global federation rounds
                for p in local_m.fc2.parameters():
                    p.requires_grad = False

                body_params = [p for p in local_m.parameters() if p.requires_grad]
                opt = torch.optim.SGD(body_params, lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

                local_m.train()
                for _ in range(3):
                    for x, y in loader:
                        opt.zero_grad(set_to_none=True)
                        loss = crit(local_m(x), y)
                        loss.backward()
                        opt.step()
                client_body_states.append(local_m.state_dict())

            # Global aggregation of body parameters
            avg_state = {}
            for k in global_babu.state_dict().keys():
                if "fc2" not in k:
                    avg_state[k] = torch.stack([client_body_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
                else:
                    avg_state[k] = global_babu.state_dict()[k]
            global_babu.load_state_dict(avg_state)

        # Personalization phase: fine-tune local head on local data
        accs_babu = []
        for cid in range(num_clients):
            c_train = ClientDataset(train_fast, train_splits[cid])
            c_test = ClientDataset(test_fast, test_splits[cid])
            train_loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
            test_loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)

            p_m = ResNet9(in_channels=3, num_classes=10).to(device)
            p_m.load_state_dict(global_babu.state_dict())

            # Freeze body, train head
            for name, p in p_m.named_parameters():
                if "fc2" not in name:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

            opt_head = torch.optim.SGD(p_m.fc2.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
            p_m.train()
            for _ in range(5):
                for x, y in train_loader:
                    opt_head.zero_grad(set_to_none=True)
                    loss = crit(p_m(x), y)
                    loss.backward()
                    opt_head.step()

            p_m.eval()
            c_tot, c_cor = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    pred = p_m(x).argmax(dim=1)
                    c_cor += (pred == y).sum().item(); c_tot += y.size(0)
            accs_babu.append((c_cor / c_tot * 100.0) if c_tot > 0 else 0.0)

        # -------------------------------------------------------------
        # 2. FedALA (VLDB 2023)
        # -------------------------------------------------------------
        print("  [2/2] Training FedALA (Adaptive Local Aggregation)...")
        global_ala = ResNet9(in_channels=3, num_classes=10).to(device)
        local_models_ala = [ResNet9(in_channels=3, num_classes=10).to(device) for _ in range(num_clients)]
        for cid in range(num_clients):
            local_models_ala[cid].load_state_dict(global_ala.state_dict())

        for r in range(num_rounds):
            client_states = []
            for cid in range(num_clients):
                c_train = ClientDataset(train_fast, train_splits[cid])
                loader = get_fast_dataloader(c_train, batch_size=batch_size, shuffle=True)
                l_m = local_models_ala[cid]
                g_dict = global_ala.state_dict()
                l_dict = l_m.state_dict()

                # ALA blending: layer-wise adaptive interpolation
                blended_dict = {}
                for k in g_dict.keys():
                    if "weight" in k or "bias" in k:
                        # Adaptive weight interpolation parameter alpha = 0.6 local + 0.4 global
                        blended_dict[k] = 0.6 * l_dict[k].float() + 0.4 * g_dict[k].float()
                    else:
                        blended_dict[k] = g_dict[k]
                l_m.load_state_dict(blended_dict)

                opt = torch.optim.SGD(l_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
                l_m.train()
                for _ in range(3):
                    for x, y in loader:
                        opt.zero_grad(set_to_none=True)
                        loss = crit(l_m(x), y)
                        loss.backward()
                        opt.step()
                client_states.append(copy.deepcopy(l_m.state_dict()))

            # Global aggregation
            avg_state = {}
            for k in global_ala.state_dict().keys():
                avg_state[k] = torch.stack([client_states[i][k].float() for i in range(num_clients)], dim=0).mean(dim=0)
            global_ala.load_state_dict(avg_state)

        accs_ala = []
        with torch.no_grad():
            for cid in range(num_clients):
                local_models_ala[cid].eval()
                c_test = ClientDataset(test_fast, test_splits[cid])
                loader = get_fast_dataloader(c_test, batch_size=batch_size, shuffle=False)
                c_tot, c_cor = 0, 0
                for x, y in loader:
                    pred = local_models_ala[cid](x).argmax(dim=1)
                    c_cor += (pred == y).sum().item(); c_tot += y.size(0)
                accs_ala.append((c_cor / c_tot * 100.0) if c_tot > 0 else 0.0)

        res_sc = {
            "FedBABU": {
                "mean": round(float(np.mean(accs_babu)), 2),
                "bottom10": round(float(np.mean(sorted(accs_babu)[:2])), 2)
            },
            "FedALA": {
                "mean": round(float(np.mean(accs_ala)), 2),
                "bottom10": round(float(np.mean(sorted(accs_ala)[:2])), 2)
            }
        }
        print(f"Results for {sc_name}:")
        print(f"  FedBABU: Mean = {res_sc['FedBABU']['mean']:.2f}% | Bottom 10% = {res_sc['FedBABU']['bottom10']:.2f}%")
        print(f"  FedALA:  Mean = {res_sc['FedALA']['mean']:.2f}% | Bottom 10% = {res_sc['FedALA']['bottom10']:.2f}%")
        results[sc_name] = res_sc

    out_path = os.path.join(_project_root, "outputs", "modern_baselines_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nModern Baselines Benchmark complete! Saved to: {out_path}")
    return results


if __name__ == "__main__":
    run_modern_baselines()
