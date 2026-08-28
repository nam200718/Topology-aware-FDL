"""
Test Byzantine Robustness of HEP with Server-Side Coordinate-Wise Trimmed Mean (beta=0.20)
against Sign-Flipping, Label-Flipping, and Gaussian Noise attacks.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.model import MultiHeadResNet9
from src.core.loss import ClassFrequencyBalancedMaskedLoss
from src.data.dataset import get_cifar10, partition_data, ClientDataset, get_fast_dataloader, FastDataset
from src.experiments.builder import detect_device


def trimmed_mean(tensor_list, beta=0.20):
    """
    Computes coordinate-wise trimmed mean across client updates.
    Discards top and bottom beta fraction of values per coordinate.
    """
    stacked = torch.stack(tensor_list, dim=0) # [N, ...]
    N = stacked.size(0)
    k = int(N * beta)
    if k == 0 or 2 * k >= N:
        return stacked.mean(dim=0)
    sorted_t, _ = torch.sort(stacked, dim=0)
    trimmed = sorted_t[k:N-k]
    return trimmed.mean(dim=0)


def run_byzantine_benchmark(attack_type="sign_flip", byz_fraction=0.20, num_rounds=15, use_trimmed_mean=True):
    device = detect_device()
    num_clients = 15
    num_byz = int(num_clients * byz_fraction)

    train_raw, test_raw = get_cifar10(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)
    train_fast = FastDataset(train_raw, device=device)
    test_fast = FastDataset(test_raw, device=device)

    train_splits = partition_data(train_fast, num_clients=num_clients, non_iid=True, alpha=0.5, seed=42)
    test_splits = partition_data(test_fast, num_clients=num_clients, non_iid=True, alpha=0.5, seed=42)

    byz_clients = set(range(num_byz))

    global_model = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
    cluster_heads_w = [global_model.fc2_root.weight.data.clone() for _ in range(5)]
    cluster_heads_b = [global_model.fc2_root.bias.data.clone() for _ in range(5)]
    local_heads_w = [global_model.fc2_root.weight.data.clone() for _ in range(num_clients)]
    local_heads_b = [global_model.fc2_root.bias.data.clone() for _ in range(num_clients)]

    cluster_assignments = [i % 5 for i in range(num_clients)]
    loss_fn = ClassFrequencyBalancedMaskedLoss(gamma=0.5)

    for r in range(num_rounds):
        client_deltas_bb = []
        client_deltas_rw = []
        client_deltas_rb = []

        for cid in range(num_clients):
            is_attacker = cid in byz_clients
            c_train = ClientDataset(train_fast, train_splits[cid])
            loader = get_fast_dataloader(c_train, batch_size=32, shuffle=True)

            local_m = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
            local_m.load_state_dict(global_model.state_dict())
            k_assigned = cluster_assignments[cid]
            local_m.fc2_parent.weight.data.copy_(cluster_heads_w[k_assigned])
            local_m.fc2_parent.bias.data.copy_(cluster_heads_b[k_assigned])
            local_m.fc2_local.weight.data.copy_(local_heads_w[cid])
            local_m.fc2_local.bias.data.copy_(local_heads_b[cid])

            opt = torch.optim.SGD(local_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)

            local_m.train()
            for _ in range(3):
                for x, y in loader:
                    if is_attacker and attack_type == "label_flip":
                        y = 9 - y
                    opt.zero_grad(set_to_none=True)
                    zr, zp, zl = local_m(x, head="all")
                    loss = F.cross_entropy(zr, y) + 0.5 * F.cross_entropy(zp, y) + 0.5 * loss_fn(zl, y)
                    loss.backward()
                    opt.step()

            local_heads_w[cid].copy_(local_m.fc2_local.weight.data)
            local_heads_b[cid].copy_(local_m.fc2_local.bias.data)

            delta_bb = {k: (local_m.state_dict()[k] - global_model.state_dict()[k]).float() for k in global_model.state_dict() if 'fc2' not in k}
            d_rw = (local_m.fc2_root.weight.data - global_model.fc2_root.weight.data).float()
            d_rb = (local_m.fc2_root.bias.data - global_model.fc2_root.bias.data).float()

            if is_attacker:
                if attack_type == "sign_flip":
                    delta_bb = {k: -1.5 * v for k, v in delta_bb.items()}
                    d_rw = -1.5 * d_rw
                    d_rb = -1.5 * d_rb
                elif attack_type == "gaussian_noise":
                    delta_bb = {k: v + torch.randn_like(v) * 0.5 for k, v in delta_bb.items()}
                    d_rw = d_rw + torch.randn_like(d_rw) * 0.5
                    d_rb = d_rb + torch.randn_like(d_rb) * 0.5

            client_deltas_bb.append(delta_bb)
            client_deltas_rw.append(d_rw)
            client_deltas_rb.append(d_rb)

        # Server Aggregation: Trimmed Mean vs Mean
        avg_bb = {}
        for k in client_deltas_bb[0].keys():
            t_list = [client_deltas_bb[i][k] for i in range(num_clients)]
            agg_delta = trimmed_mean(t_list, beta=0.20) if use_trimmed_mean else torch.stack(t_list).mean(dim=0)
            avg_bb[k] = global_model.state_dict()[k] + agg_delta

        agg_rw = trimmed_mean(client_deltas_rw, beta=0.20) if use_trimmed_mean else torch.stack(client_deltas_rw).mean(dim=0)
        agg_rb = trimmed_mean(client_deltas_rb, beta=0.20) if use_trimmed_mean else torch.stack(client_deltas_rb).mean(dim=0)

        global_model.load_state_dict(avg_bb, strict=False)
        global_model.fc2_root.weight.data.copy_(global_model.fc2_root.weight.data + agg_rw)
        global_model.fc2_root.bias.data.copy_(global_model.fc2_root.bias.data + agg_rb)

    # Evaluate clean clients
    clean_accs = []
    global_model.eval()
    for cid in range(num_byz, num_clients):
        c_test = ClientDataset(test_fast, test_splits[cid])
        loader = get_fast_dataloader(c_test, batch_size=32, shuffle=False)
        local_m = MultiHeadResNet9(in_channels=3, num_classes=10).to(device)
        local_m.load_state_dict(global_model.state_dict())
        k_assigned = cluster_assignments[cid]
        local_m.fc2_parent.weight.data.copy_(cluster_heads_w[k_assigned])
        local_m.fc2_parent.bias.data.copy_(cluster_heads_b[k_assigned])
        local_m.fc2_local.weight.data.copy_(local_heads_w[cid])
        local_m.fc2_local.bias.data.copy_(local_heads_b[cid])

        c_total, c_corr = 0, 0
        with torch.no_grad():
            for x, y in loader:
                zr, zp, zl = local_m(x, head="all")
                z_blend = 0.5 * (zl / 0.6) + 0.3 * (zp / 0.8) + 0.2 * (zr / 1.0)
                pred = z_blend.argmax(dim=1)
                c_corr += (pred == y).sum().item()
                c_total += y.size(0)
        clean_accs.append((c_corr / c_total * 100.0) if c_total > 0 else 0.0)

    mean_acc = round(float(np.mean(clean_accs)), 2)
    return mean_acc


if __name__ == "__main__":
    attacks = ["label_flip", "sign_flip", "gaussian_noise"]
    fractions = [0.10, 0.20, 0.40]

    print("="*70)
    print("BYZANTINE ROBUSTNESS BENCHMARK WITH SERVER-SIDE TRIMMED MEAN")
    print("="*70)

    results = {}
    for atk in attacks:
        results[atk] = {}
        for f in fractions:
            acc_defended = run_byzantine_benchmark(attack_type=atk, byz_fraction=f, num_rounds=10, use_trimmed_mean=True)
            acc_undefended = run_byzantine_benchmark(attack_type=atk, byz_fraction=f, num_rounds=10, use_trimmed_mean=False)
            results[atk][f"f={int(f*100)}%"] = {
                "defended_trimmed_mean": acc_defended,
                "undefended_mean": acc_undefended
            }
            print(f"Attack [{atk:14s} | f={int(f*100)}%]: Trimmed Mean = {acc_defended:5.2f}% | Standard Mean = {acc_undefended:5.2f}%")

    out_file = os.path.join(_project_root, "outputs", "byzantine_trimmed_mean_benchmark.json")
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nRobustness results saved to: {out_file}")
