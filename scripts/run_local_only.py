"""
Fast Local-Only (Standalone) Baseline Evaluator (Runs on DirectML GPU with foreach=False).
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.model import ResNet9
from src.data.dataset import get_cifar10, partition_data, ClientDataset, get_fast_dataloader
from src.experiments.builder import detect_device


def evaluate_fast_local_only():
    device = detect_device()
    print(f"Target Hardware Device: {device}")

    train_dataset, test_dataset = get_cifar10(data_dir="./data", train_subset=10000, test_subset=3000, seed=42)

    scenarios = [
        ("IID (alpha=inf)", False, 0.0),
        ("Mild (alpha=1.0)", True, 1.0),
        ("Moderate (alpha=0.5)", True, 0.5),
        ("Severe (alpha=0.1)", True, 0.1),
        ("Extreme (alpha=0.05)", True, 0.05),
    ]

    results = {}

    print("\n" + "=" * 75)
    print("STANDALONE LOCAL-ONLY BENCHMARK ACROSS 5 DIRICHLET SCENARIOS")
    print("=" * 75)

    for sc_name, non_iid, alpha in scenarios:
        train_splits = partition_data(train_dataset, num_clients=15, non_iid=non_iid, alpha=alpha, seed=42)
        test_splits = partition_data(test_dataset, num_clients=15, non_iid=non_iid, alpha=alpha, seed=42)

        client_accs = []
        for cid in range(15):
            c_train = ClientDataset(train_dataset, train_splits[cid])
            c_test = ClientDataset(test_dataset, test_splits[cid])

            train_loader = get_fast_dataloader(c_train, batch_size=64, shuffle=True)
            test_loader = get_fast_dataloader(c_test, batch_size=64, shuffle=False)

            model = ResNet9(in_channels=3, num_classes=10).to(device)
            # Use foreach=False to prevent DML CPU fallback warning
            opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, foreach=False)
            crit = nn.CrossEntropyLoss()

            for _ in range(15):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad(set_to_none=True)
                    out = model(x)
                    loss = crit(out, y)
                    loss.backward()
                    opt.step()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)

            acc = (correct / total) * 100.0 if total > 0 else 0.0
            client_accs.append(acc)

        mean_acc = float(np.mean(client_accs))
        sorted_accs = sorted(client_accs)
        bottom10 = float(np.mean(sorted_accs[:2]))
        std_acc = float(np.std(client_accs))

        results[sc_name] = {
            "mean_accuracy": round(mean_acc, 2),
            "bottom_10_accuracy": round(bottom10, 2),
            "std_accuracy": round(std_acc, 2),
        }
        print(f"  {sc_name:<22} | Mean: {mean_acc:5.2f}% | Bottom 10% (Worst-Case): {bottom10:5.2f}% | Std: {std_acc:4.2f}%")

    out_path = os.path.join(_project_root, "outputs", "local_only_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLocal-Only results saved to: {out_path}")
    return results


if __name__ == "__main__":
    evaluate_fast_local_only()
