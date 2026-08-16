"""
Master Orchestrator for All Paper Strengthening Experiments.

Sequentially executes:
1. CIFAR-100 High-Class-Cardinality Benchmark (C = 100)
2. 50-Client Population Scaling with Partial Participation (N=50, Cp=0.2)
3. Cluster Count Sensitivity Sweep (K in {1, 2, 3, 5, 8})
4. Advanced Multi-Attack Byzantine Suite (Label-flipping, Sign-flipping, Gaussian Noise)
5. Head-Epoch Budget Ratio Ablation (5:3:2 vs 3:3:3 vs 2:3:5 vs 5:0:0)

Usage:
    python scripts/run_all_paper_experiments.py
"""

import os
import sys
import time
import json

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.run_cifar100_benchmark import run_cifar100_experiment
from scripts.run_scale_50clients import run_50clients_scaling
from scripts.run_cluster_k_sensitivity import run_k_sensitivity
from scripts.run_multi_attack_byzantine import run_multi_attack_byzantine_suite
from scripts.run_epoch_budget_ablation import run_budget_ablation


def main():
    t_start = time.time()
    print("=" * 80)
    print("STARTING FULL EXPERIMENTAL STRENGTHENING SUITE FOR RESEARCH PAPER")
    print("=" * 80)

    # 1. CIFAR-100 Benchmark
    c100_path = os.path.join(_project_root, "outputs", "cifar100_results.json")
    if os.path.exists(c100_path):
        print("\n>>> STEP 1/5: CIFAR-100 results already exist, loading from disk...")
        with open(c100_path) as f: cifar100_res = json.load(f)
    else:
        print("\n>>> STEP 1/5: Running CIFAR-100 Benchmark (C = 100)...")
        t0 = time.time()
        cifar100_res = run_cifar100_experiment(num_clients=15, num_rounds=20, batch_size=64)
        print(f"--- Step 1 finished in {time.time() - t0:.1f}s ---")

    # 2. 50-Client Scaling
    scale_path = os.path.join(_project_root, "outputs", "scale_50clients_results.json")
    if os.path.exists(scale_path):
        print("\n>>> STEP 2/5: 50-Client Scalability results already exist, loading from disk...")
        with open(scale_path) as f: scale_res = json.load(f)
    else:
        print("\n>>> STEP 2/5: Running 50-Client Scalability Benchmark (N=50, Cp=0.2)...")
        t0 = time.time()
        scale_res = run_50clients_scaling(num_clients=50, clients_per_round=10, num_rounds=20, batch_size=64)
        print(f"--- Step 2 finished in {time.time() - t0:.1f}s ---")

    # 3. Cluster K Sensitivity
    k_path = os.path.join(_project_root, "outputs", "cluster_k_sensitivity.json")
    if os.path.exists(k_path):
        print("\n>>> STEP 3/5: Cluster sensitivity results already exist, loading from disk...")
        with open(k_path) as f: k_res = json.load(f)
    else:
        print("\n>>> STEP 3/5: Running Cluster Count Sensitivity Sweep (K in {1, 2, 3, 5, 8})...")
        t0 = time.time()
        k_res = run_k_sensitivity(num_clients=15, num_rounds=20, batch_size=64)
        print(f"--- Step 3 finished in {time.time() - t0:.1f}s ---")

    # 4. Multi-Attack Byzantine Suite
    print("\n>>> STEP 4/5: Running Multi-Attack Byzantine Suite...")
    t0 = time.time()
    byz_res = run_multi_attack_byzantine_suite(num_clients=15, num_rounds=15, batch_size=64)
    print(f"--- Step 4 finished in {time.time() - t0:.1f}s ---")

    # 5. Head Budget Ablation
    print("\n>>> STEP 5/5: Running Head-Epoch Budget Ratio Ablation...")
    t0 = time.time()
    budget_res = run_budget_ablation(num_clients=15, num_rounds=20, batch_size=64)
    print(f"--- Step 5 finished in {time.time() - t0:.1f}s ---")

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS SUCCESSFULLY COMPLETED IN {total_time/60.0:.2f} MINUTES!")
    print("=" * 80)

    summary = {
        "cifar100_results": cifar100_res,
        "scale_50clients_results": scale_res,
        "cluster_k_sensitivity": k_res,
        "byzantine_multi_attack": byz_res,
        "epoch_budget_ablation": budget_res,
        "total_execution_seconds": round(total_time, 2)
    }

    summary_path = os.path.join(_project_root, "outputs", "full_strengthening_suite_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Consolidated summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
