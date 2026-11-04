"""
CLI Entry Point: Runs the Midterm Evaluation Suite across Ideas 1, 2, and 3:
1. Defense Scope Matrix (Global vs Cluster vs Both vs None) under Targeted Attacks.
2. ALIE Attack Suite (FedAvg, Ditto, Multi-Krum, Defended FedHEP).
3. Number of Peer Clusters (K) Sensitivity Sweep under non-IID Skews.

Usage examples:
    python -m oreh.compare_scope_alie --dataset cifar100 --idea sanity
    python -m oreh.compare_scope_alie --dataset cifar100 --idea 1 --rounds 25
    python -m oreh.compare_scope_alie --dataset cifar100 --idea 2 --rounds 25
    python -m oreh.compare_scope_alie --dataset cifar100 --idea 3 --rounds 25
    python -m oreh.compare_scope_alie --dataset cifar100 --idea all --rounds 25
"""

import argparse
import sys
import os
import pandas as pd
from oreh.comparison_runner import OrehComparisonRunner


def parse_args():
    parser = argparse.ArgumentParser(description="FedHEP Midterm Evaluation Suite (Ideas 1, 2, 3)")
    parser.add_argument("--idea", type=str, default="1", choices=["1", "2", "3", "all", "sanity"],
                        help="Which Midterm Idea to execute (1: Scope, 2: ALIE Suite, 3: K Sweep, all: All three, sanity: 1-round preflight)")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"], help="Dataset to evaluate")
    parser.add_argument("--rounds", type=int, default=25, help="Number of federated communication rounds (default: 25)")
    parser.add_argument("--num_clients", type=int, default=15, help="Number of edge clients (default: 15)")
    parser.add_argument("--num_clusters", type=int, default=3, help="Number of peer clusters K (default: 3)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter for label skew (default: 0.5)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', or auto-detect)")
    parser.add_argument("--train_subset", type=int, default=10000, help="Subset size for training data (default: 10000)")
    parser.add_argument("--test_subset", type=int, default=3000, help="Subset size for test data (default: 3000)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/oreh_eval", help="Output directory for CSV and logs")
    parser.add_argument("--quick", action="store_true", help="Quick mode with 2 rounds for testing")

    # Idea 1 specific overrides
    parser.add_argument("--attacks", type=str, default="alie,sign_flip", help="Comma-separated attacks for Idea 1")
    parser.add_argument("--scopes", type=str, default="global,cluster,both,none", help="Comma-separated defense scopes for Idea 1")
    parser.add_argument("--targets", type=str, default="root_only,parent_only,all", help="Comma-separated attack target tiers for Idea 1")
    parser.add_argument("--byzantine_rate", type=float, default=0.2, help="Byzantine fraction q for Idea 1 (default: 0.2)")

    return parser.parse_args()


def main():
    args = parse_args()

    rounds = 2 if args.quick else args.rounds
    train_subset = 1000 if args.quick else args.train_subset
    test_subset = 500 if args.quick else args.test_subset

    print("=" * 75)
    print("      FEDHEP MIDTERM EVALUATION SUITE (OREH BENCHMARK)")
    print(f" Idea Selected: {args.idea.upper()} | Dataset: {args.dataset.upper()} | Rounds: {rounds}")
    print(f" Device: {args.device or 'Auto'} | Clients: {args.num_clients} | Output Dir: {args.output_dir}")
    print("=" * 75)

    runner = OrehComparisonRunner(
        dataset=args.dataset,
        num_rounds=rounds,
        num_clients=args.num_clients,
        num_clusters=args.num_clusters,
        alpha=args.alpha,
        device=args.device,
        train_subset=train_subset,
        test_subset=test_subset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    if args.idea == "sanity":
        print(">> Running 1-round pre-flight sanity check...")
        sanity_runner = OrehComparisonRunner(
            dataset=args.dataset,
            num_rounds=1,
            num_clients=args.num_clients,
            num_clusters=args.num_clusters,
            alpha=args.alpha,
            device=args.device,
            train_subset=min(1000, train_subset),
            test_subset=min(500, test_subset),
            data_dir=args.data_dir,
            output_dir="./outputs/oreh_sanity",
        )
        res = sanity_runner.run_single_experiment(
            defense_scope="both",
            attack_target_tier="all",
            attack_type="alie",
            byzantine_rate=0.2,
            seed=42,
        )
        print(">> Pre-flight sanity check PASSED!")
        print(f"Result: Acc={res['final_test_acc']}% | Defense Latency={res['total_defense_latency_ms']} ms | Byz Trust={res['byzantine_trust_score']}")
        return

    if args.idea in ("1", "all"):
        print("\n" + "#" * 75)
        print("# EXECUTING IDEA 1: DEFENSE SCOPE VS TARGETED ATTACK TIERS")
        print("#" * 75)
        attack_types = [a.strip() for a in args.attacks.split(",") if a.strip()]
        scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
        df_1 = runner.run_idea1_scope_matrix(
            attack_types=attack_types,
            scopes=scopes,
            target_tiers=targets,
            byzantine_rate=args.byzantine_rate,
            seed=42,
        )
        if not df_1.empty:
            cols_1 = ["attack_type", "attack_target_tier", "defense_scope", "final_test_acc", "bottom_10_fairness", "total_defense_latency_ms", "byzantine_trust_score"]
            print("\n--- IDEA 1 SUMMARY TABLE ---")
            print(df_1[[c for c in cols_1 if c in df_1.columns]].to_string(index=False))

    if args.idea in ("2", "all"):
        print("\n" + "#" * 75)
        print("# EXECUTING IDEA 2: ALIE ATTACK SUITE ACROSS BASELINES")
        print("#" * 75)
        df_2 = runner.run_idea2_alie_suite(
            methods=["fedavg", "ditto", "multikrum", "fedhep"],
            byzantine_rates=[0.1, 0.2, 0.3, 0.4],
            seed=42,
        )
        if not df_2.empty:
            cols_2 = ["method", "byzantine_rate", "final_test_acc", "bottom_10_fairness", "byzantine_trust_score", "total_runtime_s"]
            print("\n--- IDEA 2 SUMMARY TABLE ---")
            print(df_2[[c for c in cols_2 if c in df_2.columns]].to_string(index=False))

    if args.idea in ("3", "all"):
        print("\n" + "#" * 75)
        print("# EXECUTING IDEA 3: PEER CLUSTERS (K) SENSITIVITY SWEEP")
        print("#" * 75)
        df_3 = runner.run_idea3_cluster_sweep(
            k_values=[1, 2, 3, 5, 8],
            alphas=[0.5, 0.05],
            byzantine_rate=0.0,
            seed=42,
        )
        if not df_3.empty:
            cols_3 = ["num_clusters", "alpha", "final_test_acc", "bottom_10_fairness", "total_runtime_s"]
            print("\n--- IDEA 3 SUMMARY TABLE ---")
            print(df_3[[c for c in cols_3 if c in df_3.columns]].to_string(index=False))

    print("\n" + "=" * 75)
    print(">> BENCHMARK COMPLETED SUCCESSFULLY! All results saved to:", args.output_dir)
    print("=" * 75)


if __name__ == "__main__":
    main()
