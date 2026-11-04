"""
CLI Entry Point: Runs the unified comparison of Defense Scopes (Root vs Cluster vs Both)
and Byzantine Attacks (ALIE, Sign-Flip, Label-Flip) across targeted tiers.

Usage:
    python -m oreh.compare_scope_alie --dataset cifar10 --rounds 15 --device cuda
    python -m oreh.compare_scope_alie --dataset cifar100 --rounds 15 --device cuda --quick
"""

import argparse
import sys
import os
import pandas as pd
from oreh.comparison_runner import OrehComparisonRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Defense Scope & ALIE Evaluation Runner")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"], help="Dataset to evaluate")
    parser.add_argument("--rounds", type=int, default=15, help="Number of federated communication rounds")
    parser.add_argument("--num_clients", type=int, default=15, help="Number of edge clients")
    parser.add_argument("--num_clusters", type=int, default=3, help="Number of peer clusters (K)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter for label skew")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', or auto-detect)")
    parser.add_argument("--quick", action="store_true", help="Quick sanity check mode with fewer combinations and 2 rounds")
    parser.add_argument("--attacks", type=str, default="alie,sign_flip", help="Comma-separated attacks to run (e.g. 'alie,sign_flip')")
    parser.add_argument("--scopes", type=str, default="global,cluster,both", help="Comma-separated defense scopes (e.g. 'global,cluster,both')")
    parser.add_argument("--targets", type=str, default="root_only,parent_only,all", help="Comma-separated attack target tiers (e.g. 'root_only,parent_only,all')")
    parser.add_argument("--byzantine_rate", type=float, default=0.2, help="Byzantine fraction q (default: 0.2)")
    parser.add_argument("--output_dir", type=str, default="./outputs/oreh_eval", help="Output directory for CSV and logs")
    return parser.parse_args()


def main():
    args = parse_args()

    rounds = 2 if args.quick else args.rounds
    train_subset = 2000 if args.quick else 10000
    test_subset = 1000 if args.quick else 3000

    attack_types = [a.strip() for a in args.attacks.split(",") if a.strip()]
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    if args.quick:
        scopes = ["global", "both"]
        targets = ["root_only", "all"]
        attack_types = ["alie"]

    print("=" * 70)
    print("      OREH: FEDHEP MULTI-TIER DEFENSE SCOPE & ALIE BENCHMARK")
    print(f" Dataset: {args.dataset.upper()} | Rounds: {rounds} | Device: {args.device or 'Auto'}")
    print(f" Defense Scopes: {scopes}")
    print(f" Targeted Tiers: {targets}")
    print(f" Attacks: {attack_types} | Byzantine Rate: {args.byzantine_rate}")
    print("=" * 70)

    runner = OrehComparisonRunner(
        dataset=args.dataset,
        num_rounds=rounds,
        num_clients=args.num_clients,
        num_clusters=args.num_clusters,
        alpha=args.alpha,
        device=args.device,
        train_subset=train_subset,
        test_subset=test_subset,
        output_dir=args.output_dir,
    )

    df = runner.run_scope_vs_target_matrix(
        attack_types=attack_types,
        scopes=scopes,
        target_tiers=targets,
        byzantine_rate=args.byzantine_rate,
        seed=42,
    )

    print("\n" + "=" * 70)
    print("                     FINAL COMPARISON TABLE")
    print("=" * 70)
    cols_to_show = [
        "attack_type",
        "attack_target_tier",
        "defense_scope",
        "final_test_acc",
        "bottom_10_fairness",
        "tier1_latency_ms",
        "tier2_latency_ms",
        "total_defense_latency_ms",
        "byzantine_trust_score",
    ]
    present_cols = [c for c in cols_to_show if c in df.columns]
    summary_df = df[present_cols]
    print(summary_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()

