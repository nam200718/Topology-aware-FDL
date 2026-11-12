"""
CLI Entry Point: Runs the Midterm & Post-Midterm Evaluation Suites across:
- Midterm Ideas 1, 2, 3:
  1. Defense Scope Matrix (Global vs Cluster vs Both vs None) under Targeted Attacks.
  2. ALIE Attack Suite (FedAvg, Ditto, Multi-Krum, Defended FedHEP).
  3. Number of Peer Clusters (K) Sensitivity Sweep under non-IID Skews.
- Post-Midterm Studies (Supervisor Feedback Integration):
  postmid_attacks:  New Attacks Suite targeting TTT (IPM, Adaptive Cosine, Delayed Onset, ALIE, Sign Flip).
  postmid_majority: Majority Attacker Stress-Test (q >= 0.5 up to 0.7) for SCCF centroid inversion.
  postmid_k:        Multi-seed (42, 123, 7) Statistical K-Cluster Sweep (K in [1, 2, 3, 5, 8, 10, 15]).
  postmid_all:      Execute all three post-midterm studies in sequence.

Usage examples:
    python -m oreh.compare_scope_alie --dataset cifar100 --idea sanity
    python -m oreh.compare_scope_alie --dataset cifar100 --idea postmid_attacks --rounds 30
    python -m oreh.compare_scope_alie --dataset cifar100 --idea postmid_majority --rounds 30
    python -m oreh.compare_scope_alie --dataset cifar100 --idea postmid_k --rounds 30
    python -m oreh.compare_scope_alie --dataset cifar100 --idea postmid_all --rounds 30
"""

import argparse
import sys
import os
import pandas as pd
from oreh.comparison_runner import OrehComparisonRunner


def parse_args():
    parser = argparse.ArgumentParser(description="FedHEP Midterm & Post-Midterm Evaluation Suite")
    parser.add_argument(
        "--idea",
        type=str,
        default="1",
        choices=[
            "1",
            "2",
            "3",
            "all",
            "sanity",
            "postmid_attacks",
            "postmid_majority",
            "postmid_k",
            "postmid_baselines",
            "postmid_k_attack",
            "postmid_seed7",
            "postmid_all",
        ],
        help="Which experiment to execute",
    )
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"], help="Dataset to evaluate")
    parser.add_argument("--rounds", type=int, default=30, help="Number of federated communication rounds (default: 30)")
    parser.add_argument("--num_clients", type=int, default=15, help="Number of edge clients (default: 15)")
    parser.add_argument("--num_clusters", type=int, default=3, help="Number of peer clusters K (default: 3)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter for label skew (default: 0.5)")
    parser.add_argument(
        "--clustering_mode",
        type=str,
        default="none",
        choices=["none", "full_state", "weights_only"],
        help="Clustering mode for FedHEP (default: 'none', matching Study B & C)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', or auto-detect)")
    parser.add_argument("--train_subset", type=int, default=10000, help="Subset size for training data (default: 10000)")
    parser.add_argument("--test_subset", type=int, default=3000, help="Subset size for test data (default: 3000)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/oreh_eval", help="Output directory for CSV and logs")
    parser.add_argument("--quick", action="store_true", help="Quick mode with 2 rounds for testing")
    parser.add_argument("--seeds", type=str, default="42,123,7", help="Comma-separated random seeds (default: 42,123,7)")

    # Midterm Idea 1 specific overrides
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
    seeds_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("=" * 75)
    print("      FEDHEP EVALUATION SUITE (OREH BENCHMARK)")
    print(f" Idea Selected: {args.idea.upper()} | Dataset: {args.dataset.upper()} | Rounds: {rounds}")
    print(f" Device: {args.device or 'Auto'} | Clients: {args.num_clients} | Output Dir: {args.output_dir}")
    print(f" Random Seeds: {seeds_list}")
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
        clustering_mode=args.clustering_mode,
    )

    if args.idea == "sanity":
        print(">> Running 1-round pre-flight sanity check across new attacks...")
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
        test_attacks = ["ipm", "adaptive_cosine", "delayed_onset"]
        for t_atk in test_attacks:
            res = sanity_runner.run_single_experiment(
                defense_scope="both",
                attack_target_tier="all",
                attack_type=t_atk,
                byzantine_rate=0.2,
                seed=42,
            )
            print(f">> Sanity check for {t_atk.upper()} PASSED! Acc={res['final_test_acc']}% | Byz Trust={res['byzantine_trust_score']}")
        return

    # Midterm Idea 1
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

    # Midterm Idea 2
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

    # Midterm Idea 3
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

    # Post-Midterm Studies
    if args.idea in ("postmid_attacks", "postmid_all"):
        print("\n" + "#" * 75)
        print("# EXECUTING POST-MIDTERM STUDY C: NEW ATTACKS TARGETING TTT")
        print("#" * 75)
        df_c = runner.run_postmid_attacks(
            attacks=["sign_flip", "alie", "ipm", "adaptive_cosine", "delayed_onset"],
            byzantine_rates=[0.2, 0.3],
            seeds=seeds_list,
        )
        if not df_c.empty:
            cols_c = ["method", "attack_type", "byzantine_rate", "seed", "final_test_acc", "bottom_10_fairness", "byzantine_trust_score", "honest_trust_score"]
            print("\n--- POST-MIDTERM STUDY C (NEW ATTACKS) SUMMARY TABLE ---")
            print(df_c[[c for c in cols_c if c in df_c.columns]].to_string(index=False))

    if args.idea in ("postmid_majority", "postmid_all"):
        print("\n" + "#" * 75)
        print("# EXECUTING POST-MIDTERM STUDY B: MAJORITY ATTACKER STRESS TEST (q >= 0.5)")
        print("#" * 75)
        df_b = runner.run_postmid_majority_q(
            byzantine_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            attacks=["sign_flip", "alie"],
            methods=["fedhep", "fedavg", "ditto"],
            seeds=seeds_list,
        )
        if not df_b.empty:
            cols_b = ["method", "attack_type", "byzantine_rate", "seed", "final_test_acc", "bottom_10_fairness", "byzantine_trust_score"]
            print("\n--- POST-MIDTERM STUDY B (MAJORITY q >= 0.5) SUMMARY TABLE ---")
            print(df_b[[c for c in cols_b if c in df_b.columns]].to_string(index=False))

    if args.idea in ("postmid_k", "postmid_all"):
        print("\n" + "#" * 75)
        print("# EXECUTING POST-MIDTERM STUDY A: MULTI-SEED K-CLUSTER SENSITIVITY SWEEP")
        print("#" * 75)
        df_a = runner.run_postmid_k_clusters(
            k_values=[1, 2, 3, 5, 8, 10, 15],
            alphas=[0.05, 0.1, 0.5],
            seeds=seeds_list,
            byzantine_rate=0.0,
        )
        if not df_a.empty:
            cols_a = ["num_clusters", "alpha", "seed", "final_test_acc", "bottom_10_fairness", "total_runtime_s"]
            print("\n--- POST-MIDTERM STUDY A (MULTI-SEED K SWEEP) SUMMARY TABLE ---")
            print(df_a[[c for c in cols_a if c in df_a.columns]].to_string(index=False))

    if args.idea == "postmid_baselines":
        print("\n" + "#" * 75)
        print("# EXECUTING POST-MIDTERM P1: ROBUST BASELINES (MULTIKRUM, MEDIAN, TRIMMED_MEAN)")
        print("#" * 75)
        df_c_base = runner.run_postmid_attacks(
            methods=["multikrum", "median"],
            attacks=["alie", "ipm", "sign_flip", "adaptive_cosine", "delayed_onset"],
            byzantine_rates=[0.2, 0.3],
            seeds=seeds_list,
            output_filename=f"{args.dataset}_postmid_attacks_baselines.csv",
        )
        if not df_c_base.empty:
            print("\n--- P1 STUDY C BASELINES SUMMARY ---")
            print(df_c_base[["method", "attack_type", "byzantine_rate", "seed", "final_test_acc", "bottom_10_fairness"]].to_string(index=False))

        df_b_base = runner.run_postmid_majority_q(
            byzantine_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            attacks=["sign_flip", "alie"],
            methods=["multikrum", "trimmed_mean"],
            seeds=[42, 123],
            output_filename=f"{args.dataset}_postmid_majority_q_baselines.csv",
        )
        if not df_b_base.empty:
            print("\n--- P1 STUDY B BASELINES SUMMARY ---")
            print(df_b_base[["method", "attack_type", "byzantine_rate", "seed", "final_test_acc", "bottom_10_fairness"]].to_string(index=False))

    if args.idea == "postmid_k_attack":
        print("\n" + "#" * 75)
        print("# EXECUTING POST-MIDTERM P2: K-SENSITIVITY UNDER ALIE ATTACK (10N / 15N / 20N)")
        print("#" * 75)
        df_ka = runner.run_postmid_k_attack_sensitivity(
            client_k_grid={10: [2, 3], 15: [2, 3, 4], 20: [4, 5]},
            attack_type="alie",
            byzantine_rate=0.2,
            seeds=[42, 123],
            alpha=args.alpha,
        )
        if not df_ka.empty:
            print("\n--- P2 K-SENSITIVITY UNDER ATTACK SUMMARY ---")
            print(df_ka[["num_clients", "num_clusters", "k_over_n_pct", "seed", "final_test_acc", "bottom_10_fairness"]].to_string(index=False))

    if args.idea == "postmid_seed7":
        print("\n" + "#" * 75)
        print("# EXECUTING POST-MIDTERM P3: STUDY B SEED=7 PHASE TRANSITION (q=0.3, 0.4)")
        print("#" * 75)
        df_s7 = runner.run_postmid_majority_q(
            byzantine_rates=[0.3, 0.4],
            attacks=["sign_flip", "alie"],
            methods=["fedhep", "fedavg", "ditto"],
            seeds=[7],
            output_filename=f"{args.dataset}_postmid_majority_q_seed7.csv",
        )
        if not df_s7.empty:
            print("\n--- P3 STUDY B SEED=7 SUMMARY ---")
            print(df_s7[["method", "attack_type", "byzantine_rate", "seed", "final_test_acc", "bottom_10_fairness"]].to_string(index=False))

    print("\n" + "=" * 75)
    print(">> BENCHMARK COMPLETED SUCCESSFULLY! All results saved to:", args.output_dir)
    print("=" * 75)


if __name__ == "__main__":
    main()
