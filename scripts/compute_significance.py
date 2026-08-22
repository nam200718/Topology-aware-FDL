"""
Paired two-tailed t-tests for the paper's significance claims (alpha=0.05).

Consumes one or more result CSVs (multi_seed all_seeds_raw.csv and/or single
comparison_results.csv with an explicit --seed tag), normalizes them into
long-form per-seed accuracies, and tests HEP against every other method per
scenario x metric using scipy.stats.ttest_rel on the common seed pairs.

Usage:
    python scripts/compute_significance.py \
        --raw outputs/multi_seed_eval_X/all_seeds_raw.csv \
        --raw outputs/comparison_study_20260822_025156/comparison_results.csv --seed 42 \
        --out outputs/paper_suite/significance_tests.csv
"""
import argparse
import os
import sys

import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEP_LABELS = ("hep", "hierarchical")


def is_hep(topology: str) -> bool:
    t = topology.lower()
    return any(h in t for h in HEP_LABELS) and "conv" not in t


def normalize(paths_with_seeds):
    frames = []
    for path, seed in paths_with_seeds:
        df = pd.read_csv(path)
        if "Seed" not in df.columns:
            df["Seed"] = seed
        keep = [c for c in ["Topology", "Scenario", "Metric", "Seed",
                            "Final Accuracy", "Last5 Avg Accuracy"] if c in df.columns]
        frames.append(df[keep])
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["Topology", "Scenario", "Metric", "Seed"], keep="last")


def paired_tests(combined, alpha=0.05):
    records = []
    metric_cols = [c for c in ["Final Accuracy", "Last5 Avg Accuracy"] if c in combined.columns]
    scenarios = sorted(combined["Scenario"].unique())

    for scenario in scenarios:
        sub = combined[combined["Scenario"] == scenario]
        hep_rows = sub[sub["Topology"].apply(is_hep)]
        if hep_rows.empty:
            continue
        for metric in metric_cols:
            hep_by_seed = hep_rows.groupby("Seed")[metric].mean()
            for method in sorted(sub["Topology"].unique()):
                if is_hep(method):
                    continue
                base_rows = sub[sub["Topology"] == method]
                base_by_seed = base_rows.groupby("Seed")[metric].mean()
                common = sorted(set(hep_by_seed.index) & set(base_by_seed.index))
                if len(common) < 2:
                    continue
                a = hep_by_seed.loc[common].values
                b = base_by_seed.loc[common].values
                t_stat, p_value = stats.ttest_rel(a, b)
                records.append({
                    "Scenario": scenario,
                    "Metric": metric,
                    "Baseline": method,
                    "HEP_mean": round(float(a.mean()), 3),
                    "Baseline_mean": round(float(b.mean()), 3),
                    "Delta_pp": round(float(a.mean() - b.mean()), 2),
                    "n_seeds": len(common),
                    "t_stat": round(float(t_stat), 3),
                    "p_value": round(float(p_value), 4),
                    f"significant@{alpha}": bool(p_value < alpha),
                })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="append", required=True,
                        help="Path to all_seeds_raw.csv or comparison_results.csv")
    parser.add_argument("--seed", type=int, action="append", default=[],
                        help="Seed tag for the preceding --raw (order-sensitive)")
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "paper_suite", "significance_tests.csv"))
    args = parser.parse_args()

    # Pair each raw with a seed tag: last --seed applies to remaining untagged tails
    seeds = list(args.seed)
    pairs = []
    for i, path in enumerate(args.raw):
        tag = seeds[i] if i < len(seeds) else (seeds[-1] if seeds else None)
        pairs.append((path, tag))

    combined = normalize(pairs)
    if combined.empty:
        print("No usable rows found.")
        sys.exit(1)

    result = paired_tests(combined)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result.to_csv(args.out, index=False)

    print(f"Wrote {len(result)} paired comparisons to {args.out}")
    if not result.empty:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
