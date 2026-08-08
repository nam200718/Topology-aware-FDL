import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed evaluation suite with compute transparency.")
    parser.add_argument("--config", type=str, default="configs/evaluation_full.yaml", help="Path to config YAML")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7], help="List of random seeds to evaluate")
    args = parser.parse_args()

    print("=" * 80)
    print(f"MULTI-SEED EVALUATION SUITE")
    print(f"Config: {args.config}")
    print(f"Seeds:  {args.seeds}")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"multi_seed_eval_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    all_seed_results = []

    for seed in args.seeds:
        print(f"\n>>> Running evaluation for Seed {seed} ...")
        runner = ExperimentRunner.from_yaml(args.config, overrides={"seed": seed})
        runner.run()

        # Locate the output directory created by runner
        # Find latest output folder starting with comparison_study_
        outputs_root = runner.exp_config.env.output_dir
        subdirs = [os.path.join(outputs_root, d) for d in os.listdir(outputs_root) if os.path.isdir(os.path.join(outputs_root, d))]
        subdirs.sort(key=os.path.getmtime)
        latest_run_dir = subdirs[-1]

        csv_path = os.path.join(latest_run_dir, "comparison_results.csv")
        if os.path.exists(csv_path):
            df_seed = pd.read_csv(csv_path)
            df_seed["Seed"] = seed
            all_seed_results.append(df_seed)

    if not all_seed_results:
        print("ERROR: No comparison results collected across seeds.")
        sys.exit(1)

    combined_df = pd.concat(all_seed_results, ignore_index=True)
    combined_df.to_csv(os.path.join(out_dir, "all_seeds_raw.csv"), index=False)

    group_cols = ["Topology", "Scenario", "Metric"]
    num_cols = ["Last5 Avg Accuracy", "Peak Accuracy", "Final Accuracy", "Wall Clock Seconds", "Total Local Epochs"]

    agg_dict = {}
    for c in ["Last5 Avg Accuracy", "Peak Accuracy", "Final Accuracy", "Wall Clock Seconds"]:
        if c in combined_df.columns:
            agg_dict[c] = ["mean", "std"]
    if "Total Local Epochs" in combined_df.columns:
        agg_dict["Total Local Epochs"] = "first"

    summary_df = combined_df.groupby(group_cols).agg(agg_dict).reset_index()

    # Flatten multi-index columns cleanly
    flattened_cols = []
    for col in summary_df.columns:
        if isinstance(col, tuple):
            if col[1]:
                flattened_cols.append(f"{col[0]}_{col[1]}")
            else:
                flattened_cols.append(col[0])
        else:
            flattened_cols.append(col)
    summary_df.columns = flattened_cols

    # Format Mean ± Std string columns for clean presentation
    if "Last5 Avg Accuracy_mean" in summary_df.columns and "Last5 Avg Accuracy_std" in summary_df.columns:
        summary_df["Last5_Avg_Mean_Std"] = summary_df.apply(
            lambda r: f"{r['Last5 Avg Accuracy_mean']:.2f} ± {r['Last5 Avg Accuracy_std']:.2f}%", axis=1
        )
    if "Peak Accuracy_mean" in summary_df.columns and "Peak Accuracy_std" in summary_df.columns:
        summary_df["Peak_Mean_Std"] = summary_df.apply(
            lambda r: f"{r['Peak Accuracy_mean']:.2f} ± {r['Peak Accuracy_std']:.2f}%", axis=1
        )

    summary_df.to_csv(os.path.join(out_dir, "multi_seed_summary.csv"), index=False)

    print("\n" + "=" * 80)
    print("MULTI-SEED EVALUATION COMPLETE")
    print(f"Results saved to: {out_dir}")
    print("=" * 80)
    print("\nSummary (Last-5 Round Avg Accuracy Mean ± Std across seeds):")
    print("-" * 80)
    display_cols = [c for c in ["Topology", "Scenario", "Metric", "Last5_Avg_Mean_Std", "Peak_Mean_Std", "Total Local Epochs_first", "Wall Clock Seconds_mean"] if c in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
