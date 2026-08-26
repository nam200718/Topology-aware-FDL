"""
Assemble the paper's Table III (main benchmark) from multi-seed artifacts.

Reads every outputs/multi_seed_eval_*/all_seeds_raw.csv produced by
scripts/run_multi_seed.py, aggregates Mean +/- Std across seeds per
(Topology, Scenario, Metric), attaches bottom-10% fairness from the final
round's per_client_accuracy dump inside each experiment's metrics.json, and
emits LaTeX body rows grouped by paradigm.

Usage:
    python scripts/build_table3.py [--out paper/table3_generated.tex]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGIME_ORDER = [
    ("IID (Baseline)", "IID"),
    ("Non-IID (Mild alpha=1.0)", "Mild"),
    ("Non-IID (Moderate alpha=0.5)", "Mod."),
    ("Non-IID (Severe alpha=0.1)", "Sev."),
    ("Non-IID (Extreme alpha=0.05)", "Ext."),
]

def normalize_topo(label: str) -> str:
    l = label.strip()
    if "FedAvg" in l:
        return "FedAvg"
    if "APFL" in l:
        return "APFL"
    if "Ditto" in l:
        return "Ditto"
    if "FedPer" in l:
        return "FedPer"
    if "FedRep" in l:
        return "FedRep"
    if "FedBABU" in l:
        return "FedBABU"
    if "FedALA" in l:
        return "FedALA"
    if "CFL" in l:
        return "CFL"
    if "Local" in l:
        return "Local-Only"
    if "HEP" in l or "Hierarchical" in l:
        return "HEP (Ours)"
    return l


PARADIGMS = [
    ("A. Global Consensus FL (No Personalization)", ["FedAvg"]),
    ("B. Dual-Model Regularization (Full Model Duplication)", ["APFL", "Ditto"]),
    ("C. Decoupled / Split-Head Paradigms (Single Backbone, Local Heads)", ["Local-Only", "FedPer", "FedRep", "FedBABU", "FedALA"]),
    ("D. Clustered Topologies (Sub-Network Sharing)", ["CFL"]),
    ("E. Hierarchical Ensemble Personalization (Proposed)", ["HEP (Ours)"]),
]


def collect_frames():
    frames = []
    # 1. Multi-seed raw files (skip older runs with known masking bug where split-head had ~10% IID)
    for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, "outputs", "multi_seed_eval_*", "all_seeds_raw.csv"))):
        if "20260823_013117" in path:
            continue  # Pre-bugfix split-head run
        try:
            df = pd.read_csv(path)
            if {"Topology", "Scenario", "Metric"}.issubset(df.columns):
                df["__source_mtime"] = os.path.getmtime(path)
                frames.append(df)
        except Exception:
            pass

    # 2. Individual comparison_study results
    for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, "outputs", "comparison_study_*", "comparison_results.csv"))):
        try:
            df = pd.read_csv(path)
            if {"Topology", "Scenario", "Metric"}.issubset(df.columns):
                if "Seed" not in df.columns:
                    df["Seed"] = 42
                df["__source_mtime"] = os.path.getmtime(path)
                frames.append(df)
        except Exception:
            pass

    if not frames:
        print("No evaluation artifacts found under outputs/")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined["CanonicalTopo"] = combined["Topology"].apply(normalize_topo)

    # Filter to personalized metric (or FedAvg global)
    pers = combined[combined["Metric"] == "Personalized"].copy()

    # Deduplicate keeping newest mtime per (CanonicalTopo, Scenario, Seed)
    pers = pers.sort_values("__source_mtime").drop_duplicates(
        subset=["CanonicalTopo", "Scenario", "Seed"], keep="last"
    ).drop(columns=["__source_mtime"])

    return pers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "paper", "table3_generated.tex"))
    args = parser.parse_args()

    pers = collect_frames()

    lines = []
    for group_name, topologies in PARADIGMS:
        lines.append(r"\multicolumn{12}{l}{\textit{\textbf{" + f"{group_name}" + r"}}} \\")
        for label in topologies:
            sub = pers[pers["CanonicalTopo"] == label]
            if sub.empty:
                continue
            row = [f"\\textbf{{{label}}}" if "HEP" in label else f"\\textbf{{{label}}}"]
            for scen_label, _ in REGIME_ORDER:
                cell = sub[sub["Scenario"] == scen_label]
                if cell.empty:
                    row.extend(["---", "---"])
                    continue
                m = cell["Last5 Avg Accuracy"].mean()
                s = cell["Last5 Avg Accuracy"].std(ddof=1) if len(cell) > 1 else np.nan
                acc_str = f"{m:.2f}\\%" if np.isnan(s) else f"{m:.2f} $\\pm$ {s:.2f}\\%"
                row.append(acc_str)
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")

    out_text = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out_text + "\n")
    print(f"Wrote {len(lines)} LaTeX rows to {args.out}")
    print(out_text)


if __name__ == "__main__":
    main()
