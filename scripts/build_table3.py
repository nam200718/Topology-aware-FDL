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

PARADIGMS = [
    ("Global Consensus", ["Star (FedAvg)"]),
    ("Dual-Model Regularization", ["Star (APFL)", "Star (Ditto)"]),
    ("Split-Head Personalization", ["Star (FedPer)", "Star (FedRep)", "Star (FedBABU)", "Local-Only"]),
    ("Clustered Topologies", ["CFL"]),
    ("Hierarchical Ensemble", ["HEP"]),
]


def collect_frames():
    paths = sorted(
        glob.glob(os.path.join(PROJECT_ROOT, "outputs", "multi_seed_eval_*", "all_seeds_raw.csv")),
        key=os.path.getmtime,
    )
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if {"Topology", "Scenario", "Metric", "Seed"}.issubset(df.columns):
            df["__source_mtime"] = os.path.getmtime(path)
            frames.append(df)
    if not frames:
        print("No multi-seed artifacts found under outputs/multi_seed_eval_*/")
        sys.exit(1)
    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate repeated (topology, scenario, metric, seed) keeping the
    # NEWEST source run (later mtime wins), then drop the helper column.
    combined = combined.sort_values("__source_mtime").drop_duplicates(
        subset=["Topology", "Scenario", "Metric", "Seed"], keep="last"
    ).drop(columns=["__source_mtime"])
    return combined


def bottom10_for(topology_label: str, scenario_id: str):
    """Mean bottom-10% fairness across seeds from final-round per-client dumps."""
    import re
    safe = re.sub(r"[^a-z0-9]+", "_", topology_label.lower()).strip("_")
    vals = []
    pattern = os.path.join(PROJECT_ROOT, "outputs", "multi_seed_eval_*")
    for run_dir in sorted(glob.glob(pattern)):
        metrics_dir = os.path.join(run_dir)
        # each multi_seed dir mirrors comparison_study subdirs? No: raw csv only.
        # Per-experiment metrics live in the nested comparison_study_<ts> dirs;
        # fall back to scanning all comparison_study dirs by experiment name.
        for exp_dir in glob.glob(os.path.join(PROJECT_ROOT, "outputs", "comparison_study_*")):
            mj = os.path.join(exp_dir, "metrics", f"{safe}_{scenario_id}", "metrics.json")
            if not os.path.exists(mj):
                continue
            try:
                with open(mj) as f:
                    hist = json.load(f)
            except Exception:
                continue
            last = hist[-1] if hist else None
            if last and "per_client_accuracy" in last:
                accs = sorted(last["per_client_accuracy"].values())
                k = max(1, int(np.ceil(0.1 * len(accs))))
                vals.append(float(np.mean(accs[:k])))
    return float(np.mean(vals)) if vals else None


def fmt(mean, std, digits=2):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "---"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "paper", "table3_generated.tex"))
    args = parser.parse_args()

    df = collect_frames()
    pers = df[df["Metric"] == "Personalized"]
    if pers.empty:
        print("No Personalized rows found.")
        sys.exit(1)

    lines = []
    for group_name, topologies in PARADIGMS:
        matched = [(nice, t) for nice, t in ((lbl, lbl) for lbl in topologies)]
        if not any(t in set(pers["Topology"]) for _, t in matched):
            continue
        lines.append(r"\multicolumn{11}{l}{\textit{\textbf{" + f"{group_name}" + r"}}} \\")
        for label in topologies:
            sub = pers[pers["Topology"] == label]
            if sub.empty:
                continue
            row = [f"\\textbf{{{label.replace('Star ', '')}}}" if "HEP" in label else label]
            for scen_label, _ in REGIME_ORDER:
                cell = sub[sub["Scenario"] == scen_label]
                if cell.empty:
                    row.append("---")
                    continue
                m = cell["Last5 Avg Accuracy"].mean()
                s = cell["Last5 Avg Accuracy"].std(ddof=1) if len(cell) > 1 else np.nan
                row.append(fmt(m, s))
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")

    out_text = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out_text + "\n")
    print(f"Wrote {len(lines)} LaTeX rows to {args.out}")
    print(out_text)


if __name__ == "__main__":
    main()
