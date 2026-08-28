"""
Generate all paper figures from collected artifacts.

Discovers the newest matching run directories under outputs/, tolerates
missing artifacts (skips with a warning), and writes PNGs to paper/figures/.

Usage:
    python scripts/make_paper_figures.py            # generate everything found
    python scripts/make_paper_figures.py --only convergence byzantine
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, "report", "figures")

METHOD_STYLE = {
    "FedAvg": dict(color="#616161", linestyle="--", linewidth=2.0),
    "Ditto": dict(color="#1976D2", linestyle="-.", linewidth=2.0),
    "FedRep": dict(color="#388E3C", linestyle=":", linewidth=2.0),
    "HEP": dict(color="#D32F2F", linestyle="-", linewidth=2.5),
}


def _latest(pattern):
    hits = sorted(glob.glob(os.path.join(PROJECT_ROOT, "outputs", pattern)), key=os.path.getmtime)
    return hits[-1] if hits else None


def _load_metrics(exp_dir, exp_name):
    path = os.path.join(exp_dir, "metrics", exp_name, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fig_convergence():
    """Fig 2: per-round personalized accuracy, panels IID + extreme skew."""
    root = _latest("comparison_study_*")
    if root is None or "conv" not in root.lower() and "paper_suite" not in root.lower():
        candidates = sorted(glob.glob(os.path.join(PROJECT_ROOT, "outputs", "comparison_study_*")), key=os.path.getmtime)
        root = None
        for cand in reversed(candidates):
            if glob.glob(os.path.join(cand, "metrics", "conv_*")):
                root = cand
                break
    if root is None:
        print("[skip] convergence: no conv runs found")
        return

    methods = [("Conv Star (FedAvg)", "FedAvg"), ("Conv Star (Ditto)", "Ditto"), ("Conv HEP", "HEP")]
    panels = [("iid", "Uniform IID Distribution"), ("non_iid_alpha_0.05", r"Extreme Non-IID Skew ($\alpha=0.05$)")]

    def sanitize(label):
        import re
        return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300)
    plotted = False
    for ax, (scen, title) in zip(axes, panels):
        for label, nice in methods:
            hist = _load_metrics(root, f"{sanitize(label)}_{scen}")
            if hist is None:
                continue
            rounds = [r["round"] for r in hist if r.get("ensemble_test_accuracy") is not None]
            accs = [r["ensemble_test_accuracy"] for r in hist if r.get("ensemble_test_accuracy") is not None]
            if not accs:
                continue
            style = METHOD_STYLE.get(nice, {})
            ax.plot(rounds, accs, marker="o", markersize=5, label=nice, **style)
            plotted = True
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Communication Round", fontsize=11.5, fontweight="bold")
        ax.set_ylabel("Personalized Accuracy (%)", fontsize=11.5, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10.5)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=10.5, loc="lower right", framealpha=0.92)
    if plotted:
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "convergence_combined.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[ok] {out}")
    else:
        print("[skip] convergence: no accuracy rows")
    plt.close(fig)


def fig_byzantine():
    """Fig 3: label-flip sweep from matrix_results.csv (+ sign-flip reference row)."""
    candidates = []
    for path in glob.glob(os.path.join(PROJECT_ROOT, "outputs", "byzantine_matrix_*", "matrix_results.csv")):
        df = pd.read_csv(path)
        rates = set(df["Byzantine Rate"].round(2))
        if {0.0, 0.1, 0.2, 0.3, 0.4}.issubset(rates):
            candidates.append((os.path.getmtime(path), df))
    if not candidates:
        print("[skip] byzantine: no full label-flip grid yet")
        return
    df = max(candidates, key=lambda t: t[0])[1]
    df["Method"] = df["Topology"].str.replace(r"Star \((.*)\)|HEP.*", lambda m: m.group(1) if m.group(1) else "HEP", regex=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
    for method, grp in df.groupby("Method"):
        grp = grp.sort_values("Byzantine Rate")
        style = METHOD_STYLE.get(method, {})
        ax.plot(grp["Byzantine Rate"] * 100, grp["Last5 Avg Accuracy"],
                marker="s", markersize=6, label=method, **style)
    ax.set_xlabel("Attacker Fraction (%)", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Personalized Accuracy (%, last-5 avg)", fontsize=11.5, fontweight="bold")
    ax.set_title(r"Label-Flipping Robustness ($\alpha=0.5$, 15 rounds)", fontsize=13, fontweight="bold", pad=8)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10.5, loc="lower left", framealpha=0.92)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "byzantine_label_flip.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[ok] {out}")
    plt.close(fig)


def fig_pareto():
    """Accuracy vs wall-clock Pareto scatter from a comparison summary."""
    candidates = []
    for path in glob.glob(os.path.join(PROJECT_ROOT, "outputs", "comparison_study_*", "comparison_results.csv")):
        df = pd.read_csv(path)
        pers = df[df["Metric"] == "Personalized"]
        if len(pers) >= 3:
            candidates.append((os.path.getmtime(path), pers))
    if not candidates:
        print("[skip] pareto: no comparison summaries found")
        return
    pers = max(candidates, key=lambda t: t[0])[1].copy()
    pers["Base"] = pers["Topology"].str.replace(r" \(Pers\.\)$", "", regex=True)

    def family(base):
        b = base.lower()
        if "hierarchical" in b or "hep" in b:
            return "HEP"
        if "ditto" in b:
            return "Ditto"
        if "apfl" in b:
            return "APFL"
        return "FedAvg-class"

    pers["Family"] = pers["Base"].apply(family)
    agg = pers.groupby(["Family", "Scenario"]).agg(acc=("Final Accuracy", "mean"), secs=("Wall Clock Seconds", "first")).reset_index()

    colors = {"HEP": "#d62728", "Ditto": "#1f77b4", "APFL": "#ff7f0e", "FedAvg-class": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for fam, grp in agg.groupby("Family"):
        ax.scatter(grp["secs"], grp["acc"], s=70, label=fam, color=colors.get(fam), edgecolors="k", linewidths=0.5)
    ax.set_xlabel("Wall-Clock per Experiment (s)")
    ax.set_ylabel("Final Personalized Accuracy (%)")
    ax.set_title("Accuracy / Cost Trade-off Across Regimes", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "pareto_accuracy_cost.png")
    fig.savefig(out, dpi=200)
    print(f"[ok] {out}")
    plt.close(fig)


def fig_budget_fairness():
    """Budget B5-vs-B10 wall-clock + accuracy bars from validation cells."""
    cells = [
        ("Moderate $\\alpha=0.5$", "comparison_study_20260822_005556"),
        ("IID", "comparison_study_20260822_013747"),
    ]
    rows = []
    for name, d in cells:
        path = os.path.join(PROJECT_ROOT, "outputs", d, "comparison_results.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        pers = df[df["Metric"] == "Personalized"]
        for _, r in pers.iterrows():
            tag = "B5" if "B5" in r["Topology"] else "B10"
            rows.append(dict(regime=name, budget=tag, acc=r["Final Accuracy"], secs=r["Wall Clock Seconds"]))
    if len(rows) < 4:
        print("[skip] budget fairness: validation cells missing")
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    regimes = df["regime"].unique()
    x = np.arange(len(regimes))
    width = 0.35
    for i, b in enumerate(["B10", "B5"]):
        sub = df[df["budget"] == b].set_index("regime").reindex(regimes)
        axes[0].bar(x + (i - 0.5) * width, sub["acc"], width, label=b)
        axes[1].bar(x + (i - 0.5) * width, sub["secs"], width, label=b)
    axes[0].set_ylabel("Final Personalized Accuracy (%)")
    axes[1].set_ylabel("Wall-Clock (s)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[1].legend(fontsize=9)
    fig.suptitle("Compute Fairness: Budget 5 matches Budget 10 at half cost", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "budget_fairness.png")
    fig.savefig(out, dpi=200)
    print(f"[ok] {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="Subset: convergence byzantine pareto budget")
    args = parser.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)

    todo = {
        "convergence": fig_convergence,
        "byzantine": fig_byzantine,
        "pareto": fig_pareto,
        "budget": fig_budget_fairness,
    }
    for name, fn in todo.items():
        if args.only is None or name in args.only:
            try:
                fn()
            except Exception as exc:  # keep generating remaining figures
                print(f"[err] {name}: {exc}")


if __name__ == "__main__":
    main()
