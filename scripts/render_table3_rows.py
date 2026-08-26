"""Render corrected Table III baseline rows from baseline-fidelity artifacts.

Scans outputs/baseline_fidelity/seed<seed>_<part>_<stamp>/comparison_study_*/
metrics/<exp>/metrics.json, extracts the personalized last-5 average accuracy
and final-round bottom-10% fairness per (method, scenario), aggregates across
seeds, and prints LaTeX rows in Table III column order.

Usage:
    python scripts/render_table3_rows.py [--root outputs/baseline_fidelity]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

SCENARIOS = [
    ("iid", "IID"),
    ("non_iid_alpha_1.0", "Mild"),
    ("non_iid_alpha_0.5", "Moderate"),
    ("non_iid_alpha_0.1", "Severe"),
    ("non_iid_alpha_0.05", "Extreme"),
]

METHODS = ["fedrep", "ditto", "apfl", "fedala", "fedper", "fedbabu", "cfl"]


def parse_run(metrics_file):
    with open(metrics_file, "r") as f:
        hist = json.load(f)
    evaluated = [h for h in hist if h.get("evaluated") and "ensemble_test_accuracy" in h]
    if not evaluated:
        return None
    last5 = float(np.mean([h["ensemble_test_accuracy"] for h in evaluated[-5:]]))
    final = evaluated[-1]
    b10 = final.get("bottom10_fairness")
    return {"last5": last5, "bot10": b10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("outputs", "baseline_fidelity"))
    ap.add_argument("--seed", type=int, default=None,
                    help="Restrict aggregation to a single seed (default: pool all)")
    args = ap.parse_args()

    # data[method][scenario_id][seed] = {last5, bot10}
    data = defaultdict(lambda: defaultdict(dict))
    pattern = os.path.join(args.root, "seed*", "comparison_study_*", "metrics", "*", "metrics.json")
    for mf in sorted(glob.glob(pattern)):
        m = re.search(r"seed(\d+)_[^\\/]+", mf)
        seed = int(m.group(1)) if m else 0
        if args.seed is not None and seed != args.seed:
            continue
        exp = os.path.basename(os.path.dirname(mf))
        em = re.match(r"star(?:_cfl)?_(fedrep|ditto|apfl|fedala|fedper|fedbabu|cfl)_(.+)$", exp)
        if not em:
            continue
        method, scen = em.group(1), em.group(2)
        if scen.startswith("non_iid_"):
            pass
        elif scen == "iid":
            pass
        else:
            continue
        res = parse_run(mf)
        if res is None:
            continue
        data[method][scen][seed] = res

    def cell(vals, key):
        xs = [v[key] for v in vals if v.get(key) is not None]
        if not xs:
            return "---"
        if len(xs) == 1:
            return f"{xs[0]:.2f}%"
        return f"{np.mean(xs):.2f} $\\pm$ {np.std(xs):.2f}%"

    for method in METHODS:
        if method not in data:
            continue
        cells = []
        complete = True
        for scen, _ in SCENARIOS:
            runs = data[method].get(scen, {})
            if not runs:
                cells.append("--- & ---")
                complete = False
                continue
            seeds = sorted(runs.keys())
            vals = [runs[s] for s in seeds]
            cells.append(f"{cell(vals, 'last5')} & {cell(vals, 'bot10')}")
            if len(seeds) < 3 or any(runs[s].get("bot10") is None for s in seeds):
                complete = False
        name = {"fedrep": "FedRep", "ditto": "Ditto", "apfl": "APFL",
                "fedala": "FedALA", "fedper": "FedPer", "fedbabu": "FedBABU",
                "cfl": "CFL"}[method]
        row = " & ".join(cells)
        status = "" if complete else "   % INCOMPLETE (missing seeds/bottom10)"
        print(f"% {name}{status}\n& {row} & \\\\\n")


if __name__ == "__main__":
    main()
