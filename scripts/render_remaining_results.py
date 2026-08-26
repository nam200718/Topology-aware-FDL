"""Summarize remaining-queue artifacts into review-ready tables.

Walks outputs/remaining_queue/**/metrics/**/metrics.json and prints:
  * comparison-type runs  -> personalized Last-5 accuracy (+bottom-10 when present)
  * byzantine-matrix runs -> Last-5 accuracy per byzantine fraction
  * convergence runs      -> round-by-round personalized trajectory

Usage:
    python scripts/render_remaining_results.py [--root outputs/remaining_queue] [--seed 42]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def load(path):
    with open(path, "r") as f:
        return json.load(f)


def summarize_comparison(exp, hist):
    evaluated = [h for h in hist if h.get("evaluated") and "ensemble_test_accuracy" in h]
    if not evaluated:
        return None
    last5 = float(np.mean([h["ensemble_test_accuracy"] for h in evaluated[-5:]]))
    b10 = evaluated[-1].get("bottom10_fairness")
    return {"last5": last5, "bot10": b10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("outputs", "remaining_queue"))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    comparison = defaultdict(dict)   # (part, exp) -> {last5, bot10}
    byz = defaultdict(dict)          # (part, label) -> {rate: acc}
    conv = defaultdict(list)         # (part, exp) -> [(round, acc)]

    pattern = os.path.join(args.root, "**", "metrics", "**", "metrics.json")
    for mf in sorted(glob.glob(pattern, recursive=True)):
        norm = mf.replace("\\", "/")
        m = re.search(r"seed(\d+)_[^/]+", norm)
        if args.seed is not None and m and int(m.group(1)) != args.seed:
            continue
        try:
            hist = load(mf)
        except Exception:
            continue
        if not isinstance(hist, list) or not hist:
            continue

        # experiment directory name: .../metrics/<exp>/metrics.json
        exp_dir = os.path.basename(os.path.dirname(mf))

        if re.search(r"_byz_(\d+)$", exp_dir):
            rate_m = re.search(r"_byz_(\d+)$", exp_dir)
            rate = int(rate_m.group(1))
            part_m = re.search(r"/([^/]+)/comparison_study", norm) or \
                re.search(r"/([^/]+)/byzantine_matrix", norm)
            part = part_m.group(1) if part_m else "?"
            label = re.sub(r"_byz_\d+$", "", exp_dir)
            evaluated = [h for h in hist if h.get("evaluated") and "ensemble_test_accuracy" in h]
            if evaluated:
                byz[(part, label)][rate] = float(
                    np.mean([h["ensemble_test_accuracy"] for h in evaluated[-5:]]))
            continue

        res = summarize_comparison(exp_dir, hist)
        if res is None:
            continue
        part_m = re.search(r"/([^/]+)/comparison_study", norm)
        part = part_m.group(1) if part_m else "?"
        comparison[(part, exp_dir)] = res
        if exp_dir.startswith("conv_"):
            traj = [(h["round"], h["ensemble_test_accuracy"]) for h in hist
                    if h.get("evaluated") and "ensemble_test_accuracy" in h]
            conv[(part, exp_dir)] = traj

    print("=" * 70)
    print("COMPARISON RUNS")
    print("=" * 70)
    for (part, exp), r in sorted(comparison.items()):
        b10 = f"{r['bot10']:.2f}" if r.get("bot10") is not None else "---"
        print(f"{part:28s} {exp:44s} last5={r['last5']:6.2f}  b10={b10}")

    print("\n" + "=" * 70)
    print("BYZANTINE RUNS (Last-5 personalized acc)")
    print("=" * 70)
    for (part, label), rates in sorted(byz.items()):
        cells = "  ".join(f"f={r:>2}%: {rates[r]:6.2f}" for r in sorted(rates))
        print(f"{part:22s} {label:34s} {cells}")

    print("\n" + "=" * 70)
    print("CONVERGENCE TRAJECTORIES (round: acc)")
    print("=" * 70)
    for (part, exp), traj in sorted(conv.items()):
        s = ", ".join(f"{rnd}:{acc:.1f}" for rnd, acc in traj)
        print(f"{exp}\n  {s}")


if __name__ == "__main__":
    main()
