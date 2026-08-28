"""Remaining-work queue driver: corrected-baseline re-runs for the paper's
secondary tables (byzantine, CIFAR-100, convergence, MobileNetV3), the K=1
bipartite certification grid, and smoke support.

Usage:
    python scripts/run_remaining_queue.py --parts k1cert convfig --rounds 2
    python scripts/run_remaining_queue.py --parts all
"""

import argparse
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

PART_CONFIGS = {
    "k1cert": "configs/paper_k1_cert.yaml",
    "convfig": "configs/paper_convergence.yaml",
    "byzflip": "configs/paper_byz_label_flip.yaml",
    "byzgauss": "configs/paper_byz_gaussian.yaml",
    "byzsign": "configs/paper_byz_signflip.yaml",
    "c100fix": "configs/paper_cifar100.yaml",
    "mnnet": "configs/paper_mobilenet.yaml",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["all"],
                    choices=["all"] + sorted(PART_CONFIGS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--rounds", type=int, default=None,
                    help="Override num_rounds (smoke testing)")
    args = ap.parse_args()

    parts = sorted(PART_CONFIGS.keys()) if "all" in args.parts else args.parts

    from src.experiments.runner import ExperimentRunner

    for seed in args.seeds:
        for part in parts:
            cfg_path = PART_CONFIGS[part]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join("outputs", "remaining_queue",
                                   f"seed{seed}_{part}_{stamp}")
            overrides = {"seed": seed, "output_dir": out_dir}
            if args.rounds is not None:
                overrides["num_rounds"] = args.rounds
            print(f"\n{'=' * 60}\n[queue] part={part} seed={seed} -> {out_dir}\n{'=' * 60}",
                  flush=True)
            runner = ExperimentRunner.from_yaml(cfg_path, overrides=overrides)
            runner.run()
            print(f"[queue] finished part={part} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
