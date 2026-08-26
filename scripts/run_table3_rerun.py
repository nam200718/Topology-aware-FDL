"""Baseline-fidelity re-run driver.

Runs the corrected/new baseline grids (canonical FedBABU, FedALA, CFL, and
Ditto/FedRep/APFL sanity rows) across the five Table III heterogeneity
regimes for one or more seeds, sequentially and resumably by part.

Usage:
    python scripts/run_table3_rerun.py --parts fedbabu fedala cfl sanity \
        --seeds 42 --device privateuseone:0
"""

import argparse
import sys
import os
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

PART_CONFIGS = {
    "fedbabu": "configs/paper_table3_rerun_fedbabu.yaml",
    "fedala": "configs/paper_table3_rerun_fedala.yaml",
    "cfl": "configs/paper_table3_rerun_cfl.yaml",
    "sanity": "configs/paper_table3_rerun_sanity.yaml",
    "sanityfull": "configs/paper_table3_rerun_sanity_full.yaml",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=list(PART_CONFIGS.keys()),
                    choices=sorted(PART_CONFIGS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--rounds", type=int, default=None,
                    help="Override num_rounds (smoke testing)")
    args = ap.parse_args()

    from src.experiments.runner import ExperimentRunner

    for seed in args.seeds:
        for part in args.parts:
            cfg_path = PART_CONFIGS[part]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join("outputs", "baseline_fidelity",
                                   f"seed{seed}_{part}_{stamp}")
            overrides = {"seed": seed, "output_dir": out_dir}
            if args.rounds is not None:
                overrides["num_rounds"] = args.rounds
            print(f"\n{'=' * 60}\n[driver] part={part} seed={seed} -> {out_dir}\n{'=' * 60}",
                  flush=True)
            runner = ExperimentRunner.from_yaml(cfg_path, overrides=overrides)
            runner.run()
            print(f"[driver] finished part={part} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
