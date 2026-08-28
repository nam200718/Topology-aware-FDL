"""Tier A queue driver for the 24h remediation plan.

Sequential runs on DirectX GPU (seed 42):
  1. configs/comparison.yaml (Table III main benchmark)
  2. configs/paper_ablation.yaml + configs/paper_ablation_moderate.yaml (Table VI ablations)
  3. configs/paper_k1_cert.yaml (Table VII K=1 certification)
  4. configs/paper_byz_label_flip.yaml (Table V Byzantine label-flip matrix)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.experiments.runner import ExperimentRunner

TIER_A_BLOCKS = [
    ("comparison", "configs/comparison.yaml", "outputs/tier_a_rerun/comparison"),
    ("ablation", "configs/paper_ablation.yaml", "outputs/tier_a_rerun/ablation"),
    ("ablation_moderate", "configs/paper_ablation_moderate.yaml", "outputs/tier_a_rerun/ablation_moderate"),
    ("k1_cert", "configs/paper_k1_cert.yaml", "outputs/tier_a_rerun/k1_cert"),
    ("byz_label_flip", "configs/paper_byz_label_flip.yaml", "outputs/tier_a_rerun/byz_label_flip"),
]

MANIFEST_PATH = os.path.join(_PROJECT_ROOT, "outputs", "tier_a_rerun", "manifest.json")


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run Tier A experiment queue")
    parser.add_argument("--only", nargs="*", default=None, help="Run only specific block names")
    parser.add_argument("--force", action="store_true", help="Force rerun even if marked complete")
    args = parser.parse_args()

    manifest = load_manifest()
    blocks = [b for b in TIER_A_BLOCKS if args.only is None or b[0] in args.only]

    print(f"Starting Tier A execution queue ({len(blocks)} blocks)...", flush=True)

    for name, cfg_path, out_dir in blocks:
        status = manifest.get(name, {})
        if status.get("state") == "complete" and not args.force:
            print(f"[skip] {name} already complete.", flush=True)
            continue

        print(f"\n{'='*60}\n[RUNNING] {name} ({cfg_path} -> {out_dir})\n{'='*60}", flush=True)
        t0 = time.time()
        os.makedirs(os.path.join(_PROJECT_ROOT, out_dir), exist_ok=True)

        try:
            full_cfg_path = os.path.join(_PROJECT_ROOT, cfg_path)
            runner = ExperimentRunner.from_yaml(full_cfg_path, overrides={"seed": 42, "output_dir": os.path.join(_PROJECT_ROOT, out_dir)})
            runner.run()
            elapsed = time.time() - t0
            manifest[name] = {
                "state": "complete",
                "config": cfg_path,
                "output_dir": out_dir,
                "elapsed_s": round(elapsed, 1),
                "finished_at": datetime.now().isoformat(timespec="seconds")
            }
            save_manifest(manifest)
            print(f"[SUCCESS] {name} completed in {elapsed/60:.2f} min.", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            manifest[name] = {
                "state": "failed",
                "config": cfg_path,
                "error": str(e),
                "elapsed_s": round(elapsed, 1),
                "finished_at": datetime.now().isoformat(timespec="seconds")
            }
            save_manifest(manifest)
            print(f"[FAILED] {name} error: {e}", flush=True)

    print("\n==== TIER A QUEUE SUMMARY ====", flush=True)
    for name, _, _ in TIER_A_BLOCKS:
        s = manifest.get(name, {})
        print(f"  {name:24s} {s.get('state', 'pending'):10s} {s.get('elapsed_s', '-')}s", flush=True)


if __name__ == "__main__":
    main()
