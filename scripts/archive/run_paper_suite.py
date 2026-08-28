"""
Master driver for the full paper experiment suite.
Runs every block sequentially on the verified accelerator, logs per-block
output, tolerates individual block failures (recorded in the manifest), and
supports resume: already-completed blocks are skipped if the manifest exists.

Usage:
    python scripts/run_paper_suite.py            # run pending blocks
    python scripts/run_paper_suite.py --only freeze ablation
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUITE_DIR = os.path.join(PROJECT_ROOT, "outputs", "paper_suite")
MANIFEST_PATH = os.path.join(SUITE_DIR, "manifest.json")
PYTHON = sys.executable

PY = "scripts/run_comparison.py"
MS = "scripts/run_multi_seed.py"

BLOCKS = [
    ("freeze_cells",       f"{PY} --config configs/paper_freeze_cells.yaml"),
    ("ablation",           f"{PY} --config configs/paper_ablation.yaml"),
    ("ksweep",             f"{PY} --config configs/paper_ksweep.yaml"),
    ("convergence",        f"{PY} --config configs/paper_convergence.yaml"),
    ("table3_baselines",   f"{MS} --config configs/paper_table3_baselines.yaml --seeds 42 123 7"),
    ("comparison_seed123", f"{MS} --config configs/comparison.yaml --seeds 123"),
    ("comparison_seed7",   f"{MS} --config configs/comparison.yaml --seeds 7"),
    ("byz_label_flip",     f"{PY} --config configs/paper_byz_label_flip.yaml"),
    ("byz_gaussian",       f"{PY} --config configs/paper_byz_gaussian.yaml"),
    ("cifar100",           f"{PY} --config configs/paper_cifar100.yaml"),
    ("scale50",            f"{PY} --config configs/paper_scale50.yaml"),
    ("mobilenet",          f"{PY} --config configs/paper_mobilenet.yaml"),
]


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only these block names")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if marked complete in manifest")
    args = parser.parse_args()

    os.makedirs(SUITE_DIR, exist_ok=True)
    manifest = load_manifest()
    env = {**os.environ, "HEP_FORCE_DEVICE": "dml"}

    blocks = [b for b in BLOCKS if args.only is None or b[0] in args.only]

    for name, cmd in blocks:
        status = manifest.get(name, {})
        if status.get("state") == "complete" and not args.force:
            print(f"[skip] {name} already complete")
            continue

        log_path = os.path.join(SUITE_DIR, f"{name}.log")
        print(f"[run ] {name}: {cmd}")
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                f"{PYTHON} {cmd}",
                cwd=PROJECT_ROOT, env=env, shell=True,
                stdout=log, stderr=subprocess.STDOUT,
            )
        elapsed = time.time() - t0

        manifest[name] = {
            "state": "complete" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "elapsed_s": round(elapsed, 1),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }
        save_manifest(manifest)
        print(f"[{'done' if proc.returncode == 0 else 'FAIL'}] {name} in {elapsed/60:.1f} min")

    failed = [n for n, s in manifest.items() if s.get("state") == "failed"]
    print("\n==== SUITE SUMMARY ====")
    for name, _ in BLOCKS:
        s = manifest.get(name, {})
        print(f"  {name:24s} {s.get('state', 'pending'):8s} {s.get('elapsed_s', '-')}s")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
