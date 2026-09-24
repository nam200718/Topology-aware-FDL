"""
AAMAS 2027 Unified Experimental Suite Orchestrator.
Runs the 4 modular research jobs on Google Colab (Free T4 GPU) or local machine.

Usage:
    python scripts/run_aamas_suite.py --job 1         # Job 1: CIFAR-10 5 Regimes x 3 Seeds
    python scripts/run_aamas_suite.py --job 2         # Job 2: Multi-Attack Byzantine Robustness
    python scripts/run_aamas_suite.py --job 3         # Job 3: CIFAR-100 & N=50 Scale
    python scripts/run_aamas_suite.py --job 4         # Job 4: MobileNetV3 Edge & Sensor Regression
    python scripts/run_aamas_suite.py --job finalize  # Assemble all LaTeX tables & figures
    python scripts/run_aamas_suite.py --job all       # Run entire pipeline end-to-end
"""

import os
import sys
import json
import time
import argparse
import subprocess

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

MANIFEST_FILE = os.path.join(_project_root, "outputs", "aamas_manifest.json")


def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_jobs": [], "timestamps": {}}


def update_manifest(job_name: str, duration: float):
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    manifest = load_manifest()
    if job_name not in manifest["completed_jobs"]:
        manifest["completed_jobs"].append(job_name)
    manifest["timestamps"][job_name] = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(duration, 2)
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def run_job_1(force: bool = False):
    """Job 1: CIFAR-100 5 Regimes Benchmark across Partition Values (IID, Mild, Moderate, Severe, Extreme)."""
    print("\n" + "=" * 80)
    print("JOB 1: CIFAR-100 5-REGIME BENCHMARK (C = 100, 5 Heterogeneity Regimes)")
    print("=" * 80)
    t0 = time.time()
    manifest = load_manifest()
    if "job_1" in manifest["completed_jobs"] and not force:
        print(">>> Job 1 already recorded as completed in manifest. Skipping.")
        return

    from scripts.run_cifar100_benchmark import run_cifar100_multiregime_experiment
    run_cifar100_multiregime_experiment(num_clients=15, num_rounds=20, batch_size=64)

    dur = time.time() - t0
    update_manifest("job_1", dur)
    print(f"\n[JOB 1 COMPLETE] Time elapsed: {dur:.1f}s")


def run_job_2(force: bool = False):
    """Job 2: CIFAR-100 Byzantine Multi-Attack Robustness Suite (Label-flipping & Sign-flipping)."""
    print("\n" + "=" * 80)
    print("JOB 2: CIFAR-100 BYZANTINE MULTI-ATTACK ROBUSTNESS & SKEW-CALIBRATED DEFENSE")
    print("=" * 80)
    t0 = time.time()
    manifest = load_manifest()
    if "job_2" in manifest["completed_jobs"] and not force:
        print(">>> Job 2 already recorded as completed in manifest. Skipping.")
        return

    from scripts.run_cifar100_benchmark import run_cifar100_byzantine_suite
    results = run_cifar100_byzantine_suite(num_clients=15, num_rounds=15, batch_size=64)

    dur = time.time() - t0
    update_manifest("job_2", dur)
    print(f"\n[JOB 2 COMPLETE] Time elapsed: {dur:.1f}s")


def run_job_3(force: bool = False):
    """Job 3: CIFAR-100 High-Cardinality (C=100) & N=50 Client Population Scaling."""
    print("\n" + "=" * 80)
    print("JOB 3: SCALE BENCHMARK (CIFAR-100 C=100 & N=50 Client Population)")
    print("=" * 80)
    t0 = time.time()
    manifest = load_manifest()
    if "job_3" in manifest["completed_jobs"] and not force:
        print(">>> Job 3 already recorded as completed in manifest. Skipping.")
        return

    from scripts.run_cifar100_benchmark import run_cifar100_experiment
    from scripts.run_scale_50clients import run_50clients_scaling

    print(">>> Subtask 3A: Running CIFAR-100 Benchmark...")
    run_cifar100_experiment(num_clients=15, num_rounds=20, batch_size=64)

    print(">>> Subtask 3B: Running N=50 Client Scaling Benchmark...")
    run_50clients_scaling(num_clients=50, clients_per_round=10, num_rounds=20, batch_size=64)

    dur = time.time() - t0
    update_manifest("job_3", dur)
    print(f"\n[JOB 3 COMPLETE] Time elapsed: {dur:.1f}s")


def run_job_4(force: bool = False):
    """Job 4: MobileNetV3 Edge Vision & Continuous Sensor Regression Generalization."""
    print("\n" + "=" * 80)
    print("JOB 4: PHYSICAL EDGE PROFILING & TASK GENERALIZATION")
    print("=" * 80)
    t0 = time.time()
    manifest = load_manifest()
    if "job_4" in manifest["completed_jobs"] and not force:
        print(">>> Job 4 already recorded as completed in manifest. Skipping.")
        return

    from scripts.run_mobilenet_benchmark import run_mobilenet_benchmark
    from scripts.demo_regression_generalization import run_regression_experiment

    print(">>> Subtask 4A: Profiling MobileNetV3-Small on Edge Hardware Footprint...")
    run_mobilenet_benchmark(num_clients=15, num_rounds=15, batch_size=32)

    print(">>> Subtask 4B: Continuous Multi-Agent Sensor Regression Benchmark...")
    run_regression_experiment()

    dur = time.time() - t0
    update_manifest("job_4", dur)
    print(f"\n[JOB 4 COMPLETE] Time elapsed: {dur:.1f}s")


def run_finalize():
    """Finalize: Generate all publication-ready LaTeX tables and high-resolution figures."""
    print("\n" + "=" * 80)
    print("FINALIZING PUBLICATION ARTIFACTS: LATEX TABLES & FIGURES")
    print("=" * 80)

    try:
        from scripts.generate_all_tables import main as gen_tables
        gen_tables()
    except Exception as e:
        print(f"Warning running generate_all_tables: {e}")

    try:
        from scripts.make_paper_figures import main as gen_figures
        gen_figures()
    except Exception as e:
        print(f"Warning running make_paper_figures: {e}")

    print("\n[FINALIZE COMPLETE] All tables and figures updated.")


def main():
    parser = argparse.ArgumentParser(description="AAMAS 2027 Unified Paper Suite Orchestrator")
    parser.add_argument("--job", type=str, default="all", choices=["1", "2", "3", "4", "finalize", "all"],
                        help="Job to run: 1, 2, 3, 4, finalize, or all")
    parser.add_argument("--force", action="store_true", help="Force re-run even if manifest marks completed")
    args = parser.parse_args()

    os.makedirs(os.path.join(_project_root, "outputs"), exist_ok=True)

    t_start = time.time()

    if args.job in ["1", "all"]:
        run_job_1(force=args.force)
    if args.job in ["2", "all"]:
        run_job_2(force=args.force)
    if args.job in ["3", "all"]:
        run_job_3(force=args.force)
    if args.job in ["4", "all"]:
        run_job_4(force=args.force)
    if args.job in ["finalize", "all"]:
        run_finalize()

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"SUITE EXECUTION FINISHED IN {total_time/60.0:.2f} MINUTES")
    print("=" * 80)


if __name__ == "__main__":
    main()
