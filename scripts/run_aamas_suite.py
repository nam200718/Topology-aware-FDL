"""
AAMAS 2027 Unified Experimental Suite Orchestrator.
Primary Execution Engine: src.baselines

Modular Research Jobs:
    1. Personalization & Heterogeneity Benchmark (CIFAR-100, 5 Regimes, 3 Seeds, 7 Methods)
    2. Multi-Attack Byzantine Robustness Benchmark (4 Attack Types, 5 Attack Rates, 4 Methods)
    3. 50-Client Scalability Benchmark with Partial Participation (N=50, Cp=0.20)
    4. MobileNetV3 Physical Hardware & Edge Footprint Profiling
    finalize: Assemble all publication-ready LaTeX tables & figures

Usage:
    python scripts/run_aamas_suite.py --job 1             # Run Job 1 (Personalization)
    python scripts/run_aamas_suite.py --job 2             # Run Job 2 (Byzantine)
    python scripts/run_aamas_suite.py --job 3             # Run Job 3 (50-Client Scale)
    python scripts/run_aamas_suite.py --job 4             # Run Job 4 (MobileNetV3 Edge)
    python scripts/run_aamas_suite.py --job finalize      # Assemble all LaTeX tables
    python scripts/run_aamas_suite.py --job all           # End-to-end execution
    python scripts/run_aamas_suite.py --smoke-test        # Fast 1-round verification across suite
"""

import os
import sys
import json
import time
import argparse
from types import SimpleNamespace

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.baselines.factory import detect_accelerator
from src.baselines.experiment_configs import CIFAR100_DEFAULTS
from src.baselines.run_all_baselines import run_personalization_suite, run_byzantine_suite, run_ablation_suite, _parse_list_arg

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


def run_job_1(args, force: bool = False):
    """Job 1: CIFAR-100 Personalization & Heterogeneity Benchmark across 5 Regimes."""
    print("\n" + "=" * 80)
    print("JOB 1: CIFAR-100 PERSONALIZATION & HETEROGENEITY BENCHMARK (src.baselines)")
    print("=" * 80)
    manifest = load_manifest()
    if "job_1" in manifest["completed_jobs"] and not force and not args.smoke_test:
        print(">>> Job 1 already recorded as completed in manifest. Skipping.")
        return

    t0 = time.time()
    device = args.device or detect_accelerator()
    seeds = _parse_list_arg(args.seeds, int) or [42, 123, 7]

    base_defaults = dict(CIFAR100_DEFAULTS)
    base_defaults["dataset"] = args.dataset
    base_defaults["data_dir"] = getattr(args, "data_dir", "./data")
    if args.rounds is not None:
        base_defaults["num_rounds"] = args.rounds
    if args.clients is not None:
        base_defaults["num_clients"] = args.clients

    if args.smoke_test:
        seeds = [42]
        base_defaults["num_rounds"] = 1
        base_defaults["eval_interval"] = 1
        base_defaults["train_subset"] = 200
        base_defaults["test_subset"] = 50

    sub_args = SimpleNamespace(
        methods=args.methods,
        regimes=args.regimes,
        output_dir=os.path.join(_project_root, "outputs", "baselines"),
        force=args.force,
    )

    run_personalization_suite(sub_args, device, seeds, base_defaults)

    dur = time.time() - t0
    if not args.smoke_test:
        update_manifest("job_1", dur)
    print(f"\n[JOB 1 COMPLETE] Time elapsed: {dur:.1f}s")


def run_job_2(args, force: bool = False):
    """Job 2: Byzantine Multi-Attack Robustness Suite."""
    print("\n" + "=" * 80)
    print("JOB 2: CIFAR-100 BYZANTINE MULTI-ATTACK ROBUSTNESS BENCHMARK (src.baselines)")
    print("=" * 80)
    manifest = load_manifest()
    if "job_2" in manifest["completed_jobs"] and not force and not args.smoke_test:
        print(">>> Job 2 already recorded as completed in manifest. Skipping.")
        return

    t0 = time.time()
    device = args.device or detect_accelerator()
    seeds = _parse_list_arg(args.seeds, int) or [42, 123, 7]

    base_defaults = dict(CIFAR100_DEFAULTS)
    base_defaults["dataset"] = args.dataset
    base_defaults["data_dir"] = getattr(args, "data_dir", "./data")
    if args.rounds is not None:
        base_defaults["num_rounds"] = args.rounds
    if args.clients is not None:
        base_defaults["num_clients"] = args.clients

    if args.smoke_test:
        seeds = [42]
        base_defaults["num_rounds"] = 1
        base_defaults["eval_interval"] = 1
        base_defaults["train_subset"] = 200
        base_defaults["test_subset"] = 50

    sub_args = SimpleNamespace(
        methods=args.methods,
        attacks=args.attacks,
        rates=args.rates,
        output_dir=os.path.join(_project_root, "outputs", "baselines"),
        force=args.force,
    )

    run_byzantine_suite(sub_args, device, seeds, base_defaults)

    dur = time.time() - t0
    if not args.smoke_test:
        update_manifest("job_2", dur)
    print(f"\n[JOB 2 COMPLETE] Time elapsed: {dur:.1f}s")


def run_job_3(args, force: bool = False):
    """Job 3: 50-Client Population Scaling with Partial Participation."""
    print("\n" + "=" * 80)
    print("JOB 3: 50-CLIENT POPULATION SCALING & FAIRNESS BENCHMARK")
    print("=" * 80)
    manifest = load_manifest()
    if "job_3" in manifest["completed_jobs"] and not force and not args.smoke_test:
        print(">>> Job 3 already recorded as completed in manifest. Skipping.")
        return

    t0 = time.time()
    from scripts.run_scale_50clients import run_50clients_scaling
    rounds = 1 if args.smoke_test else (args.rounds or 20)
    sub = 100 if args.smoke_test else None
    run_50clients_scaling(
        num_clients=50,
        clients_per_round=10,
        num_rounds=rounds,
        batch_size=64,
        device=args.device,
        train_subset=sub,
        data_dir=getattr(args, "data_dir", "./data"),
    )

    dur = time.time() - t0
    if not args.smoke_test:
        update_manifest("job_3", dur)
    print(f"\n[JOB 3 COMPLETE] Time elapsed: {dur:.1f}s")


def run_job_4(args, force: bool = False):
    """Job 4: MobileNetV3 Edge Vision Footprint Profiling."""
    print("\n" + "=" * 80)
    print("JOB 4: MOBILENETV3 PHYSICAL EDGE FOOTPRINT PROFILING")
    print("=" * 80)
    manifest = load_manifest()
    if "job_4" in manifest["completed_jobs"] and not force and not args.smoke_test:
        print(">>> Job 4 already recorded as completed in manifest. Skipping.")
        return

    t0 = time.time()
    from scripts.run_mobilenet_benchmark import run_mobilenet_benchmark
    rounds = 1 if args.smoke_test else (args.rounds or 15)
    sub = 100 if args.smoke_test else None
    run_mobilenet_benchmark(
        num_clients=15,
        num_rounds=rounds,
        batch_size=32,
        device=args.device,
        train_subset=sub,
        data_dir=getattr(args, "data_dir", "./data"),
    )

    dur = time.time() - t0
    if not args.smoke_test:
        update_manifest("job_4", dur)
    print(f"\n[JOB 4 COMPLETE] Time elapsed: {dur:.1f}s")


def run_job_5(args, force: bool = False):
    """Job 5: Ablation Study & Cluster Valuation Benchmark (src.baselines)."""
    print("\n" + "=" * 80)
    print("JOB 5: CIFAR-100 ABLATION & CLUSTER VALUATION BENCHMARK (src.baselines)")
    print("=" * 80)
    manifest = load_manifest()
    if "job_5" in manifest["completed_jobs"] and not force and not args.smoke_test:
        print(">>> Job 5 already recorded as completed in manifest. Skipping.")
        return

    t0 = time.time()
    device = args.device or detect_accelerator()
    seeds = _parse_list_arg(args.seeds, int) or [42, 123, 7]

    base_defaults = dict(CIFAR100_DEFAULTS)
    base_defaults["dataset"] = args.dataset
    base_defaults["data_dir"] = getattr(args, "data_dir", "./data")
    if args.rounds is not None:
        base_defaults["num_rounds"] = args.rounds
    if args.clients is not None:
        base_defaults["num_clients"] = args.clients

    if args.smoke_test:
        seeds = [42]
        base_defaults["num_rounds"] = 1
        base_defaults["eval_interval"] = 1
        base_defaults["train_subset"] = 200
        base_defaults["test_subset"] = 50

    sub_args = SimpleNamespace(
        methods=args.methods,
        regimes=args.regimes,
        output_dir=os.path.join(_project_root, "outputs", "baselines"),
        force=args.force,
    )

    run_ablation_suite(sub_args, device, seeds, base_defaults)

    dur = time.time() - t0
    if not args.smoke_test:
        update_manifest("job_5", dur)
    print(f"\n[JOB 5 COMPLETE] Time elapsed: {dur:.1f}s")


def run_finalize():
    """Finalize: Generate all publication-ready LaTeX tables and high-resolution figures."""
    print("\n" + "=" * 80)
    print("FINALIZING PUBLICATION ARTIFACTS: LATEX TABLES & FIGURES")
    print("=" * 80)

    try:
        from scripts.generate_all_tables import main as gen_tables
        gen_tables()
    except Exception as e:
        print(f"[warn] Failed running generate_all_tables: {e}")

    try:
        from scripts.make_paper_figures import main as gen_figures
        gen_figures(argv=[])
    except Exception as e:
        print(f"[warn] Failed running make_paper_figures: {e}")

    print("\n[FINALIZE COMPLETE] All tables and figures updated.")


def main():
    parser = argparse.ArgumentParser(description="AAMAS 2027 Unified Paper Suite Orchestrator")
    parser.add_argument(
        "--job",
        "--track",
        dest="job",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "5", "personalization", "byzantine", "scale50", "edge", "ablation", "finalize", "all"],
        help="Job to run: 1 (personalization), 2 (byzantine), 3 (scale50), 4 (edge), 5 (ablation), finalize, or all",
    )
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to run (space or comma separated)")
    parser.add_argument("--regimes", nargs="+", default=None, help="Regimes to run (space or comma separated)")
    parser.add_argument("--attacks", nargs="+", default=None, help="Attacks to run (space or comma separated)")
    parser.add_argument("--rates", nargs="+", default=None, help="Rates to run (space or comma separated)")
    parser.add_argument("--seeds", nargs="+", default=["42", "123", "7"], help="Random seeds (space or comma separated)")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "cifar10", "synthetic", "mnist"])
    parser.add_argument("--rounds", type=int, default=None, help="Override communication rounds")
    parser.add_argument("--clients", type=int, default=None, help="Override client count")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cuda', 'cpu')")
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset directory or Kaggle input mount (e.g. /kaggle/input/cifar100)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if manifest marks completed")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-round smoke test across designated jobs")
    args = parser.parse_args()

    os.makedirs(os.path.join(_project_root, "outputs"), exist_ok=True)

    t_start = time.time()

    job = args.job.lower()
    if job in ("1", "personalization", "all"):
        run_job_1(args, force=args.force)
    if job in ("2", "byzantine", "all"):
        run_job_2(args, force=args.force)
    if job in ("3", "scale50", "all"):
        run_job_3(args, force=args.force)
    if job in ("4", "edge", "all"):
        run_job_4(args, force=args.force)
    if job in ("5", "ablation", "all"):
        run_job_5(args, force=args.force)
    if job in ("finalize", "all"):
        run_finalize()

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"SUITE EXECUTION FINISHED IN {total_time/60.0:.2f} MINUTES")
    print("=" * 80)


if __name__ == "__main__":
    main()
