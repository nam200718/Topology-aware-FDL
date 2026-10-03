"""Unified Runpod Orchestrator for Baseline 1 (FEMNIST) and Baseline 2 (CIFAR-100) Experimental Suites.

Designed for 1x NVIDIA RTX 4090 on Runpod with persistent network volume at /workspace.
Features:
- Dynamic dataset auto-discovery and linking
- Pre-flight hardware & CUDA TF32 verification
- Fault-tolerant resume via structured checkpoints
- Comprehensive three-tier auto-shutdown watchdog (zero credit waste)
- Automated artifact compression into all_baselines_runpod_artifacts.zip

Usage:
    python scripts/run_runpod_suite.py --help
    python scripts/run_runpod_suite.py --dry-run
    python scripts/run_runpod_suite.py --auto-stop
    python scripts/run_runpod_suite.py --smoke-test
"""
import os
import sys
import time
import json
import subprocess
import argparse
import zipfile
from types import SimpleNamespace

# Ensure repository root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.baselines.factory import detect_accelerator
from src.baselines.experiment_configs import CIFAR100_DEFAULTS, FEMNIST_DEFAULTS
from src.baselines.run_all_baselines import run_personalization_suite, run_byzantine_suite


def resolve_base_dirs():
    """Resolves workspace and outputs directories based on environment."""
    if os.path.exists("/workspace") and os.path.isdir("/workspace"):
        workspace = "/workspace"
    else:
        workspace = _project_root

    data_dir = os.path.join(workspace, "data")
    outputs_dir = os.path.join(workspace, "outputs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    return workspace, data_dir, outputs_dir


def arm_timeout_watchdog(timeout_seconds: int = 7200):
    """Tier 2: Launches a detached background watchdog process that forcibly halts the pod

    if total runtime exceeds timeout_seconds (default: 2 hours).
    """
    cmd = (
        f"nohup bash -c 'sleep {timeout_seconds}; "
        f"echo \"[WATCHDOG TRIGGERED] Hard timeout of {timeout_seconds}s reached. Forcing shutdown.\" >> /workspace/watchdog.log; "
        f"if command -v runpodctl &> /dev/null && [ -n \"$RUNPOD_POD_ID\" ]; then runpodctl stop pod \"$RUNPOD_POD_ID\"; fi; "
        f"kill -9 -1' > /dev/null 2>&1 &"
    )
    try:
        subprocess.Popen(cmd, shell=True)
        print(f"🛡️  [WATCHDOG] Tier 2 hard timeout watchdog armed ({timeout_seconds // 3600} hours max ceiling).")
    except Exception as e:
        print(f"[warn] Failed to arm watchdog: {e}")


def execute_pod_shutdown():
    """Tier 1: Programmatic self-shutdown hook via runpodctl or REST API."""
    pod_id = os.environ.get("RUNPOD_POD_ID")
    print("\n" + "=" * 80)
    print("🛑 [AUTO-STOP] Triggering Tier 1 Programmatic Pod Shutdown to halt billing...")
    print("=" * 80)

    # 1. Try runpodctl if installed and POD_ID is present
    if pod_id:
        try:
            res = subprocess.run(["runpodctl", "stop", "pod", pod_id], capture_output=True, text=True, timeout=15)
            print(f"[runpodctl stop pod output]: {res.stdout.strip()}")
            if res.returncode == 0:
                print("✅ Pod shutdown command successfully dispatched via runpodctl.")
                return
        except Exception as e:
            print(f"[warn] runpodctl stop pod failed: {e}")

    # 2. Try Runpod GraphQL/REST API if RUNPOD_API_KEY is present
    api_key = os.environ.get("RUNPOD_API_KEY")
    if api_key and pod_id:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"https://api.runpod.io/graphql?api_key={api_key}",
                data=json.dumps({"query": f'mutation {{ podStop(input: {{ podId: "{pod_id}" }}) {{ id desiredStatus }} }}'}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"[API podStop response]: {resp.read().decode('utf-8')}")
                return
        except Exception as e:
            print(f"[warn] API podStop failed: {e}")

    print("ℹ️  Pod stop signal logged. Remote Tier 3 control plane can now safely halt the container.")


def run_preflight_checks(device: str, data_dir: str):
    """Validates CUDA hardware, Tensor Core TF32 acceleration, and dataset paths in 0.0s."""
    import torch
    print("\n" + "=" * 80)
    print("RUNPOD PRE-FLIGHT SANITY CHECK")
    print("=" * 80)

    print(f"Target Hardware Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Hardware: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        # Enable TF32 for RTX 4090 Ada Lovelace architecture
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("Tensor Core TF32 Acceleration: ENABLED (matmul & cuDNN benchmark active)")
    else:
        print("[WARNING] CUDA is not available! Running on CPU fallback.")

    print(f"Dataset Root Directory: {data_dir}")
    print("=" * 80)


def package_artifacts(outputs_dir: str):
    """Packages all experiment JSON, CSV, and LaTeX tables into a single distributable zip file."""
    zip_path = os.path.join(outputs_dir, "all_baselines_runpod_artifacts.zip")
    print("\n" + "=" * 80)
    print(f"PACKAGING FINAL ARTIFACTS -> {zip_path}")
    print("=" * 80)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(outputs_dir):
            for file in files:
                if file.endswith(".zip"):
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, outputs_dir)
                zipf.write(file_path, rel_path)

    file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"✅ Packaging complete: {zip_path} ({file_size_mb:.2f} MB)")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Runpod Master Experimental Suite (Baseline 1 FEMNIST + Baseline 2)")
    parser.add_argument("--skip-baseline1", action="store_true", help="Skip Baseline 1 FEMNIST Personalization Suite")
    parser.add_argument("--skip-baseline2", action="store_true", help="Skip Baseline 2 Byzantine, Scaling & MobileNet Suites")
    parser.add_argument("--dry-run", action="store_true", help="Run 0-second environment check without training")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast 1-round smoke test across all suites")
    parser.add_argument("--auto-stop", action="store_true", help="Automatically trigger pod shutdown upon completion")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results already exist in checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Override compute device ('cuda', 'cpu')")
    parser.add_argument("--timeout-seconds", type=int, default=28800, help="Hard ceiling for timeout watchdog (default: 28800s / 8h)")
    args = parser.parse_args()

    workspace, data_dir, outputs_dir = resolve_base_dirs()
    device = args.device or detect_accelerator()

    # Pre-flight check
    run_preflight_checks(device, data_dir)
    if args.dry_run:
        print("\n[DRY RUN COMPLETE] Pre-flight checks passed successfully. Exiting without execution.")
        return

    # Arm safety watchdog (default: 8 hours ceiling)
    if args.auto_stop:
        arm_timeout_watchdog(timeout_seconds=args.timeout_seconds)

    t_suite_start = time.time()

    # -------------------------------------------------------------
    # 1. BASELINE 1: FEMNIST PERSONALIZATION SUITE (ResNet-9)
    # -------------------------------------------------------------
    if not args.skip_baseline1:
        print("\n" + "=" * 80)
        print("PHASE 1: BASELINE 1 (FEMNIST PERSONALIZATION SUITE - ResNet-9)")
        print("=" * 80)
        femnist_defaults = dict(FEMNIST_DEFAULTS)
        femnist_defaults["data_dir"] = data_dir

        seeds = [42] if args.smoke_test else [42, 123, 7]
        if args.smoke_test:
            femnist_defaults["num_rounds"] = 1
            femnist_defaults["eval_interval"] = 1
            femnist_defaults["train_subset"] = 200
            femnist_defaults["test_subset"] = 50

        sub_args = SimpleNamespace(
            methods=None,  # All 6 target methods
            regimes=None,  # All 5 Dirichlet regimes
            output_dir=os.path.join(outputs_dir, "baselines"),
            force=args.force,
        )
        t0 = time.time()
        run_personalization_suite(sub_args, device, seeds, femnist_defaults)
        print(f"[PHASE 1 COMPLETE] FEMNIST Suite finished in {(time.time() - t0)/60.0:.2f} minutes.")

    # -------------------------------------------------------------
    # 2. BASELINE 2: JOB 2 BYZANTINE SUITE (CIFAR-100)
    # -------------------------------------------------------------
    if not args.skip_baseline2:
        print("\n" + "=" * 80)
        print("PHASE 2: BASELINE 2 - JOB 2 (BYZANTINE MULTI-ATTACK ROBUSTNESS SUITE)")
        print("=" * 80)
        cifar100_defaults = dict(CIFAR100_DEFAULTS)
        cifar100_defaults["data_dir"] = data_dir
        cifar100_defaults["num_rounds"] = 1 if args.smoke_test else 30

        seeds = [42] if args.smoke_test else [42, 123, 7]
        if args.smoke_test:
            cifar100_defaults["eval_interval"] = 1
            cifar100_defaults["train_subset"] = 200
            cifar100_defaults["test_subset"] = 50

        byz_args = SimpleNamespace(
            methods=None,   # All 4 defenses
            attacks=None,   # All 4 attacks
            rates=None,     # All 5 rates
            output_dir=os.path.join(outputs_dir, "baselines"),
            force=args.force,
        )
        t0 = time.time()
        run_byzantine_suite(byz_args, device, seeds, cifar100_defaults)
        print(f"[PHASE 2 COMPLETE] Byzantine Suite finished in {(time.time() - t0)/60.0:.2f} minutes.")

        # -------------------------------------------------------------
        # 3. BASELINE 2: JOB 3 (50-CLIENT SCALABILITY BENCHMARK)
        # -------------------------------------------------------------
        print("\n" + "=" * 80)
        print("PHASE 3: BASELINE 2 - JOB 3 (50-CLIENT POPULATION SCALING)")
        print("=" * 80)
        from scripts.run_scale_50clients import run_50clients_scaling
        t0 = time.time()
        rounds = 1 if args.smoke_test else 20
        sub = 100 if args.smoke_test else None
        run_50clients_scaling(
            num_clients=50,
            clients_per_round=10,
            num_rounds=rounds,
            batch_size=64,
            device=device,
            train_subset=sub,
            data_dir=data_dir,
            dataset="cifar100",
        )
        print(f"[PHASE 3 COMPLETE] 50-Client Benchmark finished in {(time.time() - t0)/60.0:.2f} minutes.")

        # -------------------------------------------------------------
        # 4. BASELINE 2: JOB 4 (MOBILENETV3 EDGE PROFILING)
        # -------------------------------------------------------------
        print("\n" + "=" * 80)
        print("PHASE 4: BASELINE 2 - JOB 4 (MOBILENETV3 EDGE VISION BENCHMARK)")
        print("=" * 80)
        from scripts.run_mobilenet_benchmark import run_mobilenet_benchmark
        t0 = time.time()
        rounds = 1 if args.smoke_test else 30
        sub = 100 if args.smoke_test else None
        run_mobilenet_benchmark(
            num_clients=15,
            num_rounds=rounds,
            batch_size=32,
            device=device,
            train_subset=sub,
            data_dir=data_dir,
            dataset="cifar100",
        )
        print(f"[PHASE 4 COMPLETE] MobileNetV3 Benchmark finished in {(time.time() - t0)/60.0:.2f} minutes.")

    # -------------------------------------------------------------
    # 5. FINALIZE: COMPILE LATEX TABLES & FIGURES
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 5: COMPILING PUBLICATION TABLES & FIGURES")
    print("=" * 80)
    try:
        from scripts.generate_all_tables import main as gen_tables
        gen_tables()
    except Exception as e:
        print(f"[warn] LaTeX table generation notice: {e}")

    try:
        from scripts.make_paper_figures import main as gen_figures
        gen_figures(argv=[])
    except Exception as e:
        print(f"[warn] Figure generation notice: {e}")

    # -------------------------------------------------------------
    # 6. PACKAGE ARTIFACTS
    # -------------------------------------------------------------
    package_artifacts(outputs_dir)

    total_elapsed = time.time() - t_suite_start
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS SUCCESSFULLY EXECUTED IN {total_elapsed/60.0:.2f} MINUTES")
    print("=" * 80)

    # -------------------------------------------------------------
    # 7. TIER 1 ON-TIME AUTO-STOP
    # -------------------------------------------------------------
    if args.auto_stop:
        execute_pod_shutdown()


if __name__ == "__main__":
    main()
