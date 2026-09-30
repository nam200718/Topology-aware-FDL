"""CLI entry point for running baseline experiments on CIFAR-100.

Usage examples:
    python -m src.baselines.run_all_baselines --job personalization --smoke-test
    python -m src.baselines.run_all_baselines --job personalization --seeds 42,123,7
    python -m src.baselines.run_all_baselines --job byzantine --attacks label_flip,sign_flip
"""
import argparse
import json
import os
import sys
import tempfile
import time
import pandas as pd

from src.baselines.experiment_configs import (
    METHODS,
    PERSONALIZATION_METHODS,
    BYZANTINE_METHODS,
    ABLATION_METHODS,
    REGIMES,
    ATTACK_TYPES,
    BYZANTINE_RATES,
    SEEDS,
    CIFAR100_DEFAULTS,
    get_method_meta,
    get_regime_meta,
    get_attack_meta,
    create_personalization_config,
    create_byzantine_config,
)
from src.baselines.multi_seed_runner import MultiSeedRunner
from src.baselines.factory import detect_accelerator


def _parse_list_arg(val, item_type=str):
    """Parses a CLI argument that may be given as:
    - a space-separated list of items (from nargs='+')
    - a comma-separated string or list of comma-separated strings
    - a single item or None
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        tokens = []
        for x in val:
            if isinstance(x, str):
                tokens.extend(x.replace(",", " ").split())
            else:
                tokens.append(x)
        return [item_type(t) for t in tokens if str(t).strip() != ""]
    elif isinstance(val, str):
        tokens = val.replace(",", " ").split()
        return [item_type(t) for t in tokens if t.strip() != ""]
    return [item_type(val)]



def _atomic_json_dump(data, file_path: str):
    dir_name = os.path.dirname(file_path) or "."
    os.makedirs(dir_name, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, prefix=f".{os.path.basename(file_path)}.", suffix=".tmp") as f:
        tmp_path = f.name
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, file_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run AAMAS 2027 Baseline Experiments (CIFAR-100)")
    parser.add_argument(
        "--job",
        "--track",
        dest="job",
        choices=["personalization", "byzantine", "ablation", "all"],
        default="personalization",
        help="Experiment suite to run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="List of methods to evaluate (space-separated or comma-separated, e.g. 'fedavg ditto topo' or 'fedavg,ditto')",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=None,
        help="List of heterogeneity regimes (space-separated or comma-separated, e.g. '0.1 0.5' or 'severe,moderate')",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=None,
        help="List of attack types (space-separated or comma-separated, e.g. 'label_flip sign_flip')",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        default=None,
        help="List of byzantine rates (space-separated or comma-separated, e.g. '0.0 0.1 0.2')",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["42", "123", "7"],
        help="Random seeds (space-separated or comma-separated, e.g. '42 123 7' or '42,123,7')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "cifar10", "synthetic", "mnist"],
        help="Dataset to evaluate on (default: cifar100)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override number of communication rounds",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=None,
        help="Override number of clients",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["resnet9", "simple_cnn", "mobilenetv3"],
        help="Model architecture",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda, cpu, directml)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to dataset directory or Kaggle input mount (e.g. /kaggle/input/cifar100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/baselines",
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run minimal 1-round smoke test to verify execution integrity",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if results already exist in checkpoint",
    )
    return parser.parse_args()


def run_personalization_suite(args, device, seeds, base_defaults):
    print("\n" + "=" * 70)
    print("STARTING JOB 1: CIFAR-100 PERSONALIZATION & HETEROGENEITY BENCHMARK")
    print("=" * 70)

    target_methods = [get_method_meta(m)["id"] for m in _parse_list_arg(args.methods, str)] if args.methods else PERSONALIZATION_METHODS
    target_regimes = [get_regime_meta(r)["id"] for r in _parse_list_arg(args.regimes, str)] if args.regimes else [r["id"] for r in REGIMES]

    out_dir = os.path.join(args.output_dir, "personalization")
    os.makedirs(out_dir, exist_ok=True)

    results_table = []
    checkpoint_file = os.path.join(out_dir, "results_personalization.json")
    completed_keys = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    results_table = loaded
                    for item in results_table:
                        m = item.get("method")
                        r = item.get("regime")
                        if m and r:
                            completed_keys.add((m, r))
            print(f"Loaded {len(results_table)} completed results from checkpoint: {checkpoint_file}")
        except Exception as e:
            print(f"Warning: Failed to load existing checkpoint from {checkpoint_file}: {e}")

    for m_id in target_methods:
        for r_id in target_regimes:
            if (m_id, r_id) in completed_keys and not getattr(args, "force", False):
                print(f"\n>>> Skipping Method: {m_id.upper()} | Regime: {r_id.upper()} (Already Completed) <<<")
                continue
            print(f"\n>>> Running Method: {m_id.upper()} | Regime: {r_id.upper()} <<<")
            config = create_personalization_config(
                method_id=m_id,
                regime_id=r_id,
                base_defaults=base_defaults,
                output_dir=out_dir,
            )

            runner = MultiSeedRunner(
                base_config=config,
                method_id=m_id,
                seeds=seeds,
                device=device,
            )
            res = runner.run()

            entry = {
                "method": m_id,
                "regime": r_id,
                "mean_acc": res["mean_accuracy"],
                "std_acc": res["std_accuracy"],
                "mean_loss": res["mean_loss"],
                "mean_b10": res["mean_bottom10"],
                "std_b10": res["std_bottom10"],
                "per_seed_acc": res["per_seed_accuracies"],
                "per_seed_b10": res["per_seed_bottom10"],
                "elapsed_s": res["elapsed_seconds"],
            }
            if getattr(args, "force", False):
                results_table = [item for item in results_table if not (item.get("method") == m_id and item.get("regime") == r_id)]
            results_table.append(entry)
            completed_keys.add((m_id, r_id))
            _atomic_json_dump(results_table, checkpoint_file)

    df = pd.DataFrame(results_table)
    csv_file = os.path.join(out_dir, "results_personalization.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nPersonalization suite completed! Results saved to:\n- {checkpoint_file}\n- {csv_file}")
    return df


def run_byzantine_suite(args, device, seeds, base_defaults):
    print("\n" + "=" * 70)
    print("STARTING JOB 2: CIFAR-100 BYZANTINE ROBUSTNESS BENCHMARK")
    print("=" * 70)

    target_methods = [get_method_meta(m)["id"] for m in _parse_list_arg(args.methods, str)] if args.methods else BYZANTINE_METHODS
    target_attacks = [get_attack_meta(a)["id"] for a in _parse_list_arg(args.attacks, str)] if args.attacks else [a["id"] for a in ATTACK_TYPES]
    target_rates = _parse_list_arg(args.rates, float) if args.rates else BYZANTINE_RATES

    out_dir = os.path.join(args.output_dir, "byzantine")
    os.makedirs(out_dir, exist_ok=True)

    results_table = []
    checkpoint_file = os.path.join(out_dir, "results_byzantine.json")
    completed_keys = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    results_table = loaded
                    for item in results_table:
                        m = item.get("method")
                        a = item.get("attack")
                        r = item.get("byzantine_rate")
                        if m is not None and a is not None and r is not None:
                            completed_keys.add((m, a, round(float(r), 4)))
            print(f"Loaded {len(results_table)} completed results from checkpoint: {checkpoint_file}")
        except Exception as e:
            print(f"Warning: Failed to load existing checkpoint from {checkpoint_file}: {e}")

    for atk in target_attacks:
        for rate in target_rates:
            for m_id in target_methods:
                if (m_id, atk, round(float(rate), 4)) in completed_keys and not getattr(args, "force", False):
                    print(f"\n>>> Skipping Method: {m_id.upper()} | Attack: {atk} | Byzantine Rate: {int(rate * 100)}% (Already Completed) <<<")
                    continue
                print(f"\n>>> Method: {m_id.upper()} | Attack: {atk} | Byzantine Rate: {int(rate * 100)}% <<<")
                config = create_byzantine_config(
                    method_id=m_id,
                    attack_type=atk,
                    byzantine_rate=rate,
                    base_defaults=base_defaults,
                    output_dir=out_dir,
                )

                runner = MultiSeedRunner(
                    base_config=config,
                    method_id=m_id,
                    seeds=seeds,
                    device=device,
                )
                res = runner.run()

                entry = {
                    "method": m_id,
                    "attack": atk,
                    "byzantine_rate": rate,
                    "mean_acc": res["mean_accuracy"],
                    "std_acc": res["std_accuracy"],
                    "per_seed_acc": res["per_seed_accuracies"],
                    "elapsed_s": res["elapsed_seconds"],
                }
                if getattr(args, "force", False):
                    results_table = [item for item in results_table if not (
                        item.get("method") == m_id and item.get("attack") == atk and round(float(item.get("byzantine_rate", -1)), 4) == round(float(rate), 4)
                    )]
                results_table.append(entry)
                completed_keys.add((m_id, atk, round(float(rate), 4)))
                _atomic_json_dump(results_table, checkpoint_file)

    df = pd.DataFrame(results_table)
    csv_file = os.path.join(out_dir, "results_byzantine.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nByzantine suite completed! Results saved to:\n- {checkpoint_file}\n- {csv_file}")
    return df


def run_ablation_suite(args, device, seeds, base_defaults):
    print("\n" + "=" * 70)
    print("STARTING JOB 5: CIFAR-100 ABLATION & CLUSTER VALUATION BENCHMARK")
    print("=" * 70)

    target_methods = [get_method_meta(m)["id"] for m in _parse_list_arg(args.methods, str)] if args.methods else ABLATION_METHODS
    target_regimes = [get_regime_meta(r)["id"] for r in _parse_list_arg(args.regimes, str)] if args.regimes else [r["id"] for r in REGIMES]

    out_dir = os.path.join(args.output_dir, "ablation")
    os.makedirs(out_dir, exist_ok=True)

    results_table = []
    checkpoint_file = os.path.join(out_dir, "results_ablation.json")
    completed_keys = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    results_table = loaded
                    for item in results_table:
                        m = item.get("method")
                        r = item.get("regime")
                        if m and r:
                            completed_keys.add((m, r))
            print(f"Loaded {len(results_table)} completed results from checkpoint: {checkpoint_file}")
        except Exception as e:
            print(f"Warning: Failed to load existing checkpoint from {checkpoint_file}: {e}")

    for m_id in target_methods:
        for r_id in target_regimes:
            if (m_id, r_id) in completed_keys and not getattr(args, "force", False):
                print(f"\n>>> Skipping Ablation Method: {m_id.upper()} | Regime: {r_id.upper()} (Already Completed) <<<")
                continue
            print(f"\n>>> Running Ablation Method: {m_id.upper()} | Regime: {r_id.upper()} <<<")
            config = create_personalization_config(
                method_id=m_id,
                regime_id=r_id,
                base_defaults=base_defaults,
                output_dir=out_dir,
            )

            runner = MultiSeedRunner(
                base_config=config,
                method_id=m_id,
                seeds=seeds,
                device=device,
            )
            res = runner.run()

            entry = {
                "method": m_id,
                "regime": r_id,
                "mean_acc": res["mean_accuracy"],
                "std_acc": res["std_accuracy"],
                "mean_loss": res["mean_loss"],
                "mean_b10": res["mean_bottom10"],
                "std_b10": res["std_bottom10"],
                "per_seed_acc": res["per_seed_accuracies"],
                "per_seed_b10": res["per_seed_bottom10"],
                "elapsed_s": res["elapsed_seconds"],
            }
            if getattr(args, "force", False):
                results_table = [item for item in results_table if not (item.get("method") == m_id and item.get("regime") == r_id)]
            results_table.append(entry)
            completed_keys.add((m_id, r_id))
            _atomic_json_dump(results_table, checkpoint_file)

    df = pd.DataFrame(results_table)
    csv_file = os.path.join(out_dir, "results_ablation.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nAblation suite completed! Results saved to:\n- {checkpoint_file}\n- {csv_file}")
    return df


def main():
    args = parse_args()
    device = args.device or detect_accelerator()
    seeds = _parse_list_arg(args.seeds, int) or [42, 123, 7]

    base_defaults = dict(CIFAR100_DEFAULTS)
    base_defaults["dataset"] = args.dataset
    base_defaults["data_dir"] = args.data_dir
    if args.rounds is not None:
        base_defaults["num_rounds"] = args.rounds
    if args.clients is not None:
        base_defaults["num_clients"] = args.clients
    if args.model is not None:
        base_defaults["model_name"] = args.model

    if args.smoke_test:
        print("[SMOKE TEST ENGAGED] Using 1 round, 1 seed, and minimal dataset subsets.")
        seeds = [42]
        base_defaults["num_rounds"] = 1
        base_defaults["eval_interval"] = 1
        base_defaults["train_subset"] = 200
        base_defaults["test_subset"] = 50

    if args.job in ("personalization", "all"):
        run_personalization_suite(args, device, seeds, base_defaults)

    if args.job in ("byzantine", "all"):
        run_byzantine_suite(args, device, seeds, base_defaults)

    if args.job in ("ablation", "all"):
        run_ablation_suite(args, device, seeds, base_defaults)


if __name__ == "__main__":
    main()
