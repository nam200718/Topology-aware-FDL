"""
Defense Experiment Runner.
Run sweep Byzantine defense experiment on Hierarchical Ensemble topology.
Compare no_defense (FedAvg) vs soft_cosine (SoftRejectionAggregator).

Usage:
    python scripts/run_defense_experiment.py --config configs/defense_sweep.yaml
    python scripts/run_defense_experiment.py --config configs/defense_sweep.yaml --dataset cifar10
    python scripts/run_defense_experiment.py --config configs/defense_sweep.yaml --dataset cifar10 --model resnet9 --num_rounds 30
"""
import os
import sys
import json
import argparse
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from src.config import ExperimentConfig, SimulationConfig
from src.core.aggregator import FedAvgAggregator
from src.utils.random import set_seed
from src.experiments.builder import TopologyEngineFactory, check_invariants, detect_device

from src.defense.config import DefenseConfig
from src.defense.engine import DefendedEnsembleEngine
from src.defense.visualizer import DefenseVisualizer


def run_single_defense_experiment(
    config: SimulationConfig,
    defense_mode: str,
    device: str,
) -> tuple:
    """
    Run a single experiment, return (metrics_history, trust_tracker, byzantine_ids).
    """
    set_seed(config.env.seed)

    topology, engine_cls = TopologyEngineFactory.build(config)
    aggregator = FedAvgAggregator()

    if defense_mode != "none" and config.topology.type == "hierarchical_ensemble":
        # Use DefendedEnsembleEngine instead of HierarchicalEnsembleEngine
        defense_config = DefenseConfig(defense_mode=defense_mode)
        engine = DefendedEnsembleEngine(
            config=config,
            topology=topology,
            aggregator=aggregator,
            device=device,
            defense_config=defense_config,
        )
    else:
        # No defense or other topology → use original engine
        engine = engine_cls(
            config=config,
            topology=topology,
            aggregator=aggregator,
            device=device,
        )

    check_invariants(topology, config)

    # Collect Byzantine client IDs
    byzantine_ids = [
        cid for cid, state in engine.clients_state.items() if state.is_byzantine
    ]

    engine.run()

    trust_tracker = getattr(engine, "trust_tracker", None)
    return engine.metrics.get_history(), trust_tracker, byzantine_ids


def main():
    parser = argparse.ArgumentParser(description="Defense Sweep Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_project_root, "configs", "defense_sweep.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset (mnist/cifar10)")
    parser.add_argument("--model", type=str, default=None, help="Override model (simple_cnn/resnet9)")
    parser.add_argument("--num_rounds", type=int, default=None, help="Override num_rounds")
    args = parser.parse_args()

    device = detect_device()
    print(f"Using device: {device}")

    # Load config
    exp_config = ExperimentConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.dataset:
        exp_config.env.dataset = args.dataset
    if args.num_rounds:
        exp_config.num_rounds = args.num_rounds

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = os.path.join(
        exp_config.env.output_dir, f"defense_sweep_{timestamp}"
    )
    plots_dir = os.path.join(experiment_root, "plots")
    metrics_dir = os.path.join(experiment_root, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Build experiment entries
    entries = exp_config.build_configs(metrics_dir=metrics_dir)

    print(f"Total runs: {len(entries)}")
    print(f"Output directory: {experiment_root}")
    print("-" * 65)

    # Collect results for accuracy vs. byzantine rate chart
    accuracy_results = []
    last_trust_tracker = None
    last_byzantine_ids = []

    for entry in entries:
        config = entry["config"]
        topo_label = entry["topo_label"]
        rate = entry["byzantine_rate"]

        # Extract defense_mode from topology params
        defense_mode = config.topology.params.get("defense_mode", "none")

        # Apply model override
        if args.model:
            config.clients.model_name = args.model

        print(f"\nRunning: {topo_label} | Byz Rate: {rate:.1f} | Defense: {defense_mode}")

        history, trust_tracker, byzantine_ids = run_single_defense_experiment(
            config, defense_mode, device
        )

        # Get final accuracy
        final_acc = history[-1].get("test_accuracy", 0.0)
        if "ensemble_test_accuracy" in history[-1]:
            final_acc = history[-1]["ensemble_test_accuracy"]

        print(f"  Final Accuracy: {final_acc:.2f}%")

        accuracy_results.append({
            "byzantine_rate": rate,
            "defense_mode": defense_mode,
            "final_accuracy": final_acc,
            "label": topo_label,
        })

        # Save per-run metrics
        run_dir = os.path.join(metrics_dir, config.experiment_name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

        # Keep trust tracker from defense runs for heatmap
        if trust_tracker is not None and rate > 0:
            last_trust_tracker = trust_tracker
            last_byzantine_ids = byzantine_ids

    # ============================================
    # Generate Deep Analytics
    # ============================================
    print("\n" + "=" * 65)
    print("Generating Deep Analytics...")

    # Deep Analytic 1: Trust Score Heatmap (từ run có Byzantine + defense)
    if last_trust_tracker is not None:
        DefenseVisualizer.plot_trust_heatmap(
            trust_tracker=last_trust_tracker,
            byzantine_ids=last_byzantine_ids,
            output_path=os.path.join(plots_dir, "trust_score_heatmap.png"),
            title="Trust Score Evolution — Soft Rejection Defense",
        )

    # Deep Analytic 2: Accuracy vs. Byzantine Rate
    DefenseVisualizer.plot_accuracy_vs_byzantine_rate(
        results=accuracy_results,
        output_path=os.path.join(plots_dir, "accuracy_vs_byzantine_rate.png"),
        title=f"Accuracy vs. Byzantine Rate ({exp_config.env.dataset.upper()}, {exp_config.num_rounds} rounds)",
    )

    # Save summary CSV
    df = pd.DataFrame(accuracy_results)
    df.to_csv(os.path.join(experiment_root, "defense_sweep_results.csv"), index=False)

    print(f"\nAll results saved to: {experiment_root}")
    print("=" * 65)


if __name__ == "__main__":
    main()
