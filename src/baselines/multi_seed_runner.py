"""Multi-seed experiment runner for baselines with automatic statistical aggregation.
"""
from typing import List, Dict, Any, Optional
import time
import numpy as np

from src.config import SimulationConfig
from src.utils.random import set_seed
from src.baselines.factory import build_baseline_engine, detect_accelerator


class MultiSeedRunner:
    """Executes a baseline configuration across multiple independent random seeds."""

    DEFAULT_SEEDS = [42, 123, 7]

    def __init__(
        self,
        base_config: SimulationConfig,
        method_id: Optional[str] = None,
        seeds: Optional[List[int]] = None,
        device: Optional[str] = None,
    ):
        self.base_config = base_config
        self.method_id = method_id
        self.seeds = seeds if seeds is not None else list(self.DEFAULT_SEEDS)
        self.device = device or detect_accelerator()

    def run(self) -> Dict[str, Any]:
        """Run all seeds and return aggregated statistical performance metrics."""
        runs = []
        start_time = time.time()

        for idx, seed in enumerate(self.seeds, 1):
            print(f"\n--- Running [{self.method_id or 'baseline'}] Seed {seed} ({idx}/{len(self.seeds)}) ---")
            config = self.base_config.model_copy(deep=True)
            config.env.seed = seed
            config.experiment_name = f"{self.base_config.experiment_name}_seed{seed}"

            set_seed(seed)
            topology, aggregator, engine = build_baseline_engine(
                config=config,
                method_id=self.method_id,
                device=self.device,
            )

            engine.run()
            history = engine.metrics.get_history()
            runs.append({"seed": seed, "history": history})

            # Explicit GPU memory cleanup between seeds to prevent VRAM fragmentation on Kaggle
            del engine, topology, aggregator
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        total_elapsed = time.time() - start_time
        summary = self._aggregate_metrics(runs)
        summary["elapsed_seconds"] = round(total_elapsed, 2)
        summary["method_id"] = self.method_id
        summary["num_seeds"] = len(self.seeds)
        return summary

    def _aggregate_metrics(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        accuracies = []
        losses = []
        b10_fairnesses = []

        for r in runs:
            history = r["history"]
            if not history:
                continue
            final_round = history[-1]

            # Use ensemble / personalized accuracy if available, else test_accuracy
            acc = final_round.get("ensemble_test_accuracy", final_round.get("test_accuracy", 0.0))
            loss = final_round.get("ensemble_test_loss", final_round.get("test_loss", 0.0))
            b10 = final_round.get("bottom10_fairness", None)

            accuracies.append(float(acc))
            losses.append(float(loss))
            if b10 is not None:
                b10_fairnesses.append(float(b10))

        result = {
            "mean_accuracy": round(float(np.mean(accuracies)), 2) if accuracies else 0.0,
            "std_accuracy": round(float(np.std(accuracies)), 2) if accuracies else 0.0,
            "mean_loss": round(float(np.mean(losses)), 4) if losses else 0.0,
            "std_loss": round(float(np.std(losses)), 4) if losses else 0.0,
            "per_seed_accuracies": [round(a, 2) for a in accuracies],
            "raw_runs": runs,
        }

        if b10_fairnesses:
            result["mean_bottom10"] = round(float(np.mean(b10_fairnesses)), 2)
            result["std_bottom10"] = round(float(np.std(b10_fairnesses)), 2)
            result["per_seed_bottom10"] = [round(b, 2) for b in b10_fairnesses]
        else:
            result["mean_bottom10"] = None
            result["std_bottom10"] = None
            result["per_seed_bottom10"] = []

        return result
