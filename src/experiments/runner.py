import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from src.config import SimulationConfig, ExperimentConfig, EnvironmentConfig
from src.core.aggregator import FedAvgAggregator
from src.utils.random import set_seed

from src.experiments.builder import TopologyEngineFactory, check_invariants, detect_device
from src.experiments.visualizer import (
    plot_experiment_results,
    plot_comparison_convergence,
    plot_robustness_summary,
    plot_robustness_heatmap,
    plot_byzantine_matrix,
)


class SingleExperimentRunner:
    """Runner for a single simulation configuration run."""

    def __init__(self, config: SimulationConfig, device: Optional[str] = None):
        self.config = config
        self.device = device or detect_device()

    def run(self) -> List[Dict[str, Any]]:
        set_seed(self.config.env.seed)

        topology, engine_cls = TopologyEngineFactory.build(self.config)
        aggregator = FedAvgAggregator()

        print(f"Using device: {self.device}")
        engine = engine_cls(
            config=self.config,
            topology=topology,
            aggregator=aggregator,
            device=self.device,
        )

        check_invariants(topology, self.config)
        engine.run()
        return engine.metrics.get_history()


class ExperimentRunner:
    """
    High-level orchestrator for loading ExperimentConfig instances,
    running single/sweep/comparison/matrix suites, saving structured results,
    and generating visualization artifacts.
    """

    def __init__(self, exp_config: ExperimentConfig, device: Optional[str] = None):
        self.exp_config = exp_config
        self.device = device or detect_device()

    @classmethod
    def from_yaml(cls, path_or_preset: str, overrides: Optional[Dict[str, Any]] = None, device: Optional[str] = None) -> "ExperimentRunner":
        """
        Load ExperimentRunner from a YAML path or built-in preset name.
        Allows applying dynamic overrides (e.g. CLI arguments).
        """
        # Resolve preset name to config path if needed
        config_path = path_or_preset
        if not os.path.exists(config_path):
            preset_candidates = [
                os.path.join("configs", f"{path_or_preset}.yaml"),
                os.path.join("configs", f"{path_or_preset}_experiment.yaml"),
                os.path.join("configs", path_or_preset),
            ]
            found = False
            for cand in preset_candidates:
                if os.path.exists(cand):
                    config_path = cand
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Config file or preset not found: {path_or_preset}")

        exp_cfg = ExperimentConfig.from_yaml(config_path)

        # Apply overrides if specified
        if overrides:
            exp_cfg = cls._apply_overrides(exp_cfg, overrides)

        return cls(exp_cfg, device=device)

    @staticmethod
    def _apply_overrides(exp_cfg: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
        """Apply CLI or runtime parameter overrides to ExperimentConfig."""
        data = exp_cfg.model_dump()

        if "num_rounds" in overrides and overrides["num_rounds"] is not None:
            data["num_rounds"] = overrides["num_rounds"]

        if "seed" in overrides and overrides["seed"] is not None:
            data.setdefault("env", {})["seed"] = overrides["seed"]

        if "output_dir" in overrides and overrides["output_dir"] is not None:
            data.setdefault("env", {})["output_dir"] = overrides["output_dir"]

        if "dataset" in overrides and overrides["dataset"] is not None:
            data.setdefault("env", {})["dataset"] = overrides["dataset"]

        return ExperimentConfig(**data)

    def run(self) -> List[Dict[str, Any]]:
        """Run all experiments defined by this configuration suite."""
        exp_type = self.exp_config.experiment_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        prefix_map = {
            "single": "single_run",
            "comparison": "comparison_study",
            "byzantine_matrix": "byzantine_matrix",
        }
        dir_prefix = prefix_map.get(exp_type, "experiment")

        experiment_root = os.path.join(self.exp_config.env.output_dir, f"{dir_prefix}_{timestamp}")
        plots_dir = os.path.join(experiment_root, "plots")
        metrics_dir = os.path.join(experiment_root, "metrics")

        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        entries = self.exp_config.build_configs(metrics_dir=metrics_dir)

        print(f"Experiment type: {exp_type}")
        print(f"Total runs: {len(entries)}")
        print(f"Output directory: {experiment_root}")
        print("-" * 65)

        all_histories = []

        if exp_type == "single":
            self._run_single_suite(entries, metrics_dir, plots_dir, all_histories)
        elif exp_type == "comparison":
            self._run_comparison_suite(entries, metrics_dir, plots_dir, experiment_root, all_histories)
        elif exp_type == "byzantine_matrix":
            self._run_matrix_suite(entries, plots_dir, experiment_root, all_histories)

        print("-" * 65)
        print("All experiments complete!")
        print(f"Results organized in: {experiment_root}")
        return all_histories

    def _run_single_suite(self, entries, metrics_dir, plots_dir, all_histories):
        entry = entries[0]
        config = entry["config"]
        label = entry["topo_label"]

        print(f"\nRunning: {label} ({config.experiment_name})")
        start_time = time.time()
        single_runner = SingleExperimentRunner(config, device=self.device)
        hx = single_runner.run()
        elapsed = round(time.time() - start_time, 2)
        all_histories.append({"entry": entry, "history": hx})

        final_acc = hx[-1].get("test_accuracy", 0.0)
        peak_acc = max(h.get("test_accuracy", 0.0) for h in hx)
        last5_acc = float(np.mean([h.get("test_accuracy", 0.0) for h in hx[-5:]]))
        if config.topology.type == "hierarchical_ensemble" and "ensemble_test_accuracy" in hx[-1]:
            final_acc = hx[-1]["ensemble_test_accuracy"]
            peak_acc = max(h.get("ensemble_test_accuracy", 0.0) for h in hx if "ensemble_test_accuracy" in h)
            last5_acc = float(np.mean([h.get("ensemble_test_accuracy", 0.0) for h in hx[-5:] if "ensemble_test_accuracy" in h]))

        print(f"  Final Acc: {final_acc:.2f}% | Peak Acc: {peak_acc:.2f}% | Last-5 Avg: {last5_acc:.2f}% | Time: {elapsed}s")

        metrics_json_path = os.path.join(metrics_dir, config.experiment_name, "metrics.json")
        plot_experiment_results(metrics_json_path, output_dir=plots_dir)

    def _run_comparison_suite(self, entries, metrics_dir, plots_dir, experiment_root, all_histories):
        summary_results = []
        scenario_experiments = {s.id: [] for s in self.exp_config.scenarios}

        for entry in entries:
            config = entry["config"]
            topo_label = entry["topo_label"]
            scenario_id = entry["scenario_id"]
            scenario_label = entry["scenario_label"]

            print(f"Running: {config.topology.type:22} | {scenario_label:25}", end=" ", flush=True)
            start_time = time.time()
            single_runner = SingleExperimentRunner(config, device=self.device)
            hx = single_runner.run()
            elapsed = round(time.time() - start_time, 2)
            all_histories.append({"entry": entry, "history": hx})

            exp_dir = os.path.join(metrics_dir, config.experiment_name)
            if scenario_id in scenario_experiments:
                scenario_experiments[scenario_id].append((exp_dir, topo_label))

            num_rounds = config.num_rounds
            num_clients = config.clients.num_clients
            is_ensemble = (config.topology.type == "hierarchical_ensemble")
            p_method = getattr(config.clients, "personalization_method", "none")

            if is_ensemble:
                steps_per_round = getattr(config.clients, "total_local_steps", 10)
            elif p_method == "ditto":
                steps_per_round = getattr(config.clients, "local_steps", 1) * 2
            else:
                steps_per_round = getattr(config.clients, "local_steps", 1)

            global_acc = hx[-1].get("test_accuracy", 0.0)
            global_last5 = round(float(np.mean([h.get("test_accuracy", 0.0) for h in hx[-5:]])), 2)

            summary_results.append({
                "Topology": topo_label,
                "Scenario": scenario_label,
                "Final Accuracy": global_acc,
                "Last5 Avg Accuracy": global_last5,
                "Wall Clock Seconds": elapsed,
                "Metric": "Global"
            })

            has_pers = "ensemble_test_accuracy" in hx[-1]
            if has_pers:
                pers_acc = hx[-1]["ensemble_test_accuracy"]
                pers_last5 = round(float(np.mean([h.get("ensemble_test_accuracy", 0.0) for h in hx[-5:] if "ensemble_test_accuracy" in h])), 2)

                summary_results.append({
                    "Topology": f"{topo_label} (Pers.)",
                    "Scenario": scenario_label,
                    "Final Accuracy": pers_acc,
                    "Last5 Avg Accuracy": pers_last5,
                    "Wall Clock Seconds": elapsed,
                    "Metric": "Personalized"
                })

            print(f"| (Global) Last5: {global_last5:5.2f}%", end="")
            if has_pers:
                print(f" | (Pers.) Last5: {pers_last5:5.2f}% | Time: {elapsed}s")
            else:
                print(f" | Time: {elapsed}s")

        # Convergence charts per scenario
        for scenario in self.exp_config.scenarios:
            exps = scenario_experiments.get(scenario.id, [])
            if not exps:
                continue
            dirs, labels = zip(*exps)
            plot_path = os.path.join(plots_dir, f"convergence_{scenario.id}.png")
            plot_comparison_convergence(dirs, labels, plot_path)

        # Summary charts & exported reports
        df_summary = pd.DataFrame(summary_results)
        summary_plot = os.path.join(plots_dir, "robustness_summary.png")
        heatmap_plot = os.path.join(plots_dir, "robustness_heatmap.png")
        title = f"Robustness Comparison across Topologies (after {self.exp_config.num_rounds} rounds)"
        plot_robustness_summary(df_summary, summary_plot, title=title)
        plot_robustness_heatmap(df_summary, heatmap_plot, title=f"Real-World Accuracy Heatmap (Last-5 Round Avg)")

        with open(os.path.join(experiment_root, "summary.json"), "w", encoding='utf-8') as f:
            json.dump(summary_results, f, indent=4)
        df_summary.to_csv(os.path.join(experiment_root, "comparison_results.csv"), index=False)

    def _run_matrix_suite(self, entries, plots_dir, experiment_root, all_histories):
        results = []
        for entry in entries:
            config = entry["config"]
            topo_label = entry["topo_label"]
            rate = entry["byzantine_rate"]

            print(f"Topo: {topo_label:15} | Byz Rate: {rate:3.1f}", end=" ", flush=True)
            start_time = time.time()
            single_runner = SingleExperimentRunner(config, device=self.device)
            hx = single_runner.run()
            elapsed = round(time.time() - start_time, 2)
            all_histories.append({"entry": entry, "history": hx})

            final_acc = hx[-1].get("test_accuracy", 0.0)
            peak_acc = max(h.get("test_accuracy", 0.0) for h in hx)
            last5_acc = round(float(np.mean([h.get("test_accuracy", 0.0) for h in hx[-5:]])), 2)

            if "ensemble_test_accuracy" in hx[-1]:
                final_acc = hx[-1]["ensemble_test_accuracy"]
                peak_acc = max(h.get("ensemble_test_accuracy", 0.0) for h in hx if "ensemble_test_accuracy" in h)
                last5_acc = round(float(np.mean([h.get("ensemble_test_accuracy", 0.0) for h in hx[-5:] if "ensemble_test_accuracy" in h])), 2)

            results.append({
                "Topology": topo_label,
                "Byzantine Rate": rate,
                "Final Accuracy": final_acc,
                "Peak Accuracy": peak_acc,
                "Last5 Avg Accuracy": last5_acc,
                "Wall Clock Seconds": elapsed
            })
            print(f"| Last5 Acc: {last5_acc:6.2f}% (Peak: {peak_acc:6.2f}%) | Time: {elapsed}s")

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(experiment_root, "matrix_results.csv"), index=False)

        title = f"Byzantine Robustness Matrix (after {self.exp_config.num_rounds} rounds)"
        plot_byzantine_matrix(df, os.path.join(plots_dir, "byzantine_robustness_matrix.png"), title=title)
