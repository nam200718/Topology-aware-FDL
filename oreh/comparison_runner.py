"""
OrehComparisonRunner: Orchestrates multi-tier Byzantine evaluation across
CIFAR-100 and CIFAR-10 for the three Midterm Research Ideas:
1. Defense Scope Evaluation (Global vs Cluster vs Both vs None) under Targeted Attacks.
2. ALIE Attack Suite (FedAvg, Ditto, Multi-Krum, Defended FedHEP).
3. Number of Peer Clusters (K) Sensitivity Sweep under non-IID Skews.

Features:
- Crash & Timeout Resilience: Appends and flushes each experiment result immediately to dedicated CSV files.
- Resumable: Skips already-completed runs so overnight execution can be resumed without restarting.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Literal
import pandas as pd
import numpy as np
import torch

from src.config import (
    SimulationConfig,
    EnvironmentConfig,
    TopologyConfig,
    ClientConfig,
    RobustnessConfig,
    NonIIDConfig,
)
from src.topologies.hierarchical import HierarchicalTopology
from src.topologies.star import StarTopology
from src.core.interfaces import Aggregator, ClientState
from src.core.aggregator import FedAvgAggregator, TrimmedMeanAggregator, coordinate_median
from src.baselines.multi_krum_aggregator import MultiKrumAggregator
from src.defense.config import DefenseConfig
from src.utils.random import set_seed
from src.utils.device import resolve_device

from oreh.engine import OrehDefendedEngine, OrehCentralizedEngine
from oreh.alie_attack import compute_alie_z_max


class CoordinateMedianAggregator(Aggregator):
    """Coordinate-wise median Byzantine-resilient aggregation rule."""

    def aggregate(self, states: List[ClientState]) -> torch.Tensor:
        if not states:
            raise ValueError("Cannot aggregate empty list of states.")
        if len(states) <= 2:
            return torch.stack([s.weights for s in states]).mean(dim=0)
        stacked = torch.stack([s.weights for s in states], dim=0)
        return coordinate_median(stacked)


def _append_result_to_csv(csv_path: str, result_dict: Dict[str, Any]):
    """Appends a single experiment result immediately to CSV, aligning with existing columns."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df_row = pd.DataFrame([result_dict])
    file_has_data = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    if file_has_data:
        existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        df_row = df_row.reindex(columns=existing_cols)
    df_row.to_csv(csv_path, mode="a", header=not file_has_data, index=False)


def _normalize_method(m: str) -> str:
    m_clean = str(m).strip().lower().replace("_", "").replace(" ", "").replace("-", "")
    if "hep" in m_clean:
        return "fedhep"
    if "krum" in m_clean:
        return "multikrum"
    if "ditto" in m_clean:
        return "ditto"
    if "avg" in m_clean:
        return "fedavg"
    if "median" in m_clean:
        return "median"
    if "trim" in m_clean:
        return "trimmedmean"
    return m_clean


_STAR_METHOD_CANONICAL = {
    "fedavg": "fedavg",
    "ditto": "ditto",
    "multikrum": "multikrum",
    "median": "median",
    "trimmedmean": "trimmed_mean",
}


def _is_star_method(m: str) -> bool:
    return _normalize_method(m) in _STAR_METHOD_CANONICAL


def _canonical_star_method(m: str) -> str:
    norm = _normalize_method(m)
    if norm not in _STAR_METHOD_CANONICAL:
        raise ValueError(f"Unsupported star baseline method: '{m}'")
    return _STAR_METHOD_CANONICAL[norm]


def _normalize_key_item(col: str, val: Any) -> str:
    if col == "method":
        return _normalize_method(str(val))
    if isinstance(val, bool) or str(val).lower() in ("true", "false"):
        return str(val).strip().lower()
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        return f"{f:.4f}".rstrip("0")
    except (ValueError, TypeError):
        return str(val).strip().lower()


def _load_existing_keys(csv_path: str, key_columns: List[str]) -> set:
    """Loads a set of normalized unique key tuples of already completed experiments to allow resume."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    try:
        df = pd.read_csv(csv_path)
        existing = set()
        for _, row in df.iterrows():
            key = tuple(_normalize_key_item(col, row.get(col, "")) for col in key_columns)
            existing.add(key)
        return existing
    except Exception:
        return set()


class OrehComparisonRunner:
    """
    Runner for the 3 Midterm Research Ideas on CIFAR-100 (or CIFAR-10).
    """

    def __init__(
        self,
        dataset: str = "cifar100",
        num_rounds: int = 30,
        num_clients: int = 15,
        num_clusters: int = 3,
        alpha: float = 0.5,
        device: Optional[str] = None,
        train_subset: Optional[int] = 10000,
        test_subset: Optional[int] = 3000,
        data_dir: str = "./data",
        output_dir: str = "./outputs/oreh_eval",
        clustering_mode: str = "none",
    ):
        self.dataset = dataset.lower()
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.device = device or str(resolve_device())
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.clustering_mode = clustering_mode
        os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Standalone / Sanity Experiment Dispatcher
    # -------------------------------------------------------------------------
    def run_single_experiment(
        self,
        defense_scope: str = "both",
        attack_target_tier: str = "all",
        attack_type: str = "alie",
        byzantine_rate: float = 0.2,
        seed: int = 42,
        alie_z_max: Optional[float] = None,
        method_label: str = "FedHEP",
        method: Optional[str] = None,
        num_clusters_override: Optional[int] = None,
        alpha_override: Optional[float] = None,
        num_clients_override: Optional[int] = None,
        enable_ttt: bool = True,
        clustering_mode_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes a single experiment run (FedHEP or baseline method) with the given configuration.
        Used for pre-flight sanity checks and standalone evaluations.
        """
        target_method = method or method_label
        if _is_star_method(target_method):
            return self._run_star_baseline_experiment(
                method_id=_canonical_star_method(target_method),
                attack_type=attack_type,
                byzantine_rate=byzantine_rate,
                alie_z_max=alie_z_max,
                seed=seed,
                num_clients_override=num_clients_override,
            )
        return self._run_hep_experiment(
            defense_scope=defense_scope,
            attack_target_tier=attack_target_tier,
            attack_type=attack_type,
            byzantine_rate=byzantine_rate,
            seed=seed,
            alie_z_max=alie_z_max,
            method_label=method_label,
            num_clusters_override=num_clusters_override,
            alpha_override=alpha_override,
            num_clients_override=num_clients_override,
            enable_ttt=enable_ttt,
            clustering_mode_override=clustering_mode_override,
        )


    # -------------------------------------------------------------------------
    # IDEA 1: Defense Scope vs Targeted Attack Tiers
    # -------------------------------------------------------------------------
    def run_idea1_scope_matrix(
        self,
        attack_types: List[str] = ["alie", "sign_flip"],
        scopes: List[str] = ["global", "cluster", "both", "none"],
        target_tiers: List[str] = ["root_only", "parent_only", "all"],
        byzantine_rate: float = 0.2,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Executes Idea 1: Evaluates Defense Scopes under Targeted Tier Attacks.
        Saves continuously to {output_dir}/{dataset}_idea1_scope.csv.
        """
        csv_path = os.path.join(self.output_dir, f"{self.dataset}_idea1_scope.csv")
        key_cols = ["defense_scope", "attack_target_tier", "attack_type", "byzantine_rate", "seed"]
        completed = _load_existing_keys(csv_path, key_cols)

        print("\n" + "=" * 70)
        print(">> [IDEA 1] RUNNING DEFENSE SCOPE VS TARGETED TIER MATRIX")
        print(f" Dataset: {self.dataset.upper()} | Rounds: {self.num_rounds} | Device: {self.device}")
        print(f" Output File: {csv_path}")
        print("=" * 70)

        for atk in attack_types:
            for scope in scopes:
                for tgt in target_tiers:
                    key = tuple(_normalize_key_item(col, val) for col, val in zip(key_cols, [scope, tgt, atk, byzantine_rate, seed]))
                    if key in completed:
                        print(f">> [Skip/Done] Scope={scope} | Target={tgt} | Atk={atk}")
                        continue

                    res = self._run_hep_experiment(
                        defense_scope=scope,
                        attack_target_tier=tgt,
                        attack_type=atk,
                        byzantine_rate=byzantine_rate,
                        seed=seed,
                    )
                    _append_result_to_csv(csv_path, res)
                    completed.add(key)

        return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

    # Backward-compatible alias for CLI / scripts
    run_scope_vs_target_matrix = run_idea1_scope_matrix

    # -------------------------------------------------------------------------
    # IDEA 2: ALIE Attack Suite across Baselines (FedAvg, Ditto, Multi-Krum, FedHEP)
    # -------------------------------------------------------------------------
    def run_idea2_alie_suite(
        self,
        methods: List[str] = ["fedavg", "ditto", "multikrum", "fedhep"],
        byzantine_rates: List[float] = [0.1, 0.2, 0.3, 0.4],
        alie_z_max: Optional[float] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Executes Idea 2: ALIE Attack against FedAvg, Ditto, Multi-Krum, and Defended FedHEP.
        Saves continuously to {output_dir}/{dataset}_idea2_alie.csv.
        """
        csv_path = os.path.join(self.output_dir, f"{self.dataset}_idea2_alie.csv")
        key_cols = ["method", "attack_type", "byzantine_rate", "seed"]
        completed = _load_existing_keys(csv_path, key_cols)

        print("\n" + "=" * 70)
        print(">> [IDEA 2] RUNNING ALIE ATTACK SUITE (FEDAVG, DITTO, MULTI-KRUM, FEDHEP)")
        print(f" Dataset: {self.dataset.upper()} | Rounds: {self.num_rounds} | Device: {self.device}")
        print(f" Output File: {csv_path}")
        print("=" * 70)

        for m_id in methods:
            for q in byzantine_rates:
                key = tuple(_normalize_key_item(col, val) for col, val in zip(key_cols, [m_id, "alie", q, seed]))
                if key in completed:
                    print(f">> [Skip/Done] Method={m_id} | Atk=alie | q={q}")
                    continue

                if _is_star_method(m_id):
                    res = self._run_star_baseline_experiment(
                        method_id=_canonical_star_method(m_id),
                        attack_type="alie",
                        byzantine_rate=q,
                        alie_z_max=alie_z_max,
                        seed=seed,
                    )
                else:
                    # Defended FedHEP
                    res = self._run_hep_experiment(
                        defense_scope="both",
                        attack_target_tier="all",
                        attack_type="alie",
                        byzantine_rate=q,
                        alie_z_max=alie_z_max,
                        seed=seed,
                        method_label="Defended FedHEP",
                    )

                _append_result_to_csv(csv_path, res)
                completed.add(key)

        return pd.read_csv(csv_path)

    # -------------------------------------------------------------------------
    # IDEA 3: Number of Peer Clusters (K) Sensitivity Sweep
    # -------------------------------------------------------------------------
    def run_idea3_cluster_sweep(
        self,
        k_values: List[int] = [1, 2, 3, 5, 8],
        alphas: List[float] = [0.5, 0.05],
        byzantine_rate: float = 0.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Executes Idea 3: Sensitivity sweep over number of peer clusters K.
        Saves continuously to {output_dir}/{dataset}_idea3_k_clusters.csv.
        """
        csv_path = os.path.join(self.output_dir, f"{self.dataset}_idea3_k_clusters.csv")
        key_cols = ["num_clusters", "alpha", "byzantine_rate", "seed"]
        completed = _load_existing_keys(csv_path, key_cols)

        print("\n" + "=" * 70)
        print(">> [IDEA 3] RUNNING PEER CLUSTER COUNT (K) SENSITIVITY SWEEP")
        print(f" Dataset: {self.dataset.upper()} | Rounds: {self.num_rounds} | Device: {self.device}")
        print(f" Output File: {csv_path}")
        print("=" * 70)

        for alpha in alphas:
            for k in k_values:
                key = tuple(_normalize_key_item(col, val) for col, val in zip(key_cols, [k, alpha, byzantine_rate, seed]))
                if key in completed:
                    print(f">> [Skip/Done] K={k} | alpha={alpha}")
                    continue

                res = self._run_hep_experiment(
                    defense_scope="both" if byzantine_rate > 0 else "none",
                    attack_target_tier="all",
                    attack_type="label_flip" if byzantine_rate > 0 else "none",
                    byzantine_rate=byzantine_rate,
                    seed=seed,
                    num_clusters_override=k,
                    alpha_override=alpha,
                )
                _append_result_to_csv(csv_path, res)
                completed.add(key)

        return pd.read_csv(csv_path)

    # -------------------------------------------------------------------------
    # POST-MIDTERM 1: New Attacks Suite Targeting TTT (Idea C)
    # -------------------------------------------------------------------------
    def run_postmid_attacks(
        self,
        attacks: List[str] = ["sign_flip", "alie", "ipm", "adaptive_cosine", "delayed_onset"],
        byzantine_rates: List[float] = [0.2, 0.3],
        seeds: List[int] = [42, 123, 7],
        defense_scope: str = "both",
        attack_target_tier: str = "all",
        ipm_epsilon: float = 0.1,
        attack_onset_round: int = 15,
        methods: List[str] = ["fedhep"],
        output_filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Executes Post-Midterm Study C: Evaluates new attacking types designed to
        circumvent Temporal Trust Tracking (TTT) and subspace cosine filtering,
        and optionally compares against star robust baselines (MultiKrum, Median, TrimmedMean).
        Saves continuously to {output_dir}/{output_filename or f'{dataset}_postmid_attacks.csv'}.
        """
        csv_path = os.path.join(self.output_dir, output_filename or f"{self.dataset}_postmid_attacks.csv")
        key_cols = ["method", "attack_type", "byzantine_rate", "seed"]
        completed = _load_existing_keys(csv_path, key_cols)

        print("\n" + "=" * 70)
        print(">> [POST-MIDTERM] RUNNING NEW ATTACKS SUITE TARGETING TTT")
        print(f" Dataset: {self.dataset.upper()} | Rounds: {self.num_rounds} | Device: {self.device}")
        print(f" Methods: {methods} | Attacks: {attacks} | Byzantine Rates: {byzantine_rates} | Seeds: {seeds}")
        print(f" Output File: {csv_path}")
        print("=" * 70)

        for m_id in methods:
            is_star = _is_star_method(m_id)
            canon_label = _canonical_star_method(m_id).upper() if is_star else "Defended FedHEP"
            for atk in attacks:
                for q in byzantine_rates:
                    effective_atk = "none" if q == 0.0 else atk
                    for seed in seeds:
                        key = tuple(_normalize_key_item(col, val) for col, val in zip(key_cols, [canon_label, effective_atk, q, seed]))
                        if key in completed:
                            print(f">> [Skip/Done] Method={canon_label} | Attack={effective_atk} | q={q} | Seed={seed}")
                            continue

                        if is_star:
                            res = self._run_star_baseline_experiment(
                                method_id=_canonical_star_method(m_id),
                                attack_type=effective_atk,
                                byzantine_rate=q,
                                seed=seed,
                                ipm_epsilon=ipm_epsilon,
                                attack_onset_round=attack_onset_round,
                            )
                        else:
                            res = self._run_hep_experiment(
                                defense_scope=defense_scope,
                                attack_target_tier=attack_target_tier,
                                attack_type=effective_atk,
                                byzantine_rate=q,
                                seed=seed,
                                method_label="Defended FedHEP",
                                ipm_epsilon=ipm_epsilon,
                                attack_onset_round=attack_onset_round,
                            )
                        _append_result_to_csv(csv_path, res)
                        completed.add(key)

        return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

    # -------------------------------------------------------------------------
    # POST-MIDTERM 2: Majority Attacker Stress-Test (q >= 50%) (Idea B)
    # -------------------------------------------------------------------------
    def run_postmid_majority_q(
        self,
        byzantine_rates: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        attacks: List[str] = ["sign_flip", "alie"],
        methods: List[str] = ["fedhep", "fedavg", "ditto"],
        seeds: List[int] = [42, 123, 7],
        defense_scope: str = "both",
        output_filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Executes Post-Midterm Study B: Stress-tests FedHEP under majority attacker
        fractions (q >= 0.5) to analyze SCCF consensus inversion vs Star baselines.
        Saves continuously to {output_dir}/{output_filename or f'{dataset}_postmid_majority_q.csv'}.
        """
        csv_path = os.path.join(self.output_dir, output_filename or f"{self.dataset}_postmid_majority_q.csv")
        key_cols = ["method", "attack_type", "byzantine_rate", "seed"]
        completed = _load_existing_keys(csv_path, key_cols)

        print("\n" + "=" * 70)
        print(">> [POST-MIDTERM] RUNNING MAJORITY ATTACKER STRESS TEST (q >= 0.5)")
        print(f" Dataset: {self.dataset.upper()} | Rounds: {self.num_rounds} | Device: {self.device}")
        print(f" Methods: {methods} | Byzantine Rates: {byzantine_rates} | Seeds: {seeds}")
        print(f" Output File: {csv_path}")
        print("=" * 70)

        for m_id in methods:
            is_star = _is_star_method(m_id)
            for atk in attacks:
                for q in byzantine_rates:
                    effective_atk = "none" if q == 0.0 else atk
                    for seed in seeds:
                        key = tuple(_normalize_key_item(col, val) for col, val in zip(key_cols, [m_id, effective_atk, q, seed]))
                        if key in completed:
                            print(f">> [Skip/Done] Method={m_id} | Attack={effective_atk} | q={q} | Seed={seed}")
                            continue

                        if is_star:
                            res = self._run_star_baseline_experiment(
                                method_id=_canonical_star_method(m_id),
                                attack_type=effective_atk,
                                byzantine_rate=q,
                                seed=seed,
                            )
                        else:
                            res = self._run_hep_experiment(
                                defense_scope=defense_scope,
                                attack_target_tier="all",
                                attack_type=effective_atk,
                                byzantine_rate=q,
                                seed=seed,
                                method_label="Defended FedHEP",
                            )

                        _append_result_to_csv(csv_path, res)
                        completed.add(key)

        return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

    # -------------------------------------------------------------------------
    # POST-MIDTERM 3: Multi-Seed K-Cluster Sensitivity Sweep (Idea A)
    # -------------------------------------------------------------------------
    def run_postmid_k_clusters(
        self,
        k_values: List[int] = [1, 2, 3, 5, 8, 10, 15],
        alphas: List[float] = [0.05, 0.1, 0.5],
        seeds: List[int] = [42, 123, 7],
        byzantine_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        Executes Post-Midterm Study A: Multi-seed statistical significance sweep over
        number of peer clusters K across extreme to moderate label skews.
        Saves continuously to {output_dir}/{dataset}_postmid_k_clusters.csv.
        """
        csv_path = os.path.join(self.output_dir, f"{self.dataset}_postmid_k_clusters.csv")
        key_cols = ["num_clusters", "alpha", "byzantine_rate", "seed"]
        completed = _load_existing_keys(csv_path, key_cols)

        print("\n" + "=" * 70)
        print(">> [POST-MIDTERM] RUNNING MULTI-SEED PEER CLUSTERS (K) SWEEP")
        print(f" Dataset: {self.dataset.upper()} | Rounds: {self.num_rounds} | Device: {self.device}")
        print(f" K Values: {k_values} | Alphas: {alphas} | Seeds: {seeds}")
        print(f" Output File: {csv_path}")
        print("=" * 70)

        for alpha in alphas:
            for k in k_values:
                for seed in seeds:
                    key = tuple(_normalize_key_item(col, val) for col, val in zip(key_cols, [k, alpha, byzantine_rate, seed]))
                    if key in completed:
                        print(f">> [Skip/Done] K={k} | alpha={alpha} | Seed={seed}")
                        continue

                    res = self._run_hep_experiment(
                        defense_scope="both" if byzantine_rate > 0 else "none",
                        attack_target_tier="all",
                        attack_type="label_flip" if byzantine_rate > 0 else "none",
                        byzantine_rate=byzantine_rate,
                        seed=seed,
                        num_clusters_override=k,
                        alpha_override=alpha,
                    )
                    _append_result_to_csv(csv_path, res)
                    completed.add(key)

        return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

    def run_postmid_ttt_ablation(
        self,
        attacks: List[str] = ["sign_flip", "alie"],
        byzantine_rates: List[float] = [0.2, 0.3],
        ttt_options: List[bool] = [True, False],
        seeds: List[int] = [42, 123],
        defense_scope: str = "both",
    ) -> pd.DataFrame:
        """
        Executes Post-Midterm TTT Removal / Ablation Study:
        Empirically compares FedHEP with Temporal Trust Tracking (TTT) ON vs OFF to verify
        the hypothesis of zero/negligible contribution due to the static 0.25 threshold.
        Saves continuously to {output_dir}/{dataset}_postmid_ttt_ablation.csv.
        """
        csv_path = os.path.join(self.output_dir, f"{self.dataset}_postmid_ttt_ablation.csv")
        key_cols = ["method", "attack_type", "byzantine_rate", "enable_ttt", "seed"]
        completed = _load_existing_keys(csv_path, key_cols)

        print("\n" + "=" * 70)
        print(">> [POST-MIDTERM] RUNNING TTT ABLATION STUDY (ON vs OFF)")
        print(f" Dataset: {self.dataset.upper()} | Rounds: {self.num_rounds} | Device: {self.device}")
        print(f" Attacks: {attacks} | Byzantine Rates: {byzantine_rates} | TTT Options: {ttt_options}")
        print(f" Seeds: {seeds} | Defense Scope: {defense_scope}")
        print(f" Output File: {csv_path}")
        print("=" * 70)

        for atk in attacks:
            for q in byzantine_rates:
                for enable_ttt in ttt_options:
                    for seed in seeds:
                        key = (
                            _normalize_key_item("method", "fedhep"),
                            _normalize_key_item("attack_type", atk),
                            _normalize_key_item("byzantine_rate", q),
                            _normalize_key_item("enable_ttt", enable_ttt),
                            _normalize_key_item("seed", seed),
                        )
                        if key in completed:
                            print(f">> [Skip/Done] Atk={atk} | q={q} | enable_ttt={enable_ttt} | Seed={seed}")
                            continue

                        res = self._run_hep_experiment(
                            defense_scope=defense_scope,
                            attack_target_tier="all",
                            attack_type=atk,
                            byzantine_rate=q,
                            seed=seed,
                            enable_ttt=enable_ttt,
                        )
                        res["enable_ttt"] = enable_ttt
                        res["defense_scope"] = defense_scope
                        _append_result_to_csv(csv_path, res)
                        completed.add(key)

        return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

    # -------------------------------------------------------------------------
    # POST-MIDTERM 4: K-Cluster Sensitivity under Attack (10N / 15N / 20N)
    # -------------------------------------------------------------------------
    def run_postmid_k_attack_sensitivity(
        self,
        client_k_grid: Optional[Dict[int, List[int]]] = None,
        attack_type: str = "alie",
        byzantine_rate: float = 0.2,
        seeds: List[int] = [42, 123],
        alpha: Optional[float] = None,
        defense_scope: str = "both",
        enable_ttt: bool = True,
        output_filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Evaluates K-sensitivity under Byzantine attack across client counts N
        (e.g. 10N, 15N, 20N) to test the hypothesis K_opt ~ 20-25% of N.
        Saves continuously to {output_dir}/{output_filename or f'{dataset}_postmid_k_attack.csv'}.
        """
        grid = client_k_grid or {10: [2, 3], 15: [2, 3, 4], 20: [4, 5]}
        alpha_val = self.alpha if alpha is None else alpha
        csv_path = os.path.join(self.output_dir, output_filename or f"{self.dataset}_postmid_k_attack.csv")
        key_cols = ["num_clients", "num_clusters", "attack_type", "byzantine_rate", "alpha", "seed"]
        completed = _load_existing_keys(csv_path, key_cols)

        print("\n" + "=" * 70)
        print(">> [POST-MIDTERM] RUNNING K-SENSITIVITY UNDER ATTACK (10N / 15N / 20N)")
        print(f" Dataset: {self.dataset.upper()} | Rounds: {self.num_rounds} | Device: {self.device}")
        print(f" Grid: {grid} | Atk: {attack_type} | q={byzantine_rate} | alpha={alpha_val} | Seeds: {seeds}")
        print(f" Clustering Mode: {self.clustering_mode} | Output File: {csv_path}")
        print("=" * 70)

        for n_clients, k_list in grid.items():
            for k in k_list:
                for seed in seeds:
                    key = tuple(
                        _normalize_key_item(col, val)
                        for col, val in zip(key_cols, [n_clients, k, attack_type, byzantine_rate, alpha_val, seed])
                    )
                    if key in completed:
                        print(f">> [Skip/Done] N={n_clients} | K={k} | Atk={attack_type} | q={byzantine_rate} | Seed={seed}")
                        continue

                    res = self._run_hep_experiment(
                        defense_scope=defense_scope,
                        attack_target_tier="all",
                        attack_type=attack_type,
                        byzantine_rate=byzantine_rate,
                        seed=seed,
                        method_label="Defended FedHEP",
                        num_clusters_override=k,
                        alpha_override=alpha_val,
                        num_clients_override=n_clients,
                        enable_ttt=enable_ttt,
                    )
                    row = {
                        "num_clients": n_clients,
                        "k_over_n_pct": round(100.0 * k / n_clients, 1),
                        **res,
                    }
                    _append_result_to_csv(csv_path, row)
                    completed.add(key)

        return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

    # -------------------------------------------------------------------------
    # Helper Simulation Dispatchers
    # -------------------------------------------------------------------------
    def _run_hep_experiment(
        self,
        defense_scope: str,
        attack_target_tier: str,
        attack_type: str,
        byzantine_rate: float,
        seed: int = 42,
        alie_z_max: Optional[float] = None,
        method_label: str = "FedHEP",
        num_clusters_override: Optional[int] = None,
        alpha_override: Optional[float] = None,
        num_clients_override: Optional[int] = None,
        ipm_epsilon: float = 0.1,
        attack_onset_round: int = 15,
        enable_ttt: bool = True,
        clustering_mode_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        set_seed(seed)
        n_clients = num_clients_override or self.num_clients
        k_clusters = num_clusters_override or self.num_clusters
        alpha = alpha_override if alpha_override is not None else self.alpha
        active_clustering_mode = clustering_mode_override or self.clustering_mode
        n_suffix = f"_n{n_clients}" if num_clients_override is not None else ""
        cm_suffix = f"_cm_{active_clustering_mode}_s{seed}" if active_clustering_mode != "none" else ""
        exp_name = f"{self.dataset}_hep_s_{defense_scope}_tgt_{attack_target_tier}_{attack_type}_q{int(byzantine_rate*100)}_k{k_clusters}_a{alpha}{n_suffix}{cm_suffix}"

        print(f"\n>> HEP Run: Scope={defense_scope} | Target={attack_target_tier} | Atk={attack_type} | q={byzantine_rate} | N={n_clients} | K={k_clusters} | a={alpha} | Cluster={active_clustering_mode} | Seed={seed}")

        cfg = SimulationConfig(
            experiment_name=exp_name,
            num_rounds=self.num_rounds,
            eval_interval=max(1, self.num_rounds // 5),
            env=EnvironmentConfig(
                output_dir=self.output_dir,
                seed=seed,
                dataset=self.dataset,
                train_subset=self.train_subset,
                test_subset=self.test_subset,
                data_dir=self.data_dir,
            ),
            topology=TopologyConfig(
                type="hierarchical_ensemble",
                params={
                    "num_clusters": k_clusters,
                    "cluster_method": "update_similarity",
                    "defense_mode": "soft_cosine" if defense_scope != "none" else "none",
                    "defense_scope": defense_scope,
                },
            ),
            clients=ClientConfig(
                num_clients=n_clients,
                model_name="resnet9",
                local_lr=0.05,
                local_steps=3,
                total_local_steps=5,
                compute_optimization_mode="shared_backbone",
                use_ensemble=True,
                hierarchical_ensemble=True,
                high_cardinality_two_stage=True if self.dataset == "cifar100" else False,
            ),
            robustness=RobustnessConfig(
                byzantine_rate=byzantine_rate,
                byzantine_type=attack_type,
            ),
            non_iid=NonIIDConfig(
                enabled=True,
                alpha=alpha,
            ),
        )

        topology = HierarchicalTopology(num_clusters=k_clusters)
        aggregator = FedAvgAggregator()
        defense_cfg = DefenseConfig(
            defense_mode="soft_cosine" if defense_scope != "none" else "none",
            temperature=0.5,
            defense_scope=defense_scope if defense_scope in ("cluster", "global", "both") else "cluster",
            norm_threshold=2.0,
            hard_rejection_enabled=True,
            hard_rejection_threshold=-0.1,
        )

        engine = OrehDefendedEngine(
            config=cfg,
            topology=topology,
            aggregator=aggregator,
            device=self.device,
            defense_config=defense_cfg,
            attack_target_tier=attack_target_tier,
            alie_z_max=alie_z_max,
            ipm_epsilon=ipm_epsilon,
            attack_onset_round=attack_onset_round,
            enable_ttt=enable_ttt,
            clustering_mode=active_clustering_mode,
        )

        t_start = time.perf_counter()
        engine.run()
        total_time_s = time.perf_counter() - t_start

        history = engine.metrics.get_history()
        final_acc = 0.0
        final_b10 = 0.0
        if history:
            for rec in reversed(history):
                if rec.get("evaluated", True):
                    final_acc = float(rec.get("personalized_test_acc", rec.get("global_test_acc", rec.get("test_accuracy", 0.0))))
                    final_b10 = float(rec.get("bottom_10_acc", rec.get("bottom10_fairness", 0.0)))
                    break

        mean_tier1_lat = float(np.mean(engine.cluster_latency_history)) if engine.cluster_latency_history else 0.0
        mean_tier2_lat = float(np.mean(engine.global_latency_history)) if engine.global_latency_history else 0.0
        mean_total_def_lat = float(np.mean(engine.total_defense_latency_history)) if engine.total_defense_latency_history else 0.0
        byz_trust = float(np.mean(engine.byzantine_trust_history[-3:])) if engine.byzantine_trust_history else 0.0
        hon_trust = float(np.mean(engine.honest_trust_history[-3:])) if engine.honest_trust_history else 1.0

        res = {
            "dataset": self.dataset,
            "method": method_label,
            "defense_scope": defense_scope,
            "attack_target_tier": attack_target_tier,
            "attack_type": attack_type,
            "byzantine_rate": byzantine_rate,
            "num_clusters": k_clusters,
            "alpha": alpha,
            "seed": seed,
            "final_test_acc": round(final_acc, 2),
            "bottom_10_fairness": round(final_b10, 2),
            "tier1_latency_ms": round(mean_tier1_lat, 2),
            "tier2_latency_ms": round(mean_tier2_lat, 2),
            "total_defense_latency_ms": round(mean_total_def_lat, 2),
            "byzantine_trust_score": round(byz_trust, 4),
            "honest_trust_score": round(hon_trust, 4),
            "total_runtime_s": round(total_time_s, 2),
        }
        if active_clustering_mode != "none":
            res["clustering_mode"] = active_clustering_mode
            res["clustering_events"] = len(getattr(engine, "clustering_history", []))
            res["exp_name"] = exp_name
        return res

    def _run_star_baseline_experiment(
        self,
        method_id: str,
        attack_type: str,
        byzantine_rate: float,
        alie_z_max: Optional[float] = None,
        seed: int = 42,
        ipm_epsilon: float = 0.1,
        attack_onset_round: int = 15,
        num_clients_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        set_seed(seed)
        n_clients = num_clients_override or self.num_clients
        canon_method = _canonical_star_method(method_id)
        n_suffix = f"_n{n_clients}" if num_clients_override is not None else ""
        exp_name = f"{self.dataset}_star_{canon_method}_{attack_type}_q{int(byzantine_rate*100)}{n_suffix}_s{seed}"
        print(f"\n>> Star Baseline Run: Method={canon_method.upper()} | Atk={attack_type} | q={byzantine_rate} | N={n_clients} | Seed={seed}")

        is_ditto = canon_method == "ditto"
        client_cfg = ClientConfig(
            num_clients=n_clients,
            model_name="resnet9",
            local_lr=0.05,
            local_steps=3,
            total_local_steps=5,
            personalization_method="ditto" if is_ditto else "none",
            ditto_lambda=0.05 if is_ditto else 0.0,
            compute_optimization_mode="none",
        )

        cfg = SimulationConfig(
            experiment_name=exp_name,
            num_rounds=self.num_rounds,
            eval_interval=max(1, self.num_rounds // 5),
            env=EnvironmentConfig(
                output_dir=self.output_dir,
                seed=seed,
                dataset=self.dataset,
                train_subset=self.train_subset,
                test_subset=self.test_subset,
                data_dir=self.data_dir,
            ),
            topology=TopologyConfig(type="star"),
            clients=client_cfg,
            robustness=RobustnessConfig(
                byzantine_rate=byzantine_rate,
                byzantine_type=attack_type,
            ),
            non_iid=NonIIDConfig(
                enabled=True,
                alpha=self.alpha,
            ),
        )

        topology = StarTopology()
        if canon_method == "multikrum":
            f_expected = int(np.ceil(n_clients * byzantine_rate))
            m_selected = max(1, min(3, n_clients))
            aggregator = MultiKrumAggregator(f=f_expected, m=m_selected)
        elif canon_method == "median":
            aggregator = CoordinateMedianAggregator()
        elif canon_method == "trimmed_mean":
            aggregator = TrimmedMeanAggregator(beta=0.20)
        else:
            aggregator = FedAvgAggregator()

        engine = OrehCentralizedEngine(
            config=cfg,
            topology=topology,
            aggregator=aggregator,
            device=self.device,
            alie_z_max=alie_z_max,
            ipm_epsilon=ipm_epsilon,
            attack_onset_round=attack_onset_round,
        )

        t_start = time.perf_counter()
        engine.run()
        total_time_s = time.perf_counter() - t_start

        history = engine.metrics.get_history()
        final_acc = 0.0
        final_b10 = 0.0
        if history:
            for rec in reversed(history):
                if rec.get("evaluated", True):
                    if is_ditto:
                        final_acc = float(rec.get("ensemble_test_accuracy", rec.get("test_accuracy", 0.0)))
                    else:
                        final_acc = float(rec.get("test_accuracy", 0.0))
                    final_b10 = float(rec.get("bottom10_fairness", rec.get("bottom_10_acc", 0.0)))
                    break

        return {
            "dataset": self.dataset,
            "method": canon_method.upper(),
            "defense_scope": "star_centralized",
            "attack_target_tier": "root_only",
            "attack_type": attack_type,
            "byzantine_rate": byzantine_rate,
            "num_clusters": 1,
            "alpha": self.alpha,
            "seed": seed,
            "final_test_acc": round(final_acc, 2),
            "bottom_10_fairness": round(final_b10, 2),
            "tier1_latency_ms": 0.0,
            "tier2_latency_ms": 0.0,
            "total_defense_latency_ms": 0.0,
            "byzantine_trust_score": 1.0,
            "honest_trust_score": 1.0,
            "total_runtime_s": round(total_time_s, 2),
        }
