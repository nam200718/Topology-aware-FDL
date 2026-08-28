import os
import re
import yaml
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Optional, List, Union

TopologyType = Literal["star", "star_cfl", "star_randomized", "ring", "hierarchical", "hierarchical_ensemble", "gossip", "layered"]

class EnvironmentConfig(BaseModel):
    # Where to save outputs
    output_dir: str = "./outputs"
    # Seed for global reproducibility
    seed: int = 42
    # Optional subset sizes for faster experiments
    train_subset: Optional[int] = None
    test_subset: Optional[int] = None
    # Dataset to use ("mnist", "cifar10" or "cifar100")
    dataset: Literal["mnist", "cifar10", "cifar100"] = "mnist"


class TopologyConfig(BaseModel):
    type: TopologyType = "star"
    # Additional topology-specific parameters (e.g., degree for gossip, clusters for hierarchical)
    params: Dict[str, Any] = Field(default_factory=dict)

class ClientConfig(BaseModel):
    num_clients: int = 100
    # Model architecture selection: "simple_cnn", "resnet9" or "mobilenetv3"
    model_name: Literal["simple_cnn", "resnet9", "mobilenetv3"] = "simple_cnn"
    # Approximate model dimensions for the lightweight vector update model
    model_dim: int = 50
    # learning rate or step size for local updates
    local_lr: float = 0.1
    # number of local steps per round
    local_steps: int = 5
    # Use Local-Global Model Ensemble Personalization
    use_ensemble: bool = False
    # Use Hierarchical Ensemble (Root, Parent, Local)
    hierarchical_ensemble: bool = False
    # Weight of local model in ensemble (0.0=global only, 1.0=local only)
    ensemble_alpha: float = 0.5
    # Weight of parent model in ensemble (if hierarchical_ensemble=True)
    ensemble_beta: float = 0.2
    # Ensemble weighting strategy: "static", "dynamic_confidence", "dynamic_loss"
    ensemble_weighting_mode: Literal["static", "dynamic_confidence", "dynamic_loss"] = "dynamic_confidence"
    # Compute optimization strategy: "none", "shared_backbone", "frozen_root_anchor", "head_only"
    compute_optimization_mode: Literal["none", "shared_backbone", "frozen_root_anchor", "head_only"] = "shared_backbone"
    # Enable inter-model mutual distillation during local updates (evaluated in ablation studies)
    ensemble_distillation: bool = False
    # Scaling factor for mutual distillation loss
    distillation_lambda: float = 0.0
    # Total local step budget per round to distribute adaptively across ensemble heads.
    # Default 5 validated >= 10 across IID/moderate skew at ~2x less wall time
    # (outputs/comparison_study_20260822_005556, _013747); keeps HEP compute
    # comparable to baselines training local_steps=3.
    total_local_steps: int = 5
    # Loss scaling weight beta for loss-calibrated ensemble weighting
    loss_weight_beta: float = 1.0
    # Softmax temperature for dynamic ensemble weighting (1.0 = smooth blending, 0.1 = sharp)
    ensemble_temperature: float = 1.0
    # Use Shannon label entropy to dynamically calibrate ensemble simplex prior across IID regimes
    use_heterogeneity_prior: bool = True
    # Exponential scaling for heterogeneity prior (1.0 = linear)
    heterogeneity_gamma: float = 1.0
    # Algorithm personalization method: "none", "ditto", "apfl", "fedala",
    # "fedrep", "fedper", "fedbabu"
    personalization_method: Literal["none", "ditto", "apfl", "fedala", "fedrep", "fedper", "fedbabu"] = "none"
    # Ditto proximal L2 penalty parameter lambda
    ditto_lambda: float = 0.1
    # APFL initial mixing weight alpha
    apfl_alpha: float = 0.5
    # FedRep: head-only epochs before joint training each round (total local
    # passes == local_steps for compute parity with other baselines)
    fedrep_head_epochs: int = 1
    # FedBABU (canonical): deployment-time head fine-tuning SGD steps before
    # per-client evaluation; head stays frozen during federated rounds
    babu_finetune_steps: int = 100
    # FedALA adaptive local aggregation (official defaults from TsingZ0/FedALA)
    ala_rand_percent: int = 80
    ala_layer_idx: int = 2
    ala_eta: float = 1.0
    ala_threshold: float = 0.1
    ala_num_pre_loss: int = 10
    ala_max_epochs: int = 50

    # ------------------------------------------------------------------
    # Previously-inlined hyperparameters (hoisted during clean-refactor).
    # Defaults reproduce legacy behavior exactly; provenance noted inline.
    # ------------------------------------------------------------------
    # Cosine LR-decay floor (was an undeclared getattr fallback in BaseEngine).
    min_lr: float = 0.001
    # Local mini-batch cap (legacy: batch_size = min(32, len(dataset))).
    local_batch_size: int = 32
    # Head-training schedule: "binomial" partition-of-unity (default; closes
    # IID gap +5.1pp and moderate-skew +3.2pp vs legacy, see
    # outputs/comparison_study_20260821_182734) or "piecewise" (legacy if/else
    # budgets, kept for paper ablation tables).
    head_training_schedule: Literal["binomial", "piecewise"] = "binomial"
    # Anchor floor for root-head loss weight under extreme skew (binomial only; None/0.0 uses dynamic anchor).
    binomial_anchor_min: float = 0.0
    # Cluster count K used by the dynamic anchor a_i = max(1/(2K), |Y_i|/C).
    # Overwritten by HierarchicalEnsembleEngine with the topology's actual
    # cluster count at init; default 3 matches the paper's canonical K.
    num_clusters: int = 3
    # Enable parameter-free dynamic self-adaptive parameters (Hill-number R_skew, dynamic anchor, Poisson staleness, Kalman momentum)
    dynamic_parameters: bool = True
    # Two-stage high-cardinality schedule: freeze backbone while training the
    # linear heads, then one root-anchored backbone epoch. Targets CIFAR-100 regimes.
    high_cardinality_two_stage: bool = False
    # Engages when dataset cardinality reaches this many classes
    # (learnings doc §3, Rec. 1).
    two_stage_min_classes: int = 100
    # Suppress the weakest head at moderate skew when cluster affinity is clear.
    top2_routing: bool = False
    # Minimum cosine affinity gap vs. second-best cluster to suppress a head.
    top2_affinity_margin: float = 0.10

    # --- Ensemble inference calibration (learnings doc §5.3: sharp-local /
    #     soft-global logit rescaling; validated inference-only) ---
    # Enable dynamic logit dispersion temperature matching across heads
    dynamic_temperature: bool = True
    eval_temp_local: float = 0.6
    eval_temp_parent: float = 0.8
    eval_temp_root: float = 1.0
    # Blend-weight learning rate scale: alpha_lr = local_lr * this factor.
    ensemble_alpha_lr_scale: float = 0.05
    # Final blend mixing: w_final = mix * learned + (1 - mix) * heterogeneity prior.
    prior_mix: float = 0.5
    # Mask unobserved classes at inference time during personalized client evaluation
    active_class_inference_mask: bool = True
    # Mask unobserved classes on Local & Parent heads during training
    active_class_loss_mask: bool = True
    # Hard IID routing cutoff on R_skew at evaluation (legacy shortcut; kept for
    # ablation parity until binomial schedule fully supersedes it).
    iid_route_threshold: float = 0.85

    # --- Robust aggregation (server-side; zero client cost) ---
    # "fedavg" | "trimmed_mean" | "soft_cosine"
    robust_aggregation_mode: Literal["fedavg", "trimmed_mean", "soft_cosine"] = "fedavg"
    # Trimmed-mean trimming ratio per coordinate.
    trimmed_mean_beta: float = 0.20
    # Softmax temperature for cosine-trust weights (lower = sharper rejection).
    soft_cosine_temperature: float = 0.5
    # Clamp update norms to k * lower-quartile norm before trust weighting.
    norm_bound_k: float = 3.0
    # Aggregate BN running stats via median instead of mean ("median") or ignore
    # them entirely ("server"); "mean" preserves legacy FedAvg behavior.
    buffer_aggregation: Literal["mean", "median"] = "mean"

    # --- Partial participation fairness (S-AFR) ---
    # Fraction of clients sampled per round (1.0 = full participation, legacy).
    participation_fraction: float = 1.0
    # Staleness-aware fallback routing: fade local/parent blend weight toward
    # the root head for clients not sampled recently (exp decay).
    s_afr_enabled: bool = False
    # Staleness (rounds) beyond which fade begins.
    s_afr_staleness_window: int = 4
    # Exponential decay constant for the staleness fade.
    s_afr_fade_tau: float = 4.0
    # Server-side cluster-head momentum: w_new = beta * w_old + (1-beta) * avg.
    # 0.0 disables momentum (legacy FedAvg replacement semantics).
    cluster_momentum_beta: float = 0.0

class RobustnessConfig(BaseModel):
    byzantine_rate: float = 0.0
    # Attack options: 'label_flip'
    byzantine_type: str = "label_flip"

class NonIIDConfig(BaseModel):
    # If True, partition data non-IID
    enabled: bool = True
    # The Dirichlet distribution concentration parameter.
    # Smaller value = more non-IID.
    alpha: float = 0.5

class SimulationConfig(BaseModel):
    experiment_name: str = "baseline_run"
    num_rounds: int = 100
    # Evaluate metrics every N rounds (final round is always evaluated).
    # Evaluation consumes no RNG and does not modify model state, so this
    # changes monitoring granularity only - never training trajectories.
    eval_interval: int = 1
    
    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    clients: ClientConfig = Field(default_factory=ClientConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    non_iid: NonIIDConfig = Field(default_factory=NonIIDConfig)

    def ensure_output_dir(self):
        full_path = os.path.join(self.env.output_dir, self.experiment_name)
        os.makedirs(full_path, exist_ok=True)
        return full_path


# --- Experiment-level config (loaded from YAML) ---

ExperimentType = Literal["single", "comparison", "byzantine_matrix"]

class TopologyEntry(BaseModel):
    """One topology to include in a sweep experiment."""
    type: TopologyType
    label: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)

class ScenarioEntry(BaseModel):
    """One scenario (e.g. IID baseline, Non-IID, Attack) for comparison experiments."""
    id: str
    label: str
    non_iid: bool = False
    alpha: Optional[float] = None
    byzantine_rate: float = 0.0

class ExperimentConfig(BaseModel):
    """
    Top-level config loaded from a YAML file.
    Holds experiment-wide defaults and sweep axes.
    Can generate one or more SimulationConfig instances via build_configs().
    """
    experiment_type: ExperimentType = "single"
    num_rounds: int = 10
    # Evaluate every N-th round plus the final round. Forwarded to each
    # SimulationConfig so should_evaluate() honors YAML eval_interval keys.
    eval_interval: int = 1

    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    clients: ClientConfig = Field(default_factory=ClientConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    non_iid: NonIIDConfig = Field(default_factory=NonIIDConfig)

    # For single experiments — a single topology
    topology: Optional[TopologyEntry] = None
    # For sweep experiments — a list of topologies
    topologies: List[TopologyEntry] = Field(default_factory=list)

    # For comparison experiments — scenario definitions
    scenarios: List[ScenarioEntry] = Field(default_factory=list)
    # For byzantine_matrix experiments — rates to sweep
    byzantine_rates: List[float] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load an ExperimentConfig from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def _make_client_config(self, topo_type: str, overrides: Optional[dict] = None) -> ClientConfig:
        """
        Build a ClientConfig from defaults, auto-setting ensemble flags
        based on the topology type and applying topo-specific overrides.
        """
        if overrides is None:
            overrides = {}
        base = self.clients.model_dump()
        is_ensemble = (topo_type == "hierarchical_ensemble") or (overrides.get("compute_optimization_mode") == "shared_backbone")
        base["use_ensemble"] = is_ensemble or base.get("use_ensemble", False)
        base["hierarchical_ensemble"] = (topo_type == "hierarchical_ensemble") or base.get("hierarchical_ensemble", False)

        valid_keys = set(ClientConfig.model_fields.keys())
        for k, v in overrides.items():
            if k in valid_keys:
                base[k] = v

        # Align the compute mode with architecture reality. A globally-set
        # "shared_backbone" mode must not leak into entries that never train a
        # multi-head model, otherwise the engine seeds those clients with
        # multi-head-sized initial weights (mixed-size aggregation crash).
        # APFL intentionally routes through the canonical dual-model path:
        # the shared-trunk fast path deviates from Deng et al. (2020), where
        # global and local models are independent.
        method = base.get("personalization_method", "none")
        routes_multihead = (
            topo_type == "hierarchical_ensemble"
            or base.get("use_ensemble")
            or base.get("hierarchical_ensemble")
        )
        if not routes_multihead and base.get("compute_optimization_mode") == "shared_backbone":
            base["compute_optimization_mode"] = "none"

        return ClientConfig(**base)

    def build_configs(self, metrics_dir: Optional[str] = None) -> List[dict]:
        """
        Generate a list of experiment entries from this config.

        Each entry is a dict with:
          - "config": SimulationConfig ready to pass to run_experiment()
          - "topo_label": human-readable topology label
          - "scenario_id": scenario id (for comparison) or None
          - "scenario_label": scenario label (for comparison) or None
          - "byzantine_rate": the rate used (for byzantine_matrix) or None

        The metrics_dir parameter overrides env.output_dir for the generated
        SimulationConfig instances (used by scripts that create timestamped dirs).
        """
        output_dir = metrics_dir or self.env.output_dir
        env = self.env.model_copy(update={"output_dir": output_dir})
        entries = []

        if self.experiment_type == "single":
            topo = self.topology or (self.topologies[0] if self.topologies else TopologyEntry(type="star", label="Star"))
            label = topo.label or topo.type.replace("_", " ").capitalize()
            exp_name = f"{topo.type}_single"
            sim_config = SimulationConfig(
                experiment_name=exp_name,
                num_rounds=self.num_rounds,
                eval_interval=max(1, self.eval_interval),
                env=env,
                topology=TopologyConfig(type=topo.type, params=topo.params),
                clients=self._make_client_config(topo.type, topo.params),
                robustness=self.robustness,
                non_iid=self.non_iid,
            )
            entries.append({
                "config": sim_config,
                "topo_label": label,
                "scenario_id": None,
                "scenario_label": None,
                "byzantine_rate": self.robustness.byzantine_rate,
            })

        elif self.experiment_type == "byzantine_matrix":
            for topo in self.topologies:
                label = topo.label or topo.type.replace("_", " ").capitalize()
                for rate in self.byzantine_rates:
                    exp_name = f"{topo.type}_byz_{int(rate * 100)}"
                    sim_config = SimulationConfig(
                        experiment_name=exp_name,
                        num_rounds=self.num_rounds,
                        eval_interval=max(1, self.eval_interval),
                        env=env,
                        topology=TopologyConfig(type=topo.type, params=topo.params),
                        clients=self._make_client_config(topo.type, topo.params),
                        robustness=RobustnessConfig(
                            byzantine_rate=rate,
                            byzantine_type=self.robustness.byzantine_type,
                        ),
                        non_iid=self.non_iid,
                    )
                    entries.append({
                        "config": sim_config,
                        "topo_label": label,
                        "scenario_id": None,
                        "scenario_label": None,
                        "byzantine_rate": rate,
                    })

        elif self.experiment_type == "comparison":
            for topo in self.topologies:
                label = topo.label or topo.type.replace("_", " ").capitalize()
                # Build a unique experiment name using a sanitized label to avoid collisions
                # when multiple topologies share the same type (e.g. star_fedavg vs star_ditto).
                safe_label = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
                for scenario in self.scenarios:
                    exp_name = f"{safe_label}_{scenario.id}"
                    sim_config = SimulationConfig(
                        experiment_name=exp_name,
                        num_rounds=self.num_rounds,
                eval_interval=max(1, self.eval_interval),
                        env=env,
                        topology=TopologyConfig(type=topo.type, params=topo.params),
                        clients=self._make_client_config(topo.type, topo.params),
                        robustness=RobustnessConfig(
                            byzantine_rate=scenario.byzantine_rate,
                            byzantine_type=self.robustness.byzantine_type,
                        ),
                        non_iid=NonIIDConfig(
                            enabled=scenario.non_iid,
                            alpha=scenario.alpha if scenario.alpha is not None else self.non_iid.alpha,
                        ),
                    )
                    entries.append({
                        "config": sim_config,
                        "topo_label": label,
                        "scenario_id": scenario.id,
                        "scenario_label": scenario.label,
                        "byzantine_rate": scenario.byzantine_rate,
                    })

        return entries

