"""Experiment definitions and configurations for CIFAR-100 baseline comparisons.

Restricted to exactly the 5 target baselines + proposed Topo method:
1. FedAvg
2. FedProx
3. Multi-Krum
4. SCAFFOLD
5. Ditto
6. Proposed Topo (HEP / Defended H-ResFL)
"""
from typing import Dict, Any, List, Optional
from src.baselines.config import BaselineSimulationConfig, BaselineClientConfig
from src.config import TopologyConfig, RobustnessConfig, NonIIDConfig, EnvironmentConfig

# ============================================================
# DATASET DEFAULTS: CIFAR-100 (100 classes, 3x32x32)
# ============================================================
CIFAR100_DEFAULTS = dict(
    dataset="cifar100",
    model_name="resnet9",
    num_clients=15,
    local_lr=0.05,
    local_steps=3,
    num_rounds=20,
    eval_interval=5,
    train_subset=10000,
    test_subset=3000,
)

# Full evaluation defaults (for final paper tables)
CIFAR100_FULL_DEFAULTS = dict(
    dataset="cifar100",
    model_name="resnet9",
    num_clients=15,
    local_lr=0.05,
    local_steps=3,
    num_rounds=50,
    eval_interval=5,
    train_subset=None,
    test_subset=None,
)

# ============================================================
# TARGET METHODS: Exactly 5 Baselines + Proposed Topo
# ============================================================
METHODS = [
    {
        "id": "fedavg",
        "label": "FedAvg",
        "topo": "star",
        "pers": "none",
        "description": "Canonical Federated Averaging (McMahan et al., 2017)",
    },
    {
        "id": "fedprox",
        "label": "FedProx",
        "topo": "star",
        "pers": "fedprox",
        "params": {"fedprox_mu": 0.01},
        "description": "Proximal regularization against client drift (Li et al., 2020)",
    },
    {
        "id": "multikrum",
        "label": "Multi-Krum",
        "topo": "star",
        "pers": "none",
        "params": {"krum_num_selected": 3},
        "description": "Byzantine-resilient Multi-Krum aggregation (Blanchard et al., 2017)",
    },
    {
        "id": "scaffold",
        "label": "SCAFFOLD",
        "topo": "star",
        "pers": "scaffold",
        "description": "Stochastic controlled averaging with control variates (Karimireddy et al., 2020)",
    },
    {
        "id": "ditto",
        "label": "Ditto",
        "topo": "star",
        "pers": "ditto",
        "params": {"ditto_lambda": 0.05},
        "description": "Personalized FL via local model proximal anchor (Li et al., 2021)",
    },
    {
        "id": "topo",
        "label": "Proposed Topo (HEP)",
        "topo": "hierarchical_ensemble",
        "pers": "none",
        "params": {"num_clusters": 3, "cluster_method": "update_similarity", "defense_mode": "none"},
        "description": "Topology-aware Hierarchical Ensemble Partitioning (Proposed in Paper)",
    },
    {
        "id": "topo_defended",
        "label": "Proposed Topo (Defended H-ResFL)",
        "topo": "hierarchical_ensemble",
        "pers": "none",
        "params": {"num_clusters": 3, "cluster_method": "update_similarity", "defense_mode": "soft_cosine"},
        "description": "Topology-aware Hierarchical Ensemble with Byzantine Defense (Proposed)",
    },
]

# Personalization benchmark subset (excludes defense-only variant)
PERSONALIZATION_METHODS = ["fedavg", "fedprox", "multikrum", "scaffold", "ditto", "topo"]

# Byzantine benchmark methods
BYZANTINE_METHODS = ["fedavg", "fedprox", "multikrum", "scaffold", "ditto", "topo", "topo_defended"]

# ============================================================
# HETEROGENEITY REGIMES (5 Dirichlet concentration levels)
# ============================================================
REGIMES = [
    {"id": "iid",      "label": "IID",                "non_iid": False, "alpha": 1.0},
    {"id": "mild",     "label": "Mild (alpha=1.0)",     "non_iid": True,  "alpha": 1.0},
    {"id": "moderate", "label": "Moderate (alpha=0.5)", "non_iid": True,  "alpha": 0.5},
    {"id": "severe",   "label": "Severe (alpha=0.1)",   "non_iid": True,  "alpha": 0.1},
    {"id": "extreme",  "label": "Extreme (alpha=0.05)", "non_iid": True,  "alpha": 0.05},
]

# ============================================================
# BYZANTINE ATTACK SUITE (All 4 attack types supported by codebase)
# ============================================================
ATTACK_TYPES = [
    {"id": "label_flip",      "label": "Label Flipping"},
    {"id": "sign_flip",       "label": "Sign Flipping"},
    {"id": "gradient_ascent", "label": "Gradient Ascent"},
    {"id": "random_noise",    "label": "Gaussian Noise"},
]

BYZANTINE_RATES = [0.0, 0.1, 0.2, 0.3, 0.4]

# Seeds for multi-seed statistical significance
SEEDS = [42, 123, 7]


def get_method_meta(method_id: str) -> Dict[str, Any]:
    """Retrieve metadata dict for a method ID."""
    for m in METHODS:
        if m["id"] == method_id:
            return m
    raise ValueError(f"Unknown method ID: {method_id}")


def get_regime_meta(regime_id: str) -> Dict[str, Any]:
    """Retrieve metadata dict for a heterogeneity regime ID."""
    for r in REGIMES:
        if r["id"] == regime_id:
            return r
    raise ValueError(f"Unknown regime ID: {regime_id}")


def create_personalization_config(
    method_id: str,
    regime_id: str,
    base_defaults: Optional[Dict[str, Any]] = None,
    output_dir: str = "./outputs/baselines_personalization",
) -> BaselineSimulationConfig:
    """Create simulation config for Personalization & Heterogeneity benchmark."""
    defaults = dict(CIFAR100_DEFAULTS)
    if base_defaults:
        defaults.update(base_defaults)

    method = get_method_meta(method_id)
    regime = get_regime_meta(regime_id)

    exp_name = f"cifar100_{method_id}_{regime_id}"

    # Build client config
    client_kwargs = {
        "num_clients": defaults["num_clients"],
        "model_name": defaults["model_name"],
        "local_lr": defaults["local_lr"],
        "local_steps": defaults["local_steps"],
        "personalization_method": method["pers"],
    }
    if method_id in ("topo", "topo_defended"):
        client_kwargs["use_ensemble"] = True
        client_kwargs["hierarchical_ensemble"] = True
        client_kwargs["compute_optimization_mode"] = "shared_backbone"
    else:
        client_kwargs["compute_optimization_mode"] = "none"

    if "params" in method:
        for k, v in method["params"].items():
            if k in ("fedprox_mu", "ditto_lambda", "krum_num_selected", "krum_num_byzantine"):
                client_kwargs[k] = v

    # Build topology config
    topo_type = method["topo"]
    topo_params = {}
    if topo_type == "hierarchical_ensemble":
        topo_params["num_clusters"] = method.get("params", {}).get("num_clusters", 3)
        topo_params["cluster_method"] = method.get("params", {}).get("cluster_method", "update_similarity")
        topo_params["defense_mode"] = method.get("params", {}).get("defense_mode", "none")

    config = BaselineSimulationConfig(
        experiment_name=exp_name,
        num_rounds=defaults["num_rounds"],
        eval_interval=defaults["eval_interval"],
        env=EnvironmentConfig(
            output_dir=output_dir,
            seed=42,
            dataset=defaults["dataset"],
            train_subset=defaults.get("train_subset"),
            test_subset=defaults.get("test_subset"),
        ),
        topology=TopologyConfig(
            type=topo_type,
            params=topo_params,
        ),
        clients=BaselineClientConfig(**client_kwargs),
        robustness=RobustnessConfig(
            byzantine_rate=0.0,
            byzantine_type="label_flip",
        ),
        non_iid=NonIIDConfig(
            enabled=regime["non_iid"],
            alpha=regime["alpha"] if regime["alpha"] is not None else 0.5,
        ),
    )
    return config


def create_byzantine_config(
    method_id: str,
    attack_type: str,
    byzantine_rate: float,
    regime_id: str = "moderate",
    base_defaults: Optional[Dict[str, Any]] = None,
    output_dir: str = "./outputs/baselines_byzantine",
) -> BaselineSimulationConfig:
    """Create simulation config for Byzantine Robustness benchmark."""
    defaults = dict(CIFAR100_DEFAULTS)
    if base_defaults:
        defaults.update(base_defaults)

    method = get_method_meta(method_id)
    regime = get_regime_meta(regime_id)

    exp_name = f"cifar100_byz_{method_id}_{attack_type}_f{int(byzantine_rate * 100)}"

    client_kwargs = {
        "num_clients": defaults["num_clients"],
        "model_name": defaults["model_name"],
        "local_lr": defaults["local_lr"],
        "local_steps": defaults["local_steps"],
        "personalization_method": method["pers"],
    }
    if method_id in ("topo", "topo_defended"):
        client_kwargs["use_ensemble"] = True
        client_kwargs["hierarchical_ensemble"] = True
        client_kwargs["compute_optimization_mode"] = "shared_backbone"
    else:
        client_kwargs["compute_optimization_mode"] = "none"

    if "params" in method:
        for k, v in method["params"].items():
            if k in ("fedprox_mu", "ditto_lambda", "krum_num_selected", "krum_num_byzantine"):
                client_kwargs[k] = v

    topo_type = method["topo"]
    topo_params = {}
    if topo_type == "hierarchical_ensemble":
        topo_params["num_clusters"] = method.get("params", {}).get("num_clusters", 3)
        topo_params["cluster_method"] = method.get("params", {}).get("cluster_method", "update_similarity")
        topo_params["defense_mode"] = method.get("params", {}).get("defense_mode", "none")

    config = BaselineSimulationConfig(
        experiment_name=exp_name,
        num_rounds=defaults["num_rounds"],
        eval_interval=defaults["eval_interval"],
        env=EnvironmentConfig(
            output_dir=output_dir,
            seed=42,
            dataset=defaults["dataset"],
            train_subset=defaults.get("train_subset"),
            test_subset=defaults.get("test_subset"),
        ),
        topology=TopologyConfig(
            type=topo_type,
            params=topo_params,
        ),
        clients=BaselineClientConfig(**client_kwargs),
        robustness=RobustnessConfig(
            byzantine_rate=byzantine_rate,
            byzantine_type=attack_type,
        ),
        non_iid=NonIIDConfig(
            enabled=regime["non_iid"],
            alpha=regime["alpha"] if regime["alpha"] is not None else 0.5,
        ),
    )
    return config
