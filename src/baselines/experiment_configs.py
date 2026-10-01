"""Experiment definitions and configurations for CIFAR-100 baseline comparisons.

Restricted to exactly the 5 target baselines + proposed Topo method:
1. FedAvg
2. FedProx
3. Multi-Krum
4. SCAFFOLD
5. Ditto
6. Proposed Topo (HEP-FL / Defended HEP-FL)
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
        "id": "fedrep",
        "label": "FedRep",
        "topo": "star",
        "pers": "fedrep",
        "params": {"fedrep_head_epochs": 1},
        "description": "Federated representation learning with persistent private classifier head (Collins et al., 2021)",
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
        "label": "Proposed Topo (Defended HEP-FL)",
        "topo": "hierarchical_ensemble",
        "pers": "none",
        "params": {"num_clusters": 3, "cluster_method": "update_similarity", "defense_mode": "soft_cosine"},
        "description": "Topology-aware Hierarchical Ensemble with Byzantine Defense (Proposed)",
    },
    {
        "id": "topo_no_aclm",
        "label": "HEP (w/o ACLM)",
        "topo": "hierarchical_ensemble",
        "pers": "none",
        "params": {
            "num_clusters": 3,
            "cluster_method": "update_similarity",
            "active_class_loss_mask": False,
            "active_class_inference_mask": False,
        },
        "description": "Ablation: HEP-FL without Active-Class Logit Masking",
    },
    {
        "id": "topo_no_parent",
        "label": "HEP (w/o Parent Head)",
        "topo": "hierarchical_ensemble",
        "pers": "none",
        "params": {
            "num_clusters": 3,
            "cluster_method": "update_similarity",
            "enable_parent_head": False,
        },
        "description": "Ablation: 2-tier Bipartite HEP-FL (Root + Local, without Parent Head)",
    },
    {
        "id": "topo_k1",
        "label": "HEP (K=1 Grand Coalition)",
        "topo": "hierarchical_ensemble",
        "pers": "none",
        "params": {
            "num_clusters": 1,
            "cluster_method": "update_similarity",
        },
        "description": "Cluster Valuation: Single grand coalition without peer sub-coalitions (K=1)",
    },
    {
        "id": "topo_oracle_k3",
        "label": "HEP (K=3 Oracle Bound)",
        "topo": "hierarchical_ensemble",
        "pers": "none",
        "params": {
            "num_clusters": 3,
            "cluster_method": "label_aware",
        },
        "description": "Cluster Valuation: Theoretical Pareto upper bound via ground-truth label distribution clustering",
    },
]

# Personalization benchmark subset (excludes defense-only variant)
PERSONALIZATION_METHODS = ["fedavg", "fedprox", "multikrum", "scaffold", "ditto", "fedrep", "topo"]

# Byzantine benchmark methods
BYZANTINE_METHODS = ["fedavg", "fedprox", "multikrum", "scaffold", "ditto", "fedrep", "topo", "topo_defended"]

# Targeted Ablation & Cluster Valuation subset
ABLATION_METHODS = ["topo", "topo_no_aclm", "topo_no_parent", "topo_k1", "topo_oracle_k3"]

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
    """Retrieve metadata dict for a method ID with aliases."""
    alias_map = {
        "fedhep": "topo",
        "fed_hep": "topo",
        "fed-hep": "topo",
        "fedhep_defended": "topo_defended",
        "fedhep-defended": "topo_defended",
        "fed_hep_defended": "topo_defended",
        "hep": "topo",
        "hep_fl": "topo",
        "hep-fl": "topo",
        "hep_defended": "topo_defended",
        "hep-fl_defended": "topo_defended",
        "defended_hep": "topo_defended",
        "defended_hep_fl": "topo_defended",
        "no_aclm": "topo_no_aclm",
        "no_parent": "topo_no_parent",
        "k1": "topo_k1",
        "oracle_k3": "topo_oracle_k3",
    }
    m_clean = str(method_id).strip().lower()
    canonical = alias_map.get(m_clean, m_clean)
    for m in METHODS:
        if m["id"] == canonical:
            return m
    valid = [m["id"] for m in METHODS] + list(alias_map.keys())
    raise ValueError(f"Unknown method ID: '{method_id}'. Valid methods/aliases: {valid}")


def get_regime_meta(regime_id: str) -> Dict[str, Any]:
    """Retrieve metadata dict for a heterogeneity regime ID with aliases."""
    alias_map = {
        "0.05": "extreme",
        "dirichlet_0.05": "extreme",
        "alpha_0.05": "extreme",
        "0.1": "severe",
        "dirichlet_0.1": "severe",
        "alpha_0.1": "severe",
        "0.5": "moderate",
        "dirichlet_0.5": "moderate",
        "alpha_0.5": "moderate",
        "1.0": "mild",
        "1": "mild",
        "dirichlet_1.0": "mild",
        "alpha_1.0": "mild",
        "inf": "iid",
        "dirichlet_inf": "iid",
    }
    r_clean = str(regime_id).strip().lower()
    canonical = alias_map.get(r_clean, r_clean)
    for r in REGIMES:
        if r["id"] == canonical:
            return r
    valid = [r["id"] for r in REGIMES] + list(alias_map.keys())
    raise ValueError(f"Unknown regime ID: '{regime_id}'. Valid regimes/aliases: {valid}")


def get_attack_meta(attack_id: str) -> Dict[str, Any]:
    """Retrieve metadata dict for a byzantine attack ID with aliases."""
    alias_map = {
        "noise": "random_noise",
        "gaussian": "random_noise",
        "gaussian_noise": "random_noise",
        "grad_ascent": "gradient_ascent",
        "ascent": "gradient_ascent",
    }
    a_clean = str(attack_id).strip().lower()
    canonical = alias_map.get(a_clean, a_clean)
    for a in ATTACK_TYPES:
        if a["id"] == canonical:
            return a
    valid = [a["id"] for a in ATTACK_TYPES] + list(alias_map.keys())
    raise ValueError(f"Unknown attack ID: '{attack_id}'. Valid attacks/aliases: {valid}")



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
    if method_id.startswith("topo") or method_id.startswith("hep"):
        client_kwargs["use_ensemble"] = True
        client_kwargs["hierarchical_ensemble"] = True
        client_kwargs["compute_optimization_mode"] = "shared_backbone"
    else:
        client_kwargs["compute_optimization_mode"] = "none"

    if "params" in method:
        for k, v in method["params"].items():
            if k in (
                "fedprox_mu", "ditto_lambda", "krum_num_selected", "krum_num_byzantine",
                "fedrep_head_epochs", "active_class_loss_mask", "active_class_inference_mask",
                "enable_parent_head",
            ):
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
            data_dir=defaults.get("data_dir", "./data"),
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
    if method_id.startswith("topo") or method_id.startswith("hep"):
        client_kwargs["use_ensemble"] = True
        client_kwargs["hierarchical_ensemble"] = True
        client_kwargs["compute_optimization_mode"] = "shared_backbone"
    else:
        client_kwargs["compute_optimization_mode"] = "none"

    if "params" in method:
        for k, v in method["params"].items():
            if k in (
                "fedprox_mu", "ditto_lambda", "krum_num_selected", "krum_num_byzantine",
                "fedrep_head_epochs", "active_class_loss_mask", "active_class_inference_mask",
                "enable_parent_head",
            ):
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
            data_dir=defaults.get("data_dir", "./data"),
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
