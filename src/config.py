import os
import yaml
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Optional, List, Union

TopologyType = Literal["star", "star_randomized", "ring", "hierarchical", "hierarchical_ensemble", "gossip", "layered"]

class EnvironmentConfig(BaseModel):
    # Where to save outputs
    output_dir: str = "./outputs"
    # Seed for global reproducibility
    seed: int = 42
    # Optional subset sizes for faster experiments
    train_subset: Optional[int] = None
    test_subset: Optional[int] = None
    # Dataset to use ("mnist" or "cifar10")
    dataset: Literal["mnist", "cifar10"] = "mnist"


class TopologyConfig(BaseModel):
    type: TopologyType = "star"
    # Additional topology-specific parameters (e.g., degree for gossip, clusters for hierarchical)
    params: Dict[str, Any] = Field(default_factory=dict)

class ClientConfig(BaseModel):
    num_clients: int = 100
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

class RobustnessConfig(BaseModel):
    byzantine_rate: float = 0.0
    # Attack options: 'label_flip', 'sign_flip', 'random_noise', 'gradient_ascent'
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
    byzantine_rate: float = 0.0

class ExperimentConfig(BaseModel):
    """
    Top-level config loaded from a YAML file.
    Holds experiment-wide defaults and sweep axes.
    Can generate one or more SimulationConfig instances via build_configs().
    """
    experiment_type: ExperimentType = "single"
    num_rounds: int = 10

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

    def _make_client_config(self, topo_type: TopologyType, **overrides) -> ClientConfig:
        """
        Build a ClientConfig from defaults, auto-setting ensemble flags
        based on the topology type.
        """
        base = self.clients.model_dump()
        is_ensemble = (topo_type == "hierarchical_ensemble")
        base["use_ensemble"] = is_ensemble or base.get("use_ensemble", False)
        base["hierarchical_ensemble"] = is_ensemble or base.get("hierarchical_ensemble", False)
        base.update(overrides)
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
                env=env,
                topology=TopologyConfig(type=topo.type, params=topo.params),
                clients=self._make_client_config(topo.type),
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
                        env=env,
                        topology=TopologyConfig(type=topo.type, params=topo.params),
                        clients=self._make_client_config(topo.type),
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
                for scenario in self.scenarios:
                    exp_name = f"{topo.type}_{scenario.id}"
                    sim_config = SimulationConfig(
                        experiment_name=exp_name,
                        num_rounds=self.num_rounds,
                        env=env,
                        topology=TopologyConfig(type=topo.type, params=topo.params),
                        clients=self._make_client_config(topo.type),
                        robustness=RobustnessConfig(
                            byzantine_rate=scenario.byzantine_rate,
                            byzantine_type=self.robustness.byzantine_type,
                        ),
                        non_iid=NonIIDConfig(
                            enabled=scenario.non_iid,
                            alpha=self.non_iid.alpha,
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

