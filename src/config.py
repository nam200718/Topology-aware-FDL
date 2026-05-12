import os
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
    # Number of shards to divide the dataset into. 
    # More shards = more IID if distributed randomly.
    # Fewer shards = more non-IID.
    num_shards: int = 200

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
