from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch

class ClientState:
    """Represents the mutable state of a client in the simulation."""
    def __init__(self, client_id: int, initial_weights: torch.Tensor):
        self.client_id = client_id
        self.weights: torch.Tensor = initial_weights
        self.local_weights: Optional[torch.Tensor] = None  # Persistent local models for personalized ensemble
        self.parent_weights: Optional[torch.Tensor] = None # Persistent parent models for hierarchical ensemble
        
        self.parent_head_state: Optional[Dict[str, torch.Tensor]] = None
        self.local_head_state: Optional[Dict[str, torch.Tensor]] = None
        
        # Meta information that might be useful for analysis
        self.data_samples: int = 100 
        self.is_byzantine: bool = False
        self.byzantine_type: str = "label_flip"
        
        # APFL and Ensemble metrics tracking
        self.apfl_alpha: float = 0.5
        self.head_losses: Dict[str, float] = {}
        self.head_steps: Dict[str, int] = {}
        
    def copy(self):
        new_state = ClientState(self.client_id, self.weights.clone())
        if self.local_weights is not None:
            new_state.local_weights = self.local_weights.clone()
        if self.parent_weights is not None:
            new_state.parent_weights = self.parent_weights.clone()
        if self.parent_head_state is not None:
            new_state.parent_head_state = {k: v.clone() for k, v in self.parent_head_state.items()}
        if self.local_head_state is not None:
            new_state.local_head_state = {k: v.clone() for k, v in self.local_head_state.items()}
        new_state.data_samples = self.data_samples
        new_state.is_byzantine = self.is_byzantine
        new_state.byzantine_type = self.byzantine_type
        new_state.apfl_alpha = self.apfl_alpha
        new_state.head_losses = dict(self.head_losses)
        new_state.head_steps = dict(self.head_steps)
        return new_state

class Topology(ABC):
    """Defines the communication graph between clients and/or servers."""
    layers: List[int] = []
    gossip_steps: int = 0
    root_id: int = -1

    @abstractmethod
    def build(self, num_clients: int, seed: int) -> None:
        pass
    
    @abstractmethod
    def get_neighbors(self, node_id: int) -> List[int]:
        """Returns neighbor client IDs for a given node. Might return Server ID (-1) for Star."""
        pass
        
    @abstractmethod
    def get_server_connected_clients(self) -> List[int]:
        """Which clients directly talk to the central server (if applicable)."""
        pass

    # Optional methods that might be implemented by specific topologies
    def get_children(self, node_id: int) -> List[int]:
        return []

    def get_layer_peers(self, node_id: int) -> List[int]:
        return []

    def get_nodes_in_layer(self, layer_idx: int) -> List[int]:
        return []

    def get_depth(self) -> int:
        return 0

class Aggregator(ABC):
    """Abstract policy for aggregating multiple client states."""
    @abstractmethod
    def aggregate(self, states: List[ClientState]) -> torch.Tensor:
        pass


class MetricsCollector:
    """Collects outputs to be saved as JSON/CSV."""
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        
    def log_round(self, round_data: Dict[str, Any]):
        self.history.append(round_data)
        
    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
