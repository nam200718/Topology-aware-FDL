from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

class ClientState:
    """Represents the mutable state of a client in the simulation."""
    def __init__(self, client_id: int, initial_weights: np.ndarray):
        self.client_id = client_id
        self.weights = initial_weights
        self.local_weights = None  # Persistent local models for personalized ensemble
        
        # Meta information that might be useful for analysis
        self.data_samples: int = 100 
        self.is_byzantine: bool = False
        self.byzantine_type: str = "label_flip"
        
    def copy(self):
        new_state = ClientState(self.client_id, self.weights.clone())
        if self.local_weights is not None:
            new_state.local_weights = self.local_weights.clone()
        new_state.data_samples = self.data_samples
        new_state.is_byzantine = self.is_byzantine
        new_state.byzantine_type = self.byzantine_type
        return new_state

class Topology(ABC):
    """Defines the communication graph between clients and/or servers."""
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

class Aggregator(ABC):
    """Abstract policy for aggregating multiple client states."""
    @abstractmethod
    def aggregate(self, states: List[ClientState]) -> np.ndarray:
        pass


class MetricsCollector:
    """Collects outputs to be saved as JSON/CSV."""
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        
    def log_round(self, round_data: Dict[str, Any]):
        self.history.append(round_data)
        
    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
