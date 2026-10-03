import networkx as nx
from typing import List
from src.core.interfaces import Topology

class RingTopology(Topology):
    """
    [DECENTRALIZED] Ring Topology
    
    Architecture:
    - Clients are arranged in a closed loop (cycle).
    - Each client connects only to its immediate left and right neighbors.
    - Knowledge propagates through the network step-by-step.
    """
    def __init__(self):
        self.graph = None
        self.num_clients = 0

    def build(self, num_clients: int, seed: int) -> None:
        self.num_clients = num_clients
        self.graph = nx.cycle_graph(num_clients)
        
    def get_neighbors(self, node_id: int) -> List[int]:
        if self.graph is None:
            raise ValueError("Topology not built yet")
        return list(self.graph.neighbors(node_id))

    def get_server_connected_clients(self) -> List[int]:
        return []
