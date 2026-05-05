from typing import List
import networkx as nx
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
        # A cycle graph where each node i is connected to i-1 and i+1
        self.graph = nx.cycle_graph(num_clients)
        
    def get_neighbors(self, node_id: int) -> List[int]:
        if self.graph is None:
            raise ValueError("Topology not built yet")
        # In a directed ring this might just be 1, but for undirected it's 2
        return list(self.graph.neighbors(node_id))

    def get_server_connected_clients(self) -> List[int]:
        # Not applicable for pure decentralized Ring
        return []
