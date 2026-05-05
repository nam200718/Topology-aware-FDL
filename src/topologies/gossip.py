from typing import List
import networkx as nx
from src.core.interfaces import Topology
import random

class GossipTopology(Topology):
    """
    [DECENTRALIZED] Gossip P2P Topology
    
    Architecture:
    - Peer-to-peer network where each client connects to 'k' random neighbors.
    - Uses a Random Regular Graph to ensure uniform connectivity.
    - Highly robust and scalable for fully decentralized learning.
    """
    def __init__(self, degree_k: int = 3):
        self.graph = None
        self.num_clients = 0
        self.degree_k = degree_k

    def build(self, num_clients: int, seed: int) -> None:
        self.num_clients = num_clients
        # Generate a random regular graph for gossip P2P, ensuring connectivity
        # If degree_k * num_clients is odd, random_regular_graph will fail. Sub with ER graph if needed.
        # Ensure k is valid
        k = min(self.degree_k, num_clients - 1)
        if (num_clients * k) % 2 != 0:
            k -= 1 # adjust to make it even
            
        # We can seed networkx directly
        self.graph = nx.random_regular_graph(d=k, n=num_clients, seed=seed)
        
        # Verify it's connected, if not, keep trying with different seeds just for safety
        attempts = 0
        while not nx.is_connected(self.graph) and attempts < 10:
            seed += 1
            self.graph = nx.random_regular_graph(d=k, n=num_clients, seed=seed)
            attempts += 1
            
    def get_neighbors(self, node_id: int) -> List[int]:
        if self.graph is None:
            raise ValueError("Topology not built yet")
        return list(self.graph.neighbors(node_id))

    def get_server_connected_clients(self) -> List[int]:
        return []
