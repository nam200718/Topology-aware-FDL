from typing import List, Dict
import networkx as nx
from src.core.interfaces import Topology

class StarTopology(Topology):
    """
    [CENTRALIZED] Star Topology
    
    Architecture:
    - A single central server (hub) connected to all clients (spokes).
    - All communication passes through the central server.
    - Represents traditional Federated Learning (e.g., FedAvg).
    """
    def __init__(self):
        self.graph = None
        self.server_id = -1
        self.num_clients = 0

    def build(self, num_clients: int, seed: int) -> None:
        self.num_clients = num_clients
        # A star graph with 1 central node (server) and num_clients leaf nodes.
        # nx.star_graph(n) creates a star graph with n+1 nodes: one center and n outlying nodes.
        # Center is node 0.
        self.graph = nx.star_graph(num_clients)
        
    def get_neighbors(self, node_id: int) -> List[int]:
        if self.graph is None:
            raise ValueError("Topology not built yet")
            
        # In our logical mapping, 0 is server, 1..N are clients.
        # But we need to use Server_id = -1 externally to distinguish from client 0.
        # Let's handle logical mapping here.
        if node_id == self.server_id:
            # server's neighbors are all clients
            return list(range(self.num_clients))
        
        # client's neighbor is only the server
        return [self.server_id]

    def get_server_connected_clients(self) -> List[int]:
        if self.graph is None:
            raise ValueError("Topology not built yet")
        return list(range(self.num_clients))
