from typing import List
import numpy as np
from src.core.interfaces import Topology

class HierarchicalTopology(Topology):
    def __init__(self, num_clusters: int = 5):
        self.num_clients = 0
        self.num_clusters = num_clusters
        self.client_to_head = {}
        
    def build(self, num_clients: int, seed: int) -> None:
        self.num_clients = num_clients
        # Deterministically assign clients to cluster heads
        rng = np.random.RandomState(seed)
        
        # cluster ids: -2, -3, ..., -(num_clusters+1)
        for i in range(num_clients):
            cluster_idx = rng.randint(0, self.num_clusters)
            head_id = -2 - cluster_idx
            self.client_to_head[i] = head_id
            
    def get_neighbors(self, node_id: int) -> List[int]:
        if not self.client_to_head:
            raise ValueError("Topology not built yet")
        if node_id >= 0:
            # Client's neighbor is its assigned cluster head
            return [self.client_to_head[node_id]]
        else:
            # Note: in our engine, cluster heads don't explicitly call `get_neighbors`
            # as the aggregation relies on client->head message passing logic directly
            return []

    def get_server_connected_clients(self) -> List[int]:
        # Server is connected only to cluster heads
        return [-2 - i for i in range(self.num_clusters)]
