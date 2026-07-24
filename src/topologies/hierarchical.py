from typing import List
import numpy as np
from src.core.interfaces import Topology

class HierarchicalTopology(Topology):
    """
    [HIERARCHICAL] 2-Tier Cluster Topology
    
    Architecture:
    - Clients are grouped into clusters.
    - Each cluster has a "Cluster Head" (intermediate aggregator).
    - Cluster Heads communicate with the global server.
    - Models the Edge-Cloud paradigm.
    """
    def __init__(self, num_clusters: int = 5):
        self.num_clients = 0
        self.num_clusters = num_clusters
        self.client_to_head = {}
        
    def build(self, num_clients: int, seed: int, client_label_counts: dict = None) -> None:
        self.num_clients = num_clients
        rng = np.random.RandomState(seed)
        
        if client_label_counts is None or len(client_label_counts) == 0:
            # Deterministically assign clients to cluster heads uniformly
            for i in range(num_clients):
                cluster_idx = rng.randint(0, self.num_clusters)
                head_id = -2 - cluster_idx
                self.client_to_head[i] = head_id
        else:
            self.build_distribution_aware(num_clients, client_label_counts, seed)

    def build_distribution_aware(self, num_clients: int, client_label_counts: dict, seed: int) -> None:
        """
        Groups clients into clusters based on label distribution similarity (Cosine similarity).
        Clients with similar Non-IID label skew share the same cluster head.
        """
        self.num_clients = num_clients
        rng = np.random.RandomState(seed)
        
        # Convert label counts to normalized vectors
        all_labels = set()
        for counts in client_label_counts.values():
            all_labels.update(counts.keys())
        num_classes = max(all_labels) + 1 if all_labels else 10
        
        client_vectors = np.zeros((num_clients, num_classes))
        for cid, counts in client_label_counts.items():
            for lbl, count in counts.items():
                if lbl < num_classes:
                    client_vectors[cid, lbl] = count
            norm = np.linalg.norm(client_vectors[cid])
            if norm > 0:
                client_vectors[cid] /= norm

        # Simple K-Means clustering over client label distribution vectors
        num_clusters = min(self.num_clusters, num_clients)
        # Random initial centroids
        init_indices = rng.choice(num_clients, num_clusters, replace=False)
        centroids = client_vectors[init_indices].copy()

        assignments = np.zeros(num_clients, dtype=int)
        for _ in range(10): # 10 iterations of K-Means
            # Compute cosine similarities / distances
            dists = np.linalg.norm(client_vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)
            # Update centroids
            for k in range(num_clusters):
                cluster_members = client_vectors[assignments == k]
                if len(cluster_members) > 0:
                    centroids[k] = cluster_members.mean(axis=0)

        for i in range(num_clients):
            cluster_idx = assignments[i]
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
