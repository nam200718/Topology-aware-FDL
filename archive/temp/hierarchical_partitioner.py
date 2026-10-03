import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from collections import defaultdict

def partition_data_hierarchical(
    dataset,
    num_clients: int,
    num_clusters: int,
    intra_alpha: float = 5.0,
    inter_alpha: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    2-Phase Hierarchical Dirichlet Partitioning.
    
    Args:
        dataset: torchvision dataset
        num_clients: Total number of clients
        num_clusters: Total number of clusters
        intra_alpha: Dirichlet concentration parameter for intra-cluster (client level)
        inter_alpha: Dirichlet concentration parameter for inter-cluster (cluster level)
        seed: Random seed
        
    Returns:
        client_indices: Mapping from client_id to list of data indices
        client_to_cluster: Mapping from client_id to cluster_id
    """
    np.random.seed(seed)
    
    import torch
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu()
    labels = np.array(targets)
    num_classes = len(np.unique(labels))
    num_clients_per_cluster = num_clients // num_clusters
    
    # Map class to indices
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}
    
    # --- PHASE 1: Cluster-level allocation ---
    cluster_indices = {k: [] for k in range(num_clusters)}
    
    for c in range(num_classes):
        idx_c = class_indices[c]
        np.random.shuffle(idx_c)
        
        # Dirichlet distribution over K clusters
        proportions = np.random.dirichlet(np.repeat(inter_alpha, num_clusters))
        proportions = proportions / proportions.sum()
        
        # Allocate based on proportions
        counts = (proportions * len(idx_c)).astype(int)
        
        # Ensure all samples are assigned
        leftover = len(idx_c) - counts.sum()
        for i in range(leftover):
            counts[np.random.randint(num_clusters)] += 1
            
        start = 0
        for k in range(num_clusters):
            end = start + counts[k]
            cluster_indices[k].extend(idx_c[start:end])
            start = end
            
    # --- PHASE 2: Client-level allocation within clusters ---
    client_indices = {i: [] for i in range(num_clients)}
    client_to_cluster = {}
    
    client_id_counter = 0
    for k in range(num_clusters):
        # Clients in this cluster
        cluster_clients = list(range(client_id_counter, client_id_counter + num_clients_per_cluster))
        client_id_counter += num_clients_per_cluster
        
        for cid in cluster_clients:
            client_to_cluster[cid] = k
            
        # Group cluster indices by class
        cluster_labels = labels[cluster_indices[k]]
        cluster_class_indices = {c: [] for c in range(num_classes)}
        
        for idx in cluster_indices[k]:
            cluster_class_indices[labels[idx]].append(idx)
            
        for c in range(num_classes):
            idx_c = cluster_class_indices[c]
            if not idx_c:
                continue
                
            np.random.shuffle(idx_c)
            
            # Dirichlet distribution over M/K clients
            proportions = np.random.dirichlet(np.repeat(intra_alpha, num_clients_per_cluster))
            proportions = proportions / proportions.sum()
            
            counts = (proportions * len(idx_c)).astype(int)
            leftover = len(idx_c) - counts.sum()
            for i in range(leftover):
                counts[np.random.randint(num_clients_per_cluster)] += 1
                
            start = 0
            for i, cid in enumerate(cluster_clients):
                end = start + counts[i]
                client_indices[cid].extend(idx_c[start:end])
                start = end

    return client_indices, client_to_cluster

def compute_cluster_label_overlap(
    client_indices: Dict[int, List[int]],
    client_to_cluster: Dict[int, int],
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute intra and inter cluster overlap based on label distributions.
    """
    num_clients = len(client_indices)
    num_classes = len(np.unique(labels))
    
    client_dists = np.zeros((num_clients, num_classes))
    for cid, indices in client_indices.items():
        if len(indices) == 0:
            continue
        client_labels = labels[indices]
        counts = np.bincount(client_labels, minlength=num_classes)
        client_dists[cid] = counts / len(indices)
        
    client_dists_tensor = torch.tensor(client_dists, dtype=torch.float32)
    
    intra_sims = []
    inter_sims = []
    
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            sim = F.cosine_similarity(client_dists_tensor[i].unsqueeze(0), client_dists_tensor[j].unsqueeze(0)).item()
            if client_to_cluster[i] == client_to_cluster[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)
                
    return {
        "intra_cluster_overlap": np.mean(intra_sims) if intra_sims else 0.0,
        "inter_cluster_overlap": np.mean(inter_sims) if inter_sims else 0.0
    }
