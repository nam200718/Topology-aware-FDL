import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets
import sys

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temp.hierarchical_partitioner import partition_data_hierarchical, compute_cluster_label_overlap

def plot_heatmap(client_indices, client_to_cluster, labels, num_classes, title, filename):
    # Sort clients by cluster
    sorted_clients = sorted(list(client_indices.keys()), key=lambda c: client_to_cluster.get(c, 0))
    
    # Create distribution matrix
    matrix = np.zeros((len(sorted_clients), num_classes))
    for i, c in enumerate(sorted_clients):
        client_labels = labels[client_indices[c]]
        unique, counts = np.unique(client_labels, return_counts=True)
        for val, count in zip(unique, counts):
            matrix[i, val] = count
            
    # Normalize by row to get proportions
    row_sums = matrix.sum(axis=1)
    matrix = np.divide(matrix, row_sums[:, np.newaxis], out=np.zeros_like(matrix), where=row_sums[:, np.newaxis]!=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="YlGnBu")
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Clients (sorted by cluster)")
    
    # Draw horizontal lines to separate clusters
    clusters = [client_to_cluster.get(c, 0) for c in sorted_clients]
    for i in range(1, len(clusters)):
        if clusters[i] != clusters[i-1]:
            plt.axhline(i, color='red', lw=2)
            
    plt.savefig(filename)
    plt.close()

def run_sweep():
    print("Loading CIFAR-10 for label distribution...")
    try:
        ds = datasets.CIFAR10(root='./data', train=True, download=True)
    except:
        print("Could not load CIFAR-10. Using dummy labels.")
        class DummyDS:
            def __init__(self):
                self.targets = np.random.randint(0, 10, 50000).tolist()
        ds = DummyDS()
        
    labels = np.array(ds.targets)
    num_classes = 10
    num_clients = 30
    num_clusters = 5
    beta = 0.1
    
    x_values = [0.1, 1.0, 5.0, 50.0]
    
    print(f"\nRunning sweep with beta={beta}, clusters={num_clusters}, clients={num_clients}")
    print(f"{'x':<10} | {'Intra-overlap':<15} | {'Inter-overlap':<15}")
    print("-" * 45)
    
    for x in x_values:
        client_indices, client_to_cluster = partition_data_hierarchical(
            dataset=ds,
            num_clients=num_clients,
            num_clusters=num_clusters,
            intra_alpha=x,
            inter_alpha=beta,
            seed=42
        )
        
        metrics = compute_cluster_label_overlap(client_indices, client_to_cluster, labels)
        
        print(f"{x:<10.2f} | {metrics['intra_cluster_overlap']:<15.4f} | {metrics['inter_cluster_overlap']:<15.4f}")
        
        plot_heatmap(
            client_indices, 
            client_to_cluster, 
            labels, 
            num_classes, 
            f"Label Dist (x={x}, beta={beta})",
            f"temp_heatmap_x_{x}.png"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run sweep over x values")
    args = parser.parse_args()
    
    if args.sweep:
        run_sweep()
    else:
        print("Please run with --sweep")
