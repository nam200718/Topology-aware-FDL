import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets

from temp.hierarchical_partitioner import partition_data_hierarchical, compute_cluster_label_overlap

def plot_heatmap(client_indices, client_to_cluster, labels, num_classes, title, filename):
    num_clients = len(client_indices)
    matrix = np.zeros((num_clients, num_classes))
    
    # Sort clients by cluster
    sorted_clients = sorted(client_to_cluster.keys(), key=lambda c: client_to_cluster[c])
    
    for i, cid in enumerate(sorted_clients):
        idx = client_indices[cid]
        if len(idx) > 0:
            c_labels = labels[idx]
            counts = np.bincount(c_labels, minlength=num_classes)
            matrix[i] = counts / len(idx)
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="YlGnBu")
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Clients (sorted by cluster)")
    
    # Draw horizontal lines to separate clusters
    clusters = [client_to_cluster[c] for c in sorted_clients]
    for i in range(1, len(clusters)):
        if clusters[i] != clusters[i-1]:
            plt.axhline(i, color='red', lw=2)
            
    plt.savefig(filename)
    plt.close()

def run_sweep():
    # Load dummy dataset info (we only need labels)
    # Using CIFAR-10 training set labels
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
    num_clusters = 3
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
