import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset

class ClientDataset(Dataset):
    """A subset of a global dataset specific to a single client."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def get_mnist(data_dir="./data"):
    """Downloads and returns the MNIST train and test sets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def partition_data_non_iid(dataset, num_clients, num_shards=200, seed=42):
    """
    Partitions data into a non-IID distribution across clients.
    Approach: Sort dataset by label, divide into `num_shards`, and randomly 
    assign `num_shards / num_clients` to each client.
    """
    rng = np.random.RandomState(seed)
    
    # Extract labels safely without loading entire dataset into memory if possible
    # For MNIST, dataset.targets exists
    labels = dataset.targets.numpy()
    
    # Sort indices by label to group similar digits together
    sorted_indices = np.argsort(labels)
    
    shards_per_client = num_shards // num_clients
    shard_size = len(sorted_indices) // num_shards
    
    shard_indices = [
        sorted_indices[i * shard_size: (i + 1) * shard_size] 
        for i in range(num_shards)
    ]
    
    # Shuffle shards to randomly distribute them to clients
    rng.shuffle(shard_indices)
    
    client_indices = {}
    for i in range(num_clients):
        client_shards = shard_indices[i * shards_per_client : (i + 1) * shards_per_client]
        client_indices[i] = np.concatenate(client_shards).tolist()
        
    return client_indices
