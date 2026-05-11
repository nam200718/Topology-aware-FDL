import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, Subset

class ClientDataset(Dataset):
    """A subset of a global dataset specific to a single client."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

class FastDataset(Dataset):
    """
    A dataset that preloads all data into memory and onto the target device.
    This speeds up training by avoiding host-to-device transfers during the training loop.
    """
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device
        
        # Load all data into memory on the device
        # We assume the dataset is small enough to fit in memory (e.g., MNIST)
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for images, labels in loader:
            self.images = images.to(device)
            self.labels = labels.to(device)
            break
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_mnist(data_dir="./data", train_subset=None, test_subset=None):
    """Downloads and returns the MNIST train and test sets, optionally subsetted."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    if train_subset is not None and train_subset < len(train_dataset):
        indices = np.random.choice(len(train_dataset), train_subset, replace=False)
        train_dataset = Subset(train_dataset, indices)
        # We need to preserve .targets for partition_data
        train_dataset.targets = torch.tensor([train_dataset.dataset.targets[i] for i in indices])
        
    if test_subset is not None and test_subset < len(test_dataset):
        indices = np.random.choice(len(test_dataset), test_subset, replace=False)
        test_dataset = Subset(test_dataset, indices)
        test_dataset.targets = torch.tensor([test_dataset.dataset.targets[i] for i in indices])

    return train_dataset, test_dataset

def partition_data(dataset, num_clients, non_iid=True, num_shards=200, seed=42):
    """
    Partitions data across clients.
    If non_iid=True: Sort dataset by label, divide into shards, and assign shards to clients.
    If non_iid=False: Randomly assign samples to clients (IID).
    """
    rng = np.random.RandomState(seed)
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    
    if not non_iid:
        # IID partitioning
        rng.shuffle(indices)
        client_indices = {}
        samples_per_client = num_samples // num_clients
        for i in range(num_clients):
            client_indices[i] = indices[i * samples_per_client : (i + 1) * samples_per_client].tolist()
        return client_indices
    else:
        # Non-IID partitioning
        if hasattr(dataset, 'targets'):
            labels = dataset.targets
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
        else:
            # Fallback if targets not available (e.g. nested subset)
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
            
        sorted_indices = np.argsort(labels)
        
        shards_per_client = num_shards // num_clients
        shard_size = num_samples // num_shards
        
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

def partition_data_non_iid(dataset, num_clients, num_shards=200, seed=42):
    """Legacy wrapper for backward compatibility."""
    return partition_data(dataset, num_clients, non_iid=True, num_shards=num_shards, seed=seed)
