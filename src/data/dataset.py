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

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    @property
    def targets(self):
        """Attempts to provide targets by indexing into the underlying dataset."""
        labels = _get_labels(self.dataset)
        if labels is not None:
            return labels[self.indices]
        return None

def _get_labels(dataset):
    """
    Helper to extract labels from a dataset or its wrappers (Subset, ClientDataset, FastDataset).
    Returns a numpy array or torch tensor of labels, or None if not found.
    """
    # Use getattr to be safe
    targets = getattr(dataset, 'targets', None)
    if targets is not None:
        return targets
        
    labels = getattr(dataset, 'labels', None)
    if labels is not None:
        return labels
    
    # Handle common wrappers (Subset, ClientDataset)
    base_ds = getattr(dataset, 'dataset', None)
    indices = getattr(dataset, 'indices', None)
    if base_ds is not None and indices is not None:
        base_labels = _get_labels(base_ds)
        if base_labels is not None:
            if isinstance(base_labels, torch.Tensor):
                return base_labels[indices]
            else:
                return np.array([base_labels[i] for i in indices])
                
    return None

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

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    @property
    def targets(self):
        return self.labels

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
        
    if test_subset is not None and test_subset < len(test_dataset):
        indices = np.random.choice(len(test_dataset), test_subset, replace=False)
        test_dataset = Subset(test_dataset, indices)

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
        labels = _get_labels(dataset)
        
        if labels is None:
            # Fallback if labels not found via attributes
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
            
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            
        sorted_indices = np.argsort(labels)
        
        shard_size = num_samples // num_shards
        shard_indices = [
            sorted_indices[i * shard_size: (i + 1) * shard_size] 
            for i in range(num_shards)
        ]
        
        # Shuffle shards to randomly distribute them to clients
        rng.shuffle(shard_indices)

        # Distribute shards to clients as evenly as possible
        client_indices = {i: [] for i in range(num_clients)}
        for shard_idx, indices in enumerate(shard_indices):
            client_id = shard_idx % num_clients
            client_indices[client_id].extend(indices.tolist())
            
        return client_indices

def partition_data_non_iid(dataset, num_clients, num_shards=200, seed=42):
    """Legacy wrapper for backward compatibility."""
    return partition_data(dataset, num_clients, non_iid=True, num_shards=num_shards, seed=seed)
