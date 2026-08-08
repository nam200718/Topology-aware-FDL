import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
import ssl

# Fix SSL Certificate Verify Failed error when downloading datasets from PyTorch
ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore

class ClientDataset(Dataset):
    """A subset of a global dataset specific to a single client."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        # Performance Optimization: If underlying dataset is preloaded on device,
        # extract contiguous GPU tensors directly for zero-overhead bulk slicing.
        if isinstance(dataset, FastDataset) or (hasattr(dataset, 'images') and hasattr(dataset, 'labels') and isinstance(dataset.images, torch.Tensor)):
            device = dataset.device if hasattr(dataset, 'device') else dataset.images.device
            if isinstance(indices, list):
                indices_t = torch.tensor(indices, dtype=torch.long, device=device)
            elif isinstance(indices, torch.Tensor):
                indices_t = indices.to(device)
            else:
                indices_t = torch.as_tensor(indices, dtype=torch.long, device=device)
            
            if len(indices_t) > 0:
                self.images = dataset.images[indices_t]
                self.labels = dataset.labels[indices_t]
            else:
                self.images = dataset.images[:0]
                self.labels = dataset.labels[:0]
            self.device = device

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if hasattr(self, 'images') and hasattr(self, 'labels'):
            return self.images[index], self.labels[index]
        return self.dataset[self.indices[index]]

    @property
    def targets(self):
        """Attempts to provide targets by indexing into the underlying dataset."""
        if hasattr(self, 'labels'):
            return self.labels
        labels = _get_labels(self.dataset)
        if labels is not None:
            return labels[self.indices]
        return None

def _get_labels(dataset):
    """
    Helper to extract labels from a dataset or its wrappers (Subset, ClientDataset, FastDataset).
    Returns a numpy array of labels, or None if not found.
    """
    targets = getattr(dataset, 'targets', None)
    if targets is not None:
        if isinstance(targets, torch.Tensor):
            return targets.cpu().numpy()
        return np.asarray(targets)
        
    labels = getattr(dataset, 'labels', None)
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            return labels.cpu().numpy()
        return np.asarray(labels)
    
    # Handle common wrappers (Subset, ClientDataset)
    base_ds = getattr(dataset, 'dataset', None)
    indices = getattr(dataset, 'indices', None)
    if base_ds is not None and indices is not None:
        base_labels = _get_labels(base_ds)
        if base_labels is not None:
            return np.asarray(base_labels)[indices]
                
    return None

class FastDataset(Dataset):
    """
    A dataset that preloads all data into memory and onto the target device.
    This speeds up training by avoiding host-to-device transfers during the training loop.
    """
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device
        
        # Load data into memory in batches for optimal memory throughput
        from torch.utils.data import DataLoader
        batch_size = min(len(dataset), 512) if len(dataset) > 0 else 1
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        imgs, lbls = [], []
        for images, labels in loader:
            imgs.append(images.to(device))
            lbls.append(labels.to(device))
        
        if imgs:
            self.images = torch.cat(imgs, dim=0)
            self.labels = torch.cat(lbls, dim=0)
        else:
            self.images = torch.tensor([]).to(device)
            self.labels = torch.tensor([]).to(device)
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    @property
    def targets(self):
        return self.labels

class FastTensorDataLoader:
    """
    Lightweight, zero-overhead iterator for PyTorch tensors preloaded on GPU.
    Avoids Python PyTorch DataLoader queueing and individual sample fetching.
    """
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int = 32, shuffle: bool = True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(labels)

    def __iter__(self):
        if self.num_samples == 0:
            return
        if self.shuffle:
            indices = torch.randperm(self.num_samples)
            for i in range(0, self.num_samples, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                yield self.images[batch_idx], self.labels[batch_idx]
        else:
            for i in range(0, self.num_samples, self.batch_size):
                yield self.images[i:i + self.batch_size], self.labels[i:i + self.batch_size]

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size if self.num_samples > 0 else 0

def get_fast_dataloader(dataset, batch_size: int = 32, shuffle: bool = True):
    """
    Returns FastTensorDataLoader if dataset has GPU tensors, else standard DataLoader.
    """
    if hasattr(dataset, 'images') and hasattr(dataset, 'labels') and isinstance(dataset.images, torch.Tensor) and isinstance(dataset.labels, torch.Tensor):
        return FastTensorDataLoader(dataset.images, dataset.labels, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_mnist(data_dir="./data", train_subset=None, test_subset=None, seed=42):
    """Downloads and returns the MNIST train and test sets, optionally subsetted."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    rng = np.random.RandomState(seed)
    if train_subset is not None and train_subset < len(train_dataset):
        indices = rng.choice(len(train_dataset), train_subset, replace=False)
        train_dataset = Subset(train_dataset, indices)
        
    if test_subset is not None and test_subset < len(test_dataset):
        indices = rng.choice(len(test_dataset), test_subset, replace=False)
        test_dataset = Subset(test_dataset, indices)

    return train_dataset, test_dataset

def get_cifar10(data_dir="./data", train_subset=None, test_subset=None, seed=42):
    """Downloads and returns the CIFAR-10 train and test sets, optionally subsetted."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    
    rng = np.random.RandomState(seed)
    if train_subset is not None and train_subset < len(train_dataset):
        indices = rng.choice(len(train_dataset), train_subset, replace=False)
        train_dataset = Subset(train_dataset, indices)
        
    if test_subset is not None and test_subset < len(test_dataset):
        indices = rng.choice(len(test_dataset), test_subset, replace=False)
        test_dataset = Subset(test_dataset, indices)

    return train_dataset, test_dataset


def partition_data(dataset, num_clients, non_iid=True, alpha=0.5, seed=42):
    """
    Partitions data across clients.
    If non_iid=True: Partition dataset according to a symmetric Dirichlet distribution with concentration parameter alpha.
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
        # Non-IID partitioning using Dirichlet distribution
        labels = _get_labels(dataset)
        
        if labels is None:
            # Fallback if labels not found via attributes
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
            
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        # Dictionary tracking indices of samples for each class
        class_indices = {c: np.where(labels == c)[0] for c in unique_labels}
        
        # Dirichlet parameter vector
        p = alpha * np.ones(num_clients)
        proportions = rng.dirichlet(p, num_classes)
        
        client_indices = {i: [] for i in range(num_clients)}
        
        for class_idx, label_val in enumerate(unique_labels):
            indices = class_indices[label_val].copy()
            rng.shuffle(indices)
            
            # Calculate split sizes
            split_counts = np.floor(proportions[class_idx] * len(indices)).astype(int)
            # Ensure all indices are allocated (handle floor truncation leftovers)
            leftover = len(indices) - sum(split_counts)
            for idx in range(leftover):
                split_counts[idx % num_clients] += 1
                
            # Split indices
            split_indices = np.cumsum(split_counts)[:-1]
            splits = np.split(indices, split_indices)
            
            for client_idx in range(num_clients):
                client_indices[client_idx].extend(splits[client_idx].tolist())
                
        return client_indices

def partition_data_non_iid(dataset, num_clients, alpha=0.5, seed=42):
    """Legacy wrapper for backward compatibility."""
    return partition_data(dataset, num_clients, non_iid=True, alpha=alpha, seed=seed)
