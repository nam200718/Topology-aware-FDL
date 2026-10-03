import os
import shutil
import tarfile
import zipfile
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
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int = 32, shuffle: bool = True, drop_last: bool = False):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(labels)

    def __iter__(self):
        if self.num_samples == 0:
            return
        if self.shuffle:
            indices = torch.randperm(self.num_samples, device=self.images.device)
            for i in range(0, self.num_samples, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                if self.drop_last and len(batch_idx) < self.batch_size:
                    continue
                yield self.images[batch_idx], self.labels[batch_idx]
        else:
            for i in range(0, self.num_samples, self.batch_size):
                batch = self.images[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch, self.labels[i:i + self.batch_size]

    def __len__(self):
        if self.num_samples == 0:
            return 0
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

def get_fast_dataloader(dataset, batch_size: int = 32, shuffle: bool = True, drop_last: bool = False):
    """
    Returns FastTensorDataLoader if dataset has GPU tensors, else standard DataLoader.
    """
    if hasattr(dataset, 'images') and hasattr(dataset, 'labels') and isinstance(dataset.images, torch.Tensor) and isinstance(dataset.labels, torch.Tensor):
        return FastTensorDataLoader(dataset.images, dataset.labels, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def _acquire_download_lock(data_dir: str, name: str):
    import os, tempfile
    lock_path = os.path.join(tempfile.gettempdir(), f".{name}.lock")
    lock_file = open(lock_path, "w")
    try:
        import fcntl
        fcntl.flock(lock_file, fcntl.LOCK_EX)
    except Exception:
        pass
    return lock_file


def _release_download_lock(lock_file):
    try:
        import fcntl
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
    except Exception:
        try:
            lock_file.close()
        except Exception:
            pass


def get_mnist(data_dir="./data", train_subset=None, test_subset=None, seed=42):
    """Downloads and returns the MNIST train and test sets, optionally subsetted."""
    import os, shutil
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    lock = _acquire_download_lock(data_dir, "mnist")
    try:
        try:
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        except (EOFError, RuntimeError, OSError, ValueError):
            # Auto-heal: delete corrupted partial download and retry
            mnist_dir = os.path.join(data_dir, "MNIST")
            if os.path.exists(mnist_dir):
                shutil.rmtree(mnist_dir, ignore_errors=True)
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    finally:
        _release_download_lock(lock)
    
    rng = np.random.RandomState(seed)
    if train_subset is not None and train_subset < len(train_dataset):
        indices = rng.choice(len(train_dataset), train_subset, replace=False)
        train_dataset = Subset(train_dataset, indices)
        
    if test_subset is not None and test_subset < len(test_dataset):
        indices = rng.choice(len(test_dataset), test_subset, replace=False)
        test_dataset = Subset(test_dataset, indices)

    return train_dataset, test_dataset


def _is_valid_dataset_dir(path: str, dataset_name: str) -> bool:
    """Verifies that a directory directly contains the essential torchvision dataset files."""
    if not path or not os.path.isdir(path):
        return False
    try:
        files = set(os.listdir(path))
    except OSError:
        return False
    if dataset_name == "cifar100":
        return {"train", "test"}.issubset(files)
    elif dataset_name == "cifar10":
        return {"data_batch_1", "test_batch"}.issubset(files) or {"batches.meta", "data_batch_1"}.issubset(files)
    elif dataset_name in ("femnist", "emnist"):
        return any("byclass" in f.lower() or "emnist" in f.lower() for f in files) or "EMNIST" in files
    return False


def _auto_link_dataset(data_dir: str, dataset_name: str) -> bool:
    """Auto-detect pre-existing dataset in /kaggle/input or common cloud mount directories.
    Supports:
    1. Standard subdirectory (e.g. /kaggle/input/.../cifar-100-python)
    2. Double-nested folders (e.g. /kaggle/input/cifar-100-python/cifar-100-python)
    3. Flat root dataset (e.g. /kaggle/input/... directly containing train, test, meta)
    4. Archives (*.tar.gz, *.tgz, *.tar, *.zip)
    Links or extracts into data_dir to eliminate download times and network timeouts on Kaggle.
    """
    import os, shutil, tarfile, zipfile

    if dataset_name == "cifar100":
        folder_name = "cifar-100-python"
        archive_prefixes = ("cifar-100", "cifar100")
    elif dataset_name == "cifar10":
        folder_name = "cifar-10-batches-py"
        archive_prefixes = ("cifar-10", "cifar10")
    elif dataset_name in ("femnist", "emnist"):
        folder_name = "EMNIST"
        archive_prefixes = ("emnist", "femnist")
    else:
        return False

    target_dir = os.path.join(data_dir, folder_name)

    # Clean up broken symlink if target no longer exists
    if os.path.islink(target_dir) and not os.path.exists(target_dir):
        try:
            os.unlink(target_dir)
        except OSError:
            pass

    # If target directory already exists and contains the valid files, return immediately
    if os.path.exists(target_dir):
        if _is_valid_dataset_dir(target_dir, dataset_name):
            return True
        else:
            # Clean up invalid symlink/folder
            try:
                if os.path.islink(target_dir):
                    os.unlink(target_dir)
                else:
                    shutil.rmtree(target_dir, ignore_errors=True)
            except OSError:
                pass

    search_roots = []
    for env_var in ("KAGGLE_DATASET_PATH", "HEP_DATA_DIR", "DATA_DIR"):
        val = os.environ.get(env_var)
        if val and val not in search_roots:
            search_roots.append(val)
    for p in ["/workspace/data", "/kaggle/input", "/content", os.path.expanduser("~/.cache")]:
        if p not in search_roots:
            search_roots.append(p)

    for candidate_root in search_roots:
        if not os.path.exists(candidate_root):
            continue
        for root, dirs, files in os.walk(candidate_root, followlinks=True):
            # Limit depth to avoid traversing deep unrelated directories
            rel_depth = root[len(candidate_root):].count(os.sep)
            if rel_depth > 4:
                continue

            # Case 1: The current directory itself directly contains the dataset files
            if _is_valid_dataset_dir(root, dataset_name):
                src_path = root
                os.makedirs(data_dir, exist_ok=True)
                try:
                    os.symlink(src_path, target_dir)
                    print(f"⚡ [Kaggle Input] Symlinked {dataset_name} files from {src_path} -> {target_dir} (0.0s)")
                    return True
                except Exception:
                    try:
                        shutil.copytree(src_path, target_dir, dirs_exist_ok=True)
                        print(f"⚡ [Kaggle Input] Copied {dataset_name} files from {src_path} -> {target_dir}")
                        return True
                    except Exception:
                        pass

            # Case 2: A subfolder named folder_name directly contains the dataset files
            if folder_name in dirs:
                sub_path = os.path.join(root, folder_name)
                if _is_valid_dataset_dir(sub_path, dataset_name):
                    src_path = sub_path
                    os.makedirs(data_dir, exist_ok=True)
                    try:
                        os.symlink(src_path, target_dir)
                        print(f"⚡ [Kaggle Input] Symlinked {dataset_name} folder from {src_path} -> {target_dir} (0.0s)")
                        return True
                    except Exception:
                        try:
                            shutil.copytree(src_path, target_dir, dirs_exist_ok=True)
                            print(f"⚡ [Kaggle Input] Copied {dataset_name} folder from {src_path} -> {target_dir}")
                            return True
                        except Exception:
                            pass

            # Case 3: Archive file (*.tar.gz, *.tgz, *.zip)
            for fname in files:
                lower_f = fname.lower()
                if any(pref in lower_f for pref in archive_prefixes):
                    src_archive = os.path.join(root, fname)
                    os.makedirs(data_dir, exist_ok=True)
                    if lower_f.endswith((".tar.gz", ".tgz", ".tar")):
                        try:
                            with tarfile.open(src_archive, "r:*") as tar:
                                tar.extractall(data_dir)
                            print(f"⚡ [Kaggle Input] Extracted tar archive {src_archive} -> {data_dir}")
                            if _is_valid_dataset_dir(target_dir, dataset_name):
                                return True
                        except Exception:
                            pass
                    elif lower_f.endswith(".zip"):
                        try:
                            with zipfile.ZipFile(src_archive, "r") as zf:
                                zf.extractall(data_dir)
                            print(f"⚡ [Kaggle Input] Extracted zip archive {src_archive} -> {data_dir}")
                            if _is_valid_dataset_dir(target_dir, dataset_name):
                                return True
                        except Exception:
                            pass
    return False


def _resolve_safe_data_dir(data_dir: str) -> str:
    """Redirects read-only directories (e.g. /kaggle/input/...) to a safe writable working directory."""
    import os
    if not data_dir:
        return "./data"
    abs_dir = os.path.abspath(data_dir)
    is_kaggle_input = data_dir.startswith("/kaggle/input") or abs_dir.startswith("/kaggle/input")
    is_read_only = os.path.exists(abs_dir) and not os.access(abs_dir, os.W_OK)
    if is_kaggle_input or is_read_only:
        os.environ["KAGGLE_DATASET_PATH"] = abs_dir
        if os.path.exists("/kaggle/working"):
            safe_dir = "/kaggle/working/data"
        else:
            safe_dir = "./data"
        os.makedirs(safe_dir, exist_ok=True)
        return safe_dir
    return data_dir


def get_cifar10(data_dir="./data", train_subset=None, test_subset=None, seed=42):
    """Downloads and returns the CIFAR-10 train and test sets, optionally subsetted."""
    import os, shutil
    data_dir = _resolve_safe_data_dir(data_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    _auto_link_dataset(data_dir, "cifar10")
    lock = _acquire_download_lock(data_dir, "cifar10")
    try:
        try:
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        except (EOFError, RuntimeError, OSError, ValueError):
            # Auto-heal: delete corrupted partial download and retry
            for item in ["cifar-10-batches-py", "cifar-10-python.tar.gz"]:
                p = os.path.join(data_dir, item)
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.isfile(p):
                    try: os.remove(p)
                    except Exception: pass
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    finally:
        _release_download_lock(lock)
    
    rng = np.random.RandomState(seed)
    if train_subset is not None and train_subset < len(train_dataset):
        indices = rng.choice(len(train_dataset), train_subset, replace=False)
        train_dataset = Subset(train_dataset, indices)
        
    if test_subset is not None and test_subset < len(test_dataset):
        indices = rng.choice(len(test_dataset), test_subset, replace=False)
        test_dataset = Subset(test_dataset, indices)

    return train_dataset, test_dataset


def get_cifar100(data_dir="./data", train_subset=None, test_subset=None, seed=42):
    """Downloads and returns the CIFAR-100 train and test sets, optionally subsetted."""
    import os, shutil
    data_dir = _resolve_safe_data_dir(data_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    _auto_link_dataset(data_dir, "cifar100")
    lock = _acquire_download_lock(data_dir, "cifar100")
    try:
        try:
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
        except (EOFError, RuntimeError, OSError, ValueError):
            # Auto-heal: delete corrupted partial download and retry
            for item in ["cifar-100-python", "cifar-100-python.tar.gz"]:
                p = os.path.join(data_dir, item)
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.isfile(p):
                    try: os.remove(p)
                    except Exception: pass
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
    finally:
        _release_download_lock(lock)
    
    rng = np.random.RandomState(seed)
    if train_subset is not None and train_subset < len(train_dataset):
        indices = rng.choice(len(train_dataset), train_subset, replace=False)
        train_dataset = Subset(train_dataset, indices)
        
    if test_subset is not None and test_subset < len(test_dataset):
        indices = rng.choice(len(test_dataset), test_subset, replace=False)
        test_dataset = Subset(test_dataset, indices)

    return train_dataset, test_dataset


def get_femnist(data_dir="./data", train_subset=None, test_subset=None, seed=42):
    """Downloads and returns the 62-class FEMNIST (EMNIST byclass) train and test sets, optionally subsetted."""
    import os, shutil
    data_dir = _resolve_safe_data_dir(data_dir)

    # EMNIST raw data requires 90 deg counter-clockwise rotation and horizontal flip
    # to orient characters upright matching standard visual conventions.
    transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.hflip(transforms.functional.rotate(img, -90))),
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,)),
    ])

    _auto_link_dataset(data_dir, "femnist")
    lock = _acquire_download_lock(data_dir, "emnist")
    try:
        try:
            train_dataset = datasets.EMNIST(data_dir, split="byclass", train=True, download=True, transform=transform)
            test_dataset = datasets.EMNIST(data_dir, split="byclass", train=False, download=True, transform=transform)
        except (EOFError, RuntimeError, OSError, ValueError):
            # Auto-heal: delete corrupted partial download and retry
            emnist_dir = os.path.join(data_dir, "EMNIST")
            if os.path.exists(emnist_dir):
                shutil.rmtree(emnist_dir, ignore_errors=True)
            train_dataset = datasets.EMNIST(data_dir, split="byclass", train=True, download=True, transform=transform)
            test_dataset = datasets.EMNIST(data_dir, split="byclass", train=False, download=True, transform=transform)
    finally:
        _release_download_lock(lock)

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
