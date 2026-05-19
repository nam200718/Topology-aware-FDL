import pytest
import torch
from src.data.dataset import get_cifar10
from src.core.model import SimpleCNN

def test_simple_cnn_cifar10():
    # Verify SimpleCNN with 3 channels can handle CIFAR-10 dimensions (3, 32, 32)
    model = SimpleCNN(in_channels=3)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

def test_get_cifar10():
    # Make sure get_cifar10 function returns Subsets/Datasets correctly
    train_ds, test_ds = get_cifar10(data_dir="./data", train_subset=10, test_subset=5)
    assert len(train_ds) == 10
    assert len(test_ds) == 5
    
    # Verify shape of first sample
    img, label = train_ds[0]
    assert img.shape == (3, 32, 32)
    assert 0 <= label < 10
