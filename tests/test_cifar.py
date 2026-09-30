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

def test_kaggle_auto_linking(tmp_path):
    import os
    from src.data.dataset import _auto_link_dataset, _resolve_safe_data_dir

    # 1. Test nested directory detection
    kaggle_mock = tmp_path / "kaggle_input" / "cifar100_mock"
    cifar_sub = kaggle_mock / "cifar-100-python"
    cifar_sub.mkdir(parents=True)
    (cifar_sub / "train").write_text("train_dummy")
    (cifar_sub / "test").write_text("test_dummy")
    (cifar_sub / "meta").write_text("meta_dummy")

    target_data_dir = tmp_path / "working" / "data"
    target_data_dir.mkdir(parents=True)

    os.environ["KAGGLE_DATASET_PATH"] = str(kaggle_mock)
    try:
        found = _auto_link_dataset(str(target_data_dir), "cifar100")
        assert found is True
        assert os.path.exists(target_data_dir / "cifar-100-python" / "train")
    finally:
        os.environ.pop("KAGGLE_DATASET_PATH", None)

    # 2. Test flat directory detection
    flat_mock = tmp_path / "kaggle_input" / "cifar100_flat"
    flat_mock.mkdir(parents=True)
    (flat_mock / "train").write_text("train_dummy")
    (flat_mock / "test").write_text("test_dummy")
    (flat_mock / "meta").write_text("meta_dummy")

    target_data_dir_2 = tmp_path / "working" / "data2"
    target_data_dir_2.mkdir(parents=True)

    os.environ["KAGGLE_DATASET_PATH"] = str(flat_mock)
    try:
        found = _auto_link_dataset(str(target_data_dir_2), "cifar100")
        assert found is True
        assert os.path.exists(target_data_dir_2 / "cifar-100-python" / "train")
    finally:
        os.environ.pop("KAGGLE_DATASET_PATH", None)

    # 3. Test double-nested directory detection (dataset named 'cifar-100-python' containing 'cifar-100-python')
    double_mock = tmp_path / "kaggle_input" / "cifar-100-python"
    inner_mock = double_mock / "cifar-100-python"
    inner_mock.mkdir(parents=True)
    (inner_mock / "train").write_text("train_dummy")
    (inner_mock / "test").write_text("test_dummy")
    (inner_mock / "meta").write_text("meta_dummy")

    target_data_dir_3 = tmp_path / "working" / "data3"
    target_data_dir_3.mkdir(parents=True)

    os.environ["KAGGLE_DATASET_PATH"] = str(tmp_path / "kaggle_input")
    try:
        found = _auto_link_dataset(str(target_data_dir_3), "cifar100")
        assert found is True
        assert os.path.exists(target_data_dir_3 / "cifar-100-python" / "train")
    finally:
        os.environ.pop("KAGGLE_DATASET_PATH", None)

def test_resolve_safe_data_dir(tmp_path):
    import os
    from src.data.dataset import _resolve_safe_data_dir

    # Point to a path under /kaggle/input (simulated)
    fake_kaggle_path = "/kaggle/input/cifar100"
    safe_dir = _resolve_safe_data_dir(fake_kaggle_path)
    assert not safe_dir.startswith("/kaggle/input")
    assert os.environ.get("KAGGLE_DATASET_PATH") == fake_kaggle_path
    os.environ.pop("KAGGLE_DATASET_PATH", None)

