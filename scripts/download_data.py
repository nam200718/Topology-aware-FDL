import os
import torch
from torchvision import datasets

def download_all_datasets(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading datasets permanently to '{os.path.abspath(data_dir)}'...")
    
    # 1. Download MNIST
    print("-> Pre-fetching MNIST...")
    datasets.MNIST(data_dir, train=True, download=True)
    datasets.MNIST(data_dir, train=False, download=True)
    print("   MNIST downloaded successfully.")

    # 2. Download CIFAR-10 using fast mirror
    print("-> Pre-fetching CIFAR-10...")
    datasets.CIFAR10.url = "https://huggingface.co/datasets/anon8231489241/cifar10/resolve/main/cifar-10-python.tar.gz"
    datasets.CIFAR10(data_dir, train=True, download=True)
    datasets.CIFAR10(data_dir, train=False, download=True)
    print("   CIFAR-10 downloaded successfully.")
    
    print("\nAll datasets permanently stored in ./data/.")
    print("Subsequent experiment runs will load instantly from disk without downloading!")

if __name__ == "__main__":
    download_all_datasets()
