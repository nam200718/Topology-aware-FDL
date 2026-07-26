import os
import tarfile
import urllib.request

def download_cifar10_permanently(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    target_folder = os.path.join(data_dir, "cifar-10-batches-py")
    
    if os.path.exists(target_folder) and len(os.listdir(target_folder)) >= 5:
        print(f"CIFAR-10 is already downloaded and extracted in '{os.path.abspath(target_folder)}'.")
        return

    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    
    print(f"Downloading CIFAR-10 directly from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as response, open(tar_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print("Download finished. Extracting archive...")
    
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
        
    print(f"-> Successfully extracted to '{os.path.abspath(target_folder)}'!")
    if os.path.exists(tar_path):
        try:
            os.remove(tar_path)
        except Exception:
            pass
    print("CIFAR-10 is now permanently stored locally. PyTorch will load it instantly from disk!")

if __name__ == "__main__":
    download_cifar10_permanently()
