import json
import os

def build_session2_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🛡️ AAMAS 2027: Session 2 — Byzantine Robustness, Scalability & Edge Vision\n",
                "### Kaggle Account 2 (12-Hour GPU T4 Budget)\n",
                "\n",
                "**Paper Target:** *Hierarchical Ensemble Personalization in Federated Learning (HEP-FL)* (AAMAS 2027)  \n",
                "**Dataset:** **CIFAR-100** (Pre-attached in `/kaggle/input`, auto-linked in 0.0s)  \n",
                "**Target Hardware:** Tesla T4 GPU (16 GB VRAM)  \n",
                "**Assigned Jobs:**\n",
                "1. **Job 2 (Table 2)**: Byzantine Multi-Attack Robustness Suite (4 Defenses × 4 Attacks × 5 Rates = 80 evaluation points) calibrated at **30 rounds** (~3.9h budget with clean-baseline sharing).\n",
                "2. **Job 3 (Table 3)**: 50-Client Partial Participation Scaling ($C_p=0.2$, 10 active/round) with S-AFR (20 rounds, ~5m).\n",
                "3. **Job 4 (Table 4)**: MobileNetV3 Edge Vision Latency and Peak VRAM Profiling (instant reuse of verified artifacts or 30 rounds).\n",
                "\n",
                "> 💡 **Tip:** Set your Kaggle notebook accelerator to **GPU T4 x1**, then click **Save Version -> \"Save & Run All (Commit)\"** to run in the background!\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment Detection & Codebase Setup\n",
                "Pulls the latest verified repository with fault-tolerant checkpointing and Kaggle dataset detection.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys, shutil\n",
                "\n",
                "if os.path.exists('/kaggle/working/src'):\n",
                "    ROOT = '/kaggle/working'\n",
                "elif not os.path.exists('/kaggle/working/Topology-aware-FDL'):\n",
                "    print('🚀 Cloning repository from GitHub...')\n",
                "    !git clone https://github.com/nam200718/Topology-aware-FDL.git /kaggle/working/Topology-aware-FDL\n",
                "    ROOT = '/kaggle/working/Topology-aware-FDL'\n",
                "else:\n",
                "    ROOT = '/kaggle/working/Topology-aware-FDL'\n",
                "    print('🔄 Repository exists, syncing latest updates from main...')\n",
                "    !cd {ROOT} && git fetch origin main && git reset --hard origin/main\n",
                "\n",
                "OUT_DIR = os.path.join(ROOT, 'outputs')\n",
                "os.makedirs(OUT_DIR, exist_ok=True)\n",
                "%cd {ROOT}\n",
                "if ROOT not in sys.path:\n",
                "    sys.path.insert(0, ROOT)\n",
                "\n",
                "print('✅ Current Working Directory:', os.getcwd())\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Hardware Acceleration & Environment Verification\n",
                "Verifies GPU acceleration and sets fast cuDNN primitives.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip -q install pydantic pyyaml torchvision pandas matplotlib seaborn scipy\n",
                "import torch\n",
                "\n",
                "print(f'PyTorch Version: {torch.__version__}')\n",
                "print(f'CUDA Available: {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'Active GPU: {torch.cuda.get_device_name(0)}')\n",
                "    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                "    print(f'VRAM Capacity: {vram_gb:.2f} GB')\n",
                "    torch.backends.cudnn.benchmark = True\n",
                "else:\n",
                "    print('⚠️ No GPU detected! Please navigate to Notebook Settings -> Accelerator -> GPU T4 x1.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. ⚡ Pre-loaded Kaggle Dataset Auto-Discovery (0.0s)\n",
                "Directly verifies and symlinks `cifar-100-python` from `/kaggle/input` with zero network download.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.data.dataset import _auto_link_dataset\n",
                "\n",
                "data_dir = './data'\n",
                "found = _auto_link_dataset(data_dir, 'cifar100')\n",
                "if found:\n",
                "    print('⚡ [Kaggle Input] CIFAR-100 dataset successfully verified and linked! (0.0s latency, 0 downloads)')\n",
                "else:\n",
                "    print('ℹ️ Note: If CIFAR-100 is attached under /kaggle/input, the experiment runner will automatically link it.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. 🛡️ Job 2: Byzantine Multi-Attack Robustness Benchmark (Table 2)\n",
                "Evaluates 4 defenses (`fedavg`, `multikrum`, `ditto`, `topo_defended`) against 4 Byzantine attack types (`label_flip`, `sign_flip`, `gradient_ascent`, `random_noise`) across 5 corruption rates ($f \\in [0.0, 0.1, 0.2, 0.3, 0.4]$) at **30 communication rounds** with automatic checkpoint recovery.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run Job 2: Byzantine Robustness Suite (Table 2)\n",
                "!python -m src.baselines.run_all_baselines \\\n",
                "    --track byzantine \\\n",
                "    --methods fedavg multikrum ditto topo_defended \\\n",
                "    --attacks label_flip sign_flip gradient_ascent random_noise \\\n",
                "    --rates 0.0 0.1 0.2 0.3 0.4 \\\n",
                "    --rounds 30 \\\n",
                "    --seeds 42\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Inspect Job 2 Results Table\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, pandas as pd\n",
                "\n",
                "b_csv = 'outputs/baselines/byzantine/results_byzantine.csv'\n",
                "if os.path.exists(b_csv):\n",
                "    df_b = pd.read_csv(b_csv)\n",
                "    print(f'✅ Job 2 completed successfully! Total evaluated cells: {len(df_b)}')\n",
                "    show_cols = [c for c in ['method', 'attack', 'byzantine_rate', 'mean_acc', 'mean_b10', 'elapsed_s'] if c in df_b.columns]\n",
                "    display(df_b[show_cols])\n",
                "else:\n",
                "    print('⚠️ Checkpoint file not yet created.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. 👥 Job 3: 50-Client Scalability Benchmark with S-AFR (Table 3)\n",
                "Evaluates 50 clients with partial participation ($C_p=0.2$, 10 active clients/round) under Moderate (0.5) and Severe (0.1) Dirichlet non-IID on CIFAR-100.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run Job 3: 50-Client Scale Benchmark (Table 3)\n",
                "!python scripts/run_scale_50clients.py \\\n",
                "    --dataset cifar100 \\\n",
                "    --rounds 20 \\\n",
                "    --clients 50 \\\n",
                "    --clients-per-round 10 \\\n",
                "    --skip-if-exists\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. 📱 Job 4: MobileNetV3 Edge Vision Latency & Peak VRAM Profiling (Table 4)\n",
                "Profiles physical edge hardware metrics (latency, parameters, peak VRAM) and convergence under non-IID conditions on CIFAR-100.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run Job 4: MobileNetV3 Edge Hardware Profiling (Table 4)\n",
                "!python scripts/run_mobilenet_benchmark.py \\\n",
                "    --dataset cifar100 \\\n",
                "    --rounds 30 \\\n",
                "    --clients 15 \\\n",
                "    --skip-if-exists\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. 📦 Packaging Account 2 Artifacts\n",
                "Compresses all CSV and JSON results from Account 2 into a single zip file for 1-click download.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, zipfile\n",
                "\n",
                "zip_target = '/kaggle/working/account2_artifacts.zip'\n",
                "files_to_pack = [\n",
                "    'outputs/baselines/byzantine/results_byzantine.csv',\n",
                "    'outputs/baselines/byzantine/results_byzantine.json',\n",
                "    'outputs/scale_50clients_results.json',\n",
                "    'outputs/mobilenet_benchmark_results.json',\n",
                "]\n",
                "\n",
                "with zipfile.ZipFile(zip_target, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
                "    for f in files_to_pack:\n",
                "        if os.path.exists(f):\n",
                "            zf.write(f, arcname=os.path.basename(f))\n",
                "            print(f'  Added: {f} ({os.path.getsize(f)/1024:.1f} KB)')\n",
                "        else:\n",
                "            print(f'  Note: {f} not yet created.')\n",
                "\n",
                "if os.path.exists(zip_target):\n",
                "    print(f'\\n🎉 All Account 2 artifacts packaged into: {zip_target}')\n",
                "    print(f'File Size: {os.path.getsize(zip_target) / 1024:.1f} KB')\n",
                "    print('👉 You can now download \\'account2_artifacts.zip\\' directly from the Kaggle Output file explorer on the right!')\n"
            ]
        }
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

if __name__ == "__main__":
    nb = build_session2_notebook()
    targets = [
        "CIFAR100_baselines_2.ipynb",
        "kaggle_session2_byzantine_scaling_edge.ipynb",
        "colab/CIFAR100_baselines_2.ipynb",
    ]
    for target in targets:
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"✅ Generated {target} successfully ({len(nb['cells'])} cells).")
