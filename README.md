# FedlEARNING: Federated Learning Topology Research Simulator

`FedlEARNING` is a high-fidelity federated learning research simulator designed to evaluate the impact of network topology on convergence rates and robustness against Byzantine attacks. 

By leveraging **PyTorch** for local training dynamics and a modular **Engine-Topology** architecture, this tool enables the simulation of complex multi-tier communication graphs at scale.

## 🚀 Key Features

*   **Diverse Topologies**: Support for Star (Centralized), Ring, Gossip (P2P), Hierarchical (2-tier), **Layered** (Deep Hierarchical), and **Hierarchical Ensemble** networks.
*   **Byzantine Attack Library**: Native implementation of the following threat vectors:
    *   `label_flip`: Data poisoning via target remapping.
    *   `gradient_ascent`: Active loss maximization.
    *   `sign_flip`: Parameter sign inversion.
    *   `random_noise`: Gaussian parameter pollution.
*   **Deep Layered Aggregation**: A neural network-like DAG structure with **Intra-Layer Gossip** for lateral knowledge sharing.
*   **Research-Ready Output**: Automated generation of CSV/JSON metrics and multi-panel convergence visualizations using Matplotlib/Seaborn.
*   **Reproducibility**: Strict deterministic execution paths using seeded RNG streams across all components.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/FedlEARNING.git
   cd FedlEARNING
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # For CPU-only or default installation:
   pip install -r requirements.txt

   # 🟢 For NVIDIA GPU Support (CUDA 11.8 or 12.1+):
   # Install PyTorch with CUDA support first, then install the rest of the requirements:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

## 📊 Quick Start

### 1. Run a Smoke Test (Star Topology)
Execute the default single-run configuration on MNIST:
```bash
python main.py
```

### 2. Run the Full Experiment Matrix
Benchmark all topologies (Star, Ring, Gossip, Hierarchical, Layered) across multiple Byzantine rates (0.0, 0.1, 0.3):
```bash
python main.py --matrix
```

## 📖 Documentation

For detailed architecture explanations, experiment configuration guides, and research claims verification, refer to the documentation in the `/docs` directory:

*   [COMPLETE_GUIDE.md](docs/COMPLETE_GUIDE.md) – In-depth research and implementation details.
*   [ARCHITECTURE.md](docs/ARCHITECTURE.md) – Internal components and engine logic.
*   [EXPERIMENTS.md](docs/EXPERIMENTS.md) – Guidance on designing new research experiments.

## ⚖️ License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
