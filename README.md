# Topology-Aware Federated Learning Framework

A modular, high-performance **Federated Learning (FL) Research Simulator** built in PyTorch. This framework enables the simulation, benchmark, and evaluation of complex multi-tier network topologies, personalization dynamics, edge compute optimizations, and Byzantine threat vectors under severe Non-IID data heterogeneity.

---

## Table of Contents

- [Overview](#overview)
- [Core Architecture](#core-architecture)
- [Supported Topologies](#supported-topologies)
- [Hierarchical Ensemble Topology](#hierarchical-ensemble-topology)
- [Byzantine Threat Library](#byzantine-threat-library)
- [Data & Non-IID Partitioning](#data--non-iid-partitioning)
- [Installation & Setup](#installation--setup)
- [Execution & Experimentation Guide](#execution--experimentation-guide)
- [Configuration Reference](#configuration-reference)
- [Outputs & Visualizations](#outputs--visualizations)
- [License](#license)

---

## Overview

Traditional Federated Learning relies on a flat **Star topology** (FedAvg), where all edge clients communicate with a single central server. In heterogeneous, edge-cloud, or network-constrained deployments, flat topologies suffer from communication bottlenecks, client drift under Non-IID data distributions, and vulnerability to Byzantine attacks.

This framework allows researchers to simulate, benchmark, and compare alternative network graphs:
- **Hierarchical & Ensemble Topologies**: Multi-tier architectures modeling Edge-Cloud paradigms.
- **Decentralized Graphs**: Peer-to-peer (Gossip) and Ring communication topologies.
- **Deep DAG Topologies**: Layered networks with intra-layer lateral communication.
- **Robustness & Non-IID Evaluation**: Sweeping Dirichlet concentration parameters ($\alpha$) and Byzantine attack scenarios.

---

## Core Architecture

The codebase follows a decoupled **Engine-Topology-Updater-Aggregator** pattern:

```
                          +-------------------------+
                          |      Global Server      |
                          +-------------------------+
                                      ^
                                      | Global Aggregation
                                      v
                          +-------------------------+
                          |   Cluster Heads (Heads) |
                          +-------------------------+
                                      ^
                                      | Intra-Cluster Aggregation
                                      v
                          +-------------------------+
                          |    Edge Clients (0..N)  |
                          +-------------------------+
```

### Key Components

- **`BaseEngine`** (`src/core/base_engine.py`): Base orchestrator managing round execution, data partitioning, metric logging, and evaluation.
- **`HierarchicalEngine` & `HierarchicalEnsembleEngine`** (`src/core/hierarchical_engine.py`, `src/core/hierarchical_ensemble_engine.py`): Implements 2-tier cluster aggregation and 3-tier ensemble routing.
- **`PyTorchLocalUpdater`** (`src/core/updater.py`): Executes local SGD updates, mutual distillation loss, and shared-backbone multi-head forward passes on GPU/CPU.
- **`FedAvgAggregator` & Aggregators** (`src/core/aggregator.py`): Aggregates client weights (sample-weighted or entropy-weighted).
- **`ClientState`** (`src/core/interfaces.py`): Data structure holding client weights, parent weights, local weights, data sample counts, and Byzantine attack flags.

---

## Supported Topologies

1. **Star Topology (`star`)**: Standard centralized server (FedAvg). All clients upload to and download from the global server.
2. **Star Randomized (`star_randomized`)**: Centralized topology where a random subset of clients participates each round.
3. **Ring Topology (`ring`)**: Decentralized ring graph. Clients aggregate model updates with their left/right structural neighbors.
4. **Gossip Topology (`gossip`)**: Peer-to-peer graph. Clients exchange updates with $k$ randomly chosen neighbors per round.
5. **Hierarchical Topology (`hierarchical`)**: 2-tier topology where clients are grouped into clusters. Cluster Heads aggregate member updates locally before communicating with the Global Server.
6. **Hierarchical Ensemble Topology (`hierarchical_ensemble`)**: 3-tier topology where clients maintain Root, Parent, and Local representations.
7. **Layered Topology (`layered`)**: Deep Directed Acyclic Graph (DAG) with intra-layer gossip communication between intermediate nodes.

---

## Hierarchical Ensemble Topology

The **Hierarchical Ensemble Topology** is a 3-tier architecture designed to maintain high accuracy under severe Non-IID skew ($\alpha \le 0.05$) while preserving edge compute efficiency.

### 1. Three-Tier Model Architecture
Each client maintains three distinct representations during training:
- **Root Model ($M_{\text{root}}$)**: Global server representation learning universal features across all clients.
- **Parent Model ($M_{\text{parent}}$)**: Cluster head representation learning domain-specialized features for a specific cluster.
- **Local Model ($M_{\text{local}}$)**: Persistent personalized representation adapted exclusively to local edge data.

### 2. Dual Aggregation Routing
Aggregation updates are routed along distinct topological pathways:
- Trained **Root updates** $\rightarrow$ Sent to Global Server for global aggregation.
- Trained **Parent updates** $\rightarrow$ Sent to Cluster Heads for local cluster aggregation.
- Trained **Local updates** $\rightarrow$ Retained on edge clients (never uploaded).

### 3. Edge Compute Optimization (`shared_backbone`)
To prevent 3x compute/memory overhead on edge devices, the client uses `MultiHeadSimpleCNN`:
- **Single Shared Convolutional Backbone**: Feature extraction (`conv1`, `pool`, `conv2`, `fc1`) runs **only once** per mini-batch.
- **Three Linear Heads**: `fc2_root`, `fc2_parent`, and `fc2_local` compute predictions with negligible extra FLOPs (~65% total compute speedup).

### 4. Dynamic Confidence Ensemble Weighting
During inference, predictions are combined dynamically using prediction entropy $H(M) = -\sum p \log p$:
$$\text{logits}_{\text{ensemble}} = w_{\text{local}} \cdot z_{\text{local}} + w_{\text{parent}} \cdot z_{\text{parent}} + w_{\text{root}} \cdot z_{\text{root}}$$
$$w_m = \frac{\exp(-H(M_m) / \tau)}{\sum_k \exp(-H(M_k) / \tau)}$$

### 5. Inter-Model Mutual Distillation
During local updates, KL divergence loss transfers soft-label knowledge between the Root, Parent, and Local heads to prevent representation drift:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{distill}} \cdot \mathcal{L}_{\text{KL}}(P_{\text{sub-model}} \parallel P_{\text{ensemble}})$$

### 6. Label-Distribution Aware Graph Construction
Clients are grouped into cluster heads using K-Means clustering on normalized client label frequency vectors (`cluster_by_label_dist: true`), ensuring clients with similar Non-IID skew share the same parent node.

---

## Byzantine Threat Library

The simulator includes native implementations of common federated learning attack vectors (`src/core/updater.py`):

- **`label_flip`**: Adversarial clients flip ground-truth labels ($y' = (N - 1) - y$).
- **`gradient_ascent`**: Adversarial clients invert the sign of the local loss to actively degrade global model performance.
- **`sign_flip`**: Adversarial clients negate parameter update vectors ($\Delta W' = -\Delta W$).
- **`random_noise`**: Adversarial clients overwrite parameter updates with Gaussian noise ($\mathcal{N}(0, \sigma^2)$).

Attack intensity is controlled via `byzantine_rate` (e.g. `0.2` = 20% malicious nodes).

---

## Data & Non-IID Partitioning

### Datasets
Supported datasets include **MNIST** (1-channel) and **CIFAR-10** (3-channel), loaded via `src/data/dataset.py`.

### Non-IID Dirichlet Partitioning
Data is partitioned across clients using a symmetric Dirichlet distribution $\text{Dir}(\alpha)$:
- **$\alpha \to \infty$**: Uniform IID distribution across all clients.
- **$\alpha = 1.0$**: Mild Non-IID skew.
- **$\alpha = 0.5$**: Moderate Non-IID skew.
- **$\alpha = 0.1$**: Severe Non-IID skew.
- **$\alpha = 0.05$**: Extreme Non-IID skew (clients hold samples from 1-2 classes).

### FastDataset Acceleration
When running on GPU (`--device cuda` or CPU acceleration), datasets are preloaded directly onto target memory to eliminate host-to-device transfer overhead during training loops.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/[your-username]/Topology-aware-FDL.git
cd Topology-aware-FDL
```

### 2. Create Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### On Linux / macOS (Bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

#### CPU Installation:
```bash
pip install -r requirements.txt
```

#### NVIDIA GPU Support (CUDA 12.1+):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Execution & Experimentation Guide

### 1. Run Single Simulation (`main.py`)
Run a single experiment configuration (e.g., Star topology baseline):
```bash
python main.py
```
To run using a specific YAML config file:
```bash
python main.py --config configs/ensemble_experiment.yaml
```

### 2. Run Topology Comparison Study (`scripts/run_comparison.py`)
Compare standard baseline topologies (**Star, Hierarchical Standard**) against the **Optimized Hierarchical Ensemble Topology** across IID and Non-IID Dirichlet spectrum scenarios ($\alpha \in \{1.0, 0.5, 0.1, 0.05\}$):
```bash
python scripts/run_comparison.py
```
Or specify a custom comparison config:
```bash
python scripts/run_comparison.py --config configs/comparison.yaml
```

### 3. Run Byzantine Matrix Sweep (`scripts/run_byzantine_matrix.py`)
Sweep Byzantine attack rates across topologies:
```bash
python scripts/run_byzantine_matrix.py
```

### 4. Run Automated Unit Tests (`pytest`)
Execute the full test suite:
```bash
python -m pytest tests/
```

---

## Configuration Reference

Experiments are configured via YAML files in the `configs/` directory.

### Example Configuration (`configs/comparison.yaml`):

```yaml
experiment_type: comparison
num_rounds: 15

env:
  seed: 42
  output_dir: ./outputs
  dataset: mnist
  train_subset: 10000
  test_subset: 2000

clients:
  num_clients: 15
  local_lr: 0.05
  local_steps: 1
  # Hierarchical Ensemble settings:
  ensemble_weighting_mode: dynamic_confidence # Options: static, dynamic_confidence, dynamic_loss
  compute_optimization_mode: shared_backbone  # Options: shared_backbone, frozen_root_anchor, head_only, none
  ensemble_distillation: true                 # Enable mutual distillation loss
  distillation_lambda: 0.5                    # Distillation loss weight

robustness:
  byzantine_type: label_flip

topologies:
  - type: star
    label: "Star (FedAvg)"
  - type: hierarchical
    label: "Hierarchical (Standard)"
    params:
      num_clusters: 3
  - type: hierarchical_ensemble
    label: "Hierarchical Ensemble (Optimized)"
    params:
      num_clusters: 3
      cluster_by_label_dist: true             # Enable label distribution aware clustering

scenarios:
  - id: iid
    label: "IID (Baseline)"
    non_iid: false
  - id: non_iid_alpha_1.0
    label: "Non-IID (Mild alpha=1.0)"
    non_iid: true
    alpha: 1.0
  - id: non_iid_alpha_0.5
    label: "Non-IID (Moderate alpha=0.5)"
    non_iid: true
    alpha: 0.5
  - id: non_iid_alpha_0.1
    label: "Non-IID (Severe alpha=0.1)"
    non_iid: true
    alpha: 0.1
  - id: non_iid_alpha_0.05
    label: "Non-IID (Extreme alpha=0.05)"
    non_iid: true
    alpha: 0.05
```

---

## Outputs & Visualizations

Running an experiment automatically creates a timestamped output folder in `./outputs/`:

```
outputs/
└── comparison_study_20260724_132743/
    ├── comparison_results.csv        # Tabular summary of final accuracies
    ├── summary.json                  # JSON summary of all runs & metrics
    ├── metrics/                      # Detailed per-round JSON & CSV logs for each run
    │   ├── star_iid/
    │   ├── hierarchical_non_iid_alpha_0.1/
    │   └── hierarchical_ensemble_non_iid_alpha_0.05/
    └── plots/                        # Multi-panel visualizations
        ├── convergence_iid.png
        ├── convergence_non_iid_alpha_0.1.png
        └── robustness_summary.png
```

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.
