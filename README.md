# Topology-Aware Federated Learning Framework

A modular, high-performance **Federated Learning (FL) Research Simulator** built in PyTorch. This framework enables the simulation, benchmarking, and evaluation of multi-tier network topologies, personalization algorithms, edge compute optimizations, and Byzantine threat vectors under severe Non-IID data heterogeneity.

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

This framework allows researchers to simulate, benchmark, and compare:
- **Hierarchical Ensemble Topology**: A 3-tier architecture with Root, Parent, and Local representations combined via dynamic confidence-weighted ensemble inference.
- **Flat Star Baselines**: Standard FedAvg, Ditto (proximal regularization), and APFL (adaptive personalized FL).
- **Decentralized Graphs**: Peer-to-peer Gossip and Ring communication topologies.
- **Deep DAG Topologies**: Layered networks with intra-layer lateral communication.
- **Robustness & Non-IID Evaluation**: Sweeping Dirichlet concentration parameters ($\alpha$) and Byzantine attack scenarios.

---

## Core Architecture

The codebase follows a decoupled **Engine → Topology → Updater → Aggregator** pattern:

```
                          +-------------------------+
                          |      Global Server      |   ← Root FedAvg aggregation
                          +-------------------------+
                                      ^
                                      | Global Aggregation (Root weights)
                                      v
                          +-------------------------+
                          |      Cluster Heads      |   ← Parent FedAvg aggregation
                          +-------------------------+
                                      ^
                                      | Intra-Cluster Aggregation (Parent weights)
                                      v
                          +-------------------------+
                          |    Edge Clients (0..N)  |   ← Local weights (never uploaded)
                          +-------------------------+
```

### Key Components

| Component | File | Role |
|:---|:---|:---|
| `BaseEngine` | `src/core/base_engine.py` | Base orchestrator: round execution, data partitioning, metric logging |
| `CentralizedEngine` | `src/core/centralized_engine.py` | Star topology engine (FedAvg, Ditto, APFL) |
| `DecentralizedEngine` | `src/core/decentralized_engine.py` | Gossip and Ring topology engine |
| `HierarchicalEngine` | `src/core/hierarchical_engine.py` | Standard 2-tier hierarchical engine |
| `HierarchicalEnsembleEngine` | `src/core/hierarchical_ensemble_engine.py` | 3-tier ensemble engine with adaptive step allocation |
| `LayeredEngine` | `src/core/layered_engine.py` | Deep DAG layered topology engine |
| `PyTorchLocalUpdater` | `src/core/updater.py` | Local SGD, mutual distillation loss, shared-backbone multi-head forward passes |
| `FedAvgAggregator` | `src/core/aggregator.py` | Sample-weighted parameter averaging |
| `ClientState` | `src/core/interfaces.py` | Client data structure: weights, parent weights, local weights, Byzantine flags |

---

## Supported Topologies

| Type ID | Label | Description |
|:---|:---|:---|
| `star` | Star | Centralized server; all clients upload/download each round. Supports `personalization_method: none/ditto/apfl`. |
| `star_randomized` | Star Randomized | Star topology where a random client subset participates each round. |
| `ring` | Ring | Decentralized ring graph; clients aggregate with their left/right structural neighbors. |
| `gossip` | Gossip | Peer-to-peer graph; clients exchange updates with $k$ randomly chosen neighbors per round. |
| `hierarchical` | Hierarchical (Standard) | 2-tier: cluster heads aggregate member updates locally before communicating with the global server. |
| `hierarchical_ensemble` | Hierarchical Ensemble | 3-tier: Root, Parent, and Local heads with dynamic confidence-weighted ensemble inference. |
| `layered` | Layered DAG | Deep Directed Acyclic Graph with intra-layer gossip communication between intermediate nodes. |

---

## Hierarchical Ensemble Topology

The **Hierarchical Ensemble Topology** (`hierarchical_ensemble`) is a 3-tier architecture designed to maintain high personalized accuracy under severe Non-IID data skew while preserving edge compute efficiency.

### 1. Three-Tier Model Architecture

Each client maintains three distinct model representations simultaneously:

| Head | Scope | Aggregation |
|:---|:---|:---|
| **Root** ($M_\text{root}$) | Global — learns universal features across all clients | Sent to Global Server (FedAvg across all clients) |
| **Parent** ($M_\text{parent}$) | Cluster — learns domain-specialized features for a specific cluster | Sent to Cluster Head (FedAvg within cluster) |
| **Local** ($M_\text{local}$) | Personal — adapts exclusively to local edge data | Retained on device, never uploaded |

### 2. Dual Aggregation Routing

Trained model updates are routed along distinct topological pathways each round:
- **Root updates** → Global Server → FedAvg across all participating clients → broadcast back
- **Parent updates** → Cluster Head → FedAvg within cluster → broadcast to cluster members
- **Local updates** → Retained on device (never communicated)

### 3. Adaptive Training Step Allocation

Each round, the total local step budget is distributed across the three heads proportionally to how much each head is still improving, measured by its per-round loss improvement rate:

$$s_m^{(t)} = s_\min + (S_\text{budget} - K \cdot s_\min) \cdot \frac{\Delta L_m^{(t)}}{\sum_k \Delta L_k^{(t)}}$$

This naturally gives more steps to undertrained heads without any fixed arbitrary ratio.

### 4. Shared-Backbone Compute Optimization

To prevent 3× memory and compute overhead, the client uses a single `MultiHead` model with:
- **One shared convolutional backbone** — feature extraction runs **once** per mini-batch
- **Three linear classification heads** — `fc_root`, `fc_parent`, `fc_local` add negligible additional FLOPs

### 5. Dynamic Confidence Ensemble Weighting

At inference, predictions from all three heads are combined via prediction entropy weighting:

$$w_m = \frac{\exp(-H(M_m) / \tau)}{\sum_k \exp(-H(M_k) / \tau)}, \quad H(M) = -\sum p \log p$$

$$\text{logits}_\text{ensemble} = w_\text{local} \cdot z_\text{local} + w_\text{parent} \cdot z_\text{parent} + w_\text{root} \cdot z_\text{root}$$

Heads producing more confident predictions receive proportionally higher ensemble weight.

### 6. Root-Anchored Mutual Distillation

During local training, KL divergence loss transfers soft-label knowledge from the Root head (broadest view) to the Parent and Local heads, preventing representation drift:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{CE} + \lambda_\text{distill} \cdot \mathcal{L}_\text{KL}(P_\text{sub} \| P_\text{root})$$

### 7. Label-Distribution Aware Clustering

Clients are grouped into clusters using K-Means on normalized client label frequency vectors (`cluster_by_label_dist: true`), ensuring clients with similar Non-IID skew share the same parent node and benefit from cluster-level specialization.

---

## Byzantine Threat Library

The simulator includes native implementations of common FL attack vectors (`src/core/updater.py`):

| Attack | Description |
|:---|:---|
| `label_flip` | Adversarial clients flip ground-truth labels: $y' = (N-1) - y$ |
| `gradient_ascent` | Clients invert the local loss sign to actively degrade the global model |
| `sign_flip` | Clients negate all parameter update vectors: $\Delta W' = -\Delta W$ |
| `random_noise` | Clients replace updates with Gaussian noise: $\mathcal{N}(0, \sigma^2)$ |

Attack intensity is controlled via `byzantine_rate` (e.g., `0.2` = 20% malicious clients).

---

## Data & Non-IID Partitioning

### Datasets

| Dataset | Channels | Classes |
|:---|:---:|:---:|
| MNIST | 1 | 10 |
| CIFAR-10 | 3 | 10 |

### Non-IID Dirichlet Partitioning

Data is partitioned across clients using a symmetric Dirichlet distribution $\text{Dir}(\alpha)$:

| α | Regime | Description |
|:---:|:---|:---|
| ∞ | IID | Uniform distribution across all clients |
| 1.0 | Mild Non-IID | Slight label imbalance per client |
| 0.5 | Moderate Non-IID | Noticeable class skew per client |
| 0.1 | Severe Non-IID | Most clients hold 2–3 dominant classes |
| 0.05 | Extreme Non-IID | Clients effectively hold 1–2 classes |

### FastDataset Acceleration

When running on GPU, datasets are preloaded directly onto device memory to eliminate host-to-device transfer overhead during training loops.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/[your-username]/Topology-aware-FDL.git
cd Topology-aware-FDL
```

### 2. Create Virtual Environment

#### Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

#### CPU:
```bash
pip install -r requirements.txt
```

#### NVIDIA GPU (CUDA 12.1+):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### AMD GPU (DirectML — RX 6000/7000 Series on Windows):
```powershell
# Automated setup (creates a Python 3.12 venv):
.\setup_gpu.ps1
```
Or manually:
```bash
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install torch==2.3.1 torchvision==0.18.1 torch-directml
pip install -r requirements.txt
```

---

## Execution & Experimentation Guide

### 1. Run a Single Simulation
```bash
python main.py --config configs/ensemble_experiment.yaml
```

### 2. Run the Comparison Study
Runs all configured topologies across all Non-IID scenarios. Results and charts are saved automatically to a timestamped output directory:
```bash
python main.py --config configs/comparison.yaml
```

### 3. Run Byzantine Matrix Sweep
Sweeps Byzantine attack rates across topologies:
```bash
python scripts/run_byzantine_matrix.py
```

### 4. Run Unit Tests
```bash
python -m pytest tests/
```

---

## Configuration Reference

Experiments are configured via YAML files in the `configs/` directory.

### Example: Comparison Config (`configs/comparison.yaml`)

```yaml
experiment_type: comparison
num_rounds: 15

env:
  seed: 42
  output_dir: ./outputs
  dataset: cifar10
  train_subset: 3000   # Optional subset size for faster iteration
  test_subset: 1000

clients:
  model_name: resnet9          # Options: simple_cnn, resnet9
  num_clients: 15
  local_lr: 0.01
  local_steps: 1
  # Hierarchical Ensemble settings:
  compute_optimization_mode: shared_backbone    # Options: shared_backbone, frozen_root_anchor, head_only, none
  ensemble_weighting_mode: dynamic_confidence   # Options: static, dynamic_confidence, dynamic_loss
  ensemble_distillation: true
  distillation_lambda: 0.5
  total_local_steps: 5

robustness:
  byzantine_type: label_flip

topologies:
  - type: star
    label: "Star (FedAvg)"
    params:
      personalization_method: "none"
  - type: star
    label: "Star (Ditto)"
    params:
      personalization_method: "ditto"
      ditto_lambda: 0.1
  - type: star
    label: "Star (APFL)"
    params:
      personalization_method: "apfl"
      apfl_alpha: 0.5
  - type: hierarchical_ensemble
    label: "Hierarchical Ensemble (Intrinsic Optimized)"
    params:
      num_clusters: 3
      cluster_by_label_dist: true

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

### Key `clients` Parameters

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `model_name` | str | `simple_cnn` | Model architecture: `simple_cnn` or `resnet9` |
| `num_clients` | int | 100 | Total number of federated clients |
| `local_lr` | float | 0.1 | SGD learning rate for local updates |
| `local_steps` | int | 5 | Local SGD steps per round (non-ensemble topologies) |
| `total_local_steps` | int | 5 | Total step budget distributed adaptively across ensemble heads |
| `compute_optimization_mode` | str | `shared_backbone` | Edge compute strategy for the ensemble |
| `ensemble_weighting_mode` | str | `dynamic_confidence` | Inference weighting: `static`, `dynamic_confidence`, `dynamic_loss` |
| `ensemble_distillation` | bool | `true` | Enable Root-anchored mutual distillation during local training |
| `distillation_lambda` | float | 0.5 | Distillation loss weight $\lambda$ |
| `personalization_method` | str | `none` | Star topology personalization: `none`, `ditto`, `apfl` |
| `ditto_lambda` | float | 0.1 | Ditto proximal penalty $\lambda$ |
| `apfl_alpha` | float | 0.5 | APFL initial mixing weight $\alpha$ |

---

## Outputs & Visualizations

Running an experiment automatically creates a timestamped output folder in `./outputs/`:

```
outputs/
└── comparison_study_20260726_154200/
    ├── comparison_results.csv        # Tabular summary of all final accuracies
    ├── summary.json                  # JSON summary with full per-topology per-scenario records
    ├── metrics/                      # Per-round logs for each individual run
    │   ├── star_fedavg_iid/
    │   │   ├── metrics.json
    │   │   └── metrics.csv
    │   ├── star_ditto_non_iid_alpha_0.5/
    │   ├── star_apfl_non_iid_alpha_0.1/
    │   └── hierarchical_ensemble_intrinsic_optimized_non_iid_alpha_0.05/
    └── plots/                        # Generated visualizations
        ├── convergence_iid.png
        ├── convergence_non_iid_alpha_1.0.png
        ├── convergence_non_iid_alpha_0.5.png
        ├── convergence_non_iid_alpha_0.1.png
        ├── convergence_non_iid_alpha_0.05.png
        └── robustness_summary.png
```

Metric directories are named after the sanitized topology label and scenario ID (e.g., `star_fedavg_iid`, `star_ditto_iid`), ensuring uniqueness even when multiple topologies share the same `type` string but differ in algorithm.

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.
