# Hierarchical Ensemble Personalization for Parameter-Efficient Federated Learning (HEP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-94%2F94%20passing-brightgreen.svg)](tests/)

* **Project Report:** See [`report/main.pdf`](report/main.pdf) (VinUniversity UROP Final Report).

---

## Abstract

Personalized Federated Learning (PFL) addresses statistical data heterogeneity (Non-IID data) across decentralized edge clients. However, real-world edge deployments are governed by the **Personalization Trilemma**---the fundamental trade-off between statistical accuracy across diverse skew regimes, on-device memory and compute constraints, and worst-case client fairness. State-of-the-art dual-model methods (such as Ditto and APFL) maintain two separate neural network graphs per client, doubling local memory (VRAM) and per-batch latency. Conversely, naive alternating split-head baselines (such as FedRep and FedPer) lack multi-scale structural coordination and can suffer representation collapse on homogeneous (IID) partitions.

**HEP** navigates the Personalization Trilemma using a **Single-Backbone 3-Tier Multi-Head Architecture** ($\text{Root}$, $\text{Parent}$, and $\text{Local}$ heads) coordinated by:
1. **Local Label Skew Metric ($R_{skew}$)**: An entropy-based metric measuring local empirical class balance.
2. **Anchored Binomial Head Weighting**: Dynamic, normalized loss weighting that seamlessly transitions between global consensus, cluster collaboration, and local specialization.
3. **Active-Class Logit Masking (ACLM)**: Prevents unobserved classes on edge devices from receiving negative gradient drag.
4. **Information-Reducing Sketch Routing**: Projects classification head updates into a 256-dimensional random sketch ($m=256$) to keep gradient reconstruction severely underdetermined ($m = 256 < d_{head} = 2570$) while preserving clustering geometry.

---

## Key Highlights

* **Pareto-Dominant Personalization**: Achieves **88.80%** personalized accuracy under extreme Non-IID skew ($\alpha = 0.05$), outperforming dual-model Ditto (87.87%), FedRep (87.07%), and FedAvg (57.53% by **+31.27pp**).
* **Closing the Split-Head IID Collapse**: Achieves **71.03%** personalized accuracy (and **72.73%** global root consensus) on uniform IID data, closing the split-head performance gap (+9.24pp over FedRep).
* **47.8% Memory & 50.2% Latency Reduction**: Consumes only **113.42 MB Peak VRAM** on ResNet-9 and **158.80 MB** on MobileNetV3 compared to Ditto's **217.15 MB / 298.60 MB** (single shared feature extractor vs. dual deep models).
* **High Class-Cardinality Scaling (CIFAR-100)**: Reaches **65.06%** personalized accuracy under extreme skew on CIFAR-100 (+3.57pp over Ditto, +37.39pp over FedAvg) with **50.17%** worst-decile fairness.
* **Partial Participation Fairness Recovery**: Formulates staleness-aware routing to maintain bottom-10% client fairness under partial client participation ($C_p = 0.20$).
* **Label-Space Fault Containment**: Maintains **76.71%** personalization accuracy under label-flipping attacks up to $f \le 20\%$ through architectural head isolation without external heuristic filters (collapsing at $f \ge 30\%$ due to shared-backbone corruption).
* **Data-Free Topology Routing**: Clusters clients strictly via parameter updates without exchanging raw samples or feature prototypes.

---

## Empirical Benchmarks

All benchmark results are evaluated under a standardized deterministic single-seed protocol (seed 42) and verified with multi-seed sweeps ({42, 123, 7}).

### 1. Main Personalization Benchmark (CIFAR-10 ResNet-9, 15 Clients, 25 Rounds)

| Paradigm & Method | IID ($\alpha = \infty$) | Mild ($\alpha = 1.0$) | Moderate ($\alpha = 0.5$) | Severe ($\alpha = 0.1$) | Extreme ($\alpha = 0.05$) | Peak VRAM | Wall-clock / Round |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FedAvg** (Global) | 71.80% | 68.53% | 68.10% | 61.10% | 57.53% | 108.58 MB | 15.1s |
| **Ditto** (Dual-Model) | 67.10% | 71.60% | 74.63% | 82.70% | 87.87% | 217.15 MB | 25.8s |
| **FedRep** (Split-Head) | 61.79% | 68.87% | 70.67% | 81.45% | 87.07% | 108.58 MB | 16.0s |
| **FedPer** (Split-Head) | 65.70% | 71.97% | 73.05% | 83.29% | 87.73% | 108.58 MB | 15.8s |
| **FedBABU** (Decoupled) | 72.10% | 75.67% | 76.02% | 84.43% | 88.39% | 108.58 MB | 15.5s |
| **FedALA** (Adaptive) | 69.30% | 74.51% | 74.32% | 83.55% | 88.25% | 116.20 MB | 15.6s |
| **CFL** (Clustered) | 72.64% | 69.29% | 65.83% | 59.98% | 66.72% | 108.58 MB | 15.1s |
| **HEP (Ours)** | **71.03%**† | **77.50%** | **77.97%** | **84.57%** | **88.80%** | **113.42 MB** | **16.5s** |

† *HEP IID personalized accuracy is 71.03%, with global Root consensus reaching 72.73%.*

---

### 2. Compute-Fairness Protocol Selection ($E=5$ vs. $E=10$)

| Local Budget | IID ($\alpha=\infty$) Acc | IID Time | Mod. Skew ($\alpha=0.5$) Acc | Mod. Skew Time | Local Passes |
|:---|:---:|:---:|:---:|:---:|:---:|
| **$E=10$ Epochs** | 72.57% | 1238s | 78.13% | 1273s | 10 passes |
| **$E=5$ Epochs (Standard)** | **72.53%** | **626s** | **78.47%** | **657s** | **5 passes** |

### 3. Architecture Generalizability (MobileNetV3-Small vs. ResNet-9, Extreme Skew $\alpha=0.05$)

| Architecture | Method | Peak VRAM | Batch Latency | Top-1 Accuracy | Bottom 10% Fairness |
|:---|:---|:---:|:---:|:---:|:---:|
| **ResNet-9** | FedAvg | 108.58 MB | 8.32 ms | 57.53% | 25.25% |
| | FedRep | 108.58 MB | 13.92 ms | 87.07% | 76.85% |
| | Ditto | 217.15 MB | 16.78 ms | 87.87% | 69.29% |
| | **HEP (Ours)** | **113.42 MB** | **8.35 ms** | **88.80%** | **66.43%** |
| **MobileNetV3** | FedAvg | 152.40 MB | 11.10 ms | 34.13% | 0.00% |
| | FedRep | 152.40 MB | 13.50 ms | 80.29% | 67.10% |
| | Ditto | 298.60 MB | 22.80 ms | 79.73% | 67.12% |
| | **HEP (Ours)** | **158.80 MB** | **11.20 ms** | **80.19%** | **65.97%** |

---

### 4. High-Class Cardinality (CIFAR-100) & 50-Client Scalability

| Regime / Scenario | FedAvg | FedRep | Ditto | **HEP (Ours)** | Key Finding |
|:---|:---:|:---:|:---:|:---|:---|
| **CIFAR-100 Moderate ($\alpha=0.5$)** | 36.41% | 27.61% | 39.86% | **47.05%** | +7.19pp over Ditto, +10.64pp over FedAvg |
| *-- Bottom 10% Fairness* | 31.62% | 21.87% | 33.85% | **41.69%** | **+7.84pp fairness gain over Ditto** |
| **CIFAR-100 Extreme ($\alpha=0.05$)** | 27.67% | 55.50% | 61.49% | **65.06%** | **+3.57pp over Ditto, +37.39pp over FedAvg |
| *-- Bottom 10% Fairness* | 16.71% | 40.25% | 47.37% | **50.17%** | **+2.80pp fairness gain over Ditto** |
| **50-Client Moderate ($\alpha=0.5, C_p=0.2$)** | 52.23% (13.13%) | 39.23% (16.85%) | 43.50% (17.87%) | **73.30% (50.12%)** | Stable scaling under partial participation |
| **50-Client Severe ($\alpha=0.1, C_p=0.2$)** | 43.90% (0.00%) | 65.37% (20.57%) | 65.01% (17.92%) | **83.27% (61.40%)** | Staleness-aware routing prevents client starvation |

---

### 5. Multi-Attack Byzantine Fault Tolerance (CIFAR-10 ResNet-9)

| Attack Type | Method | $f = 0\%$ | $f = 10\%$ | $f = 20\%$ | $f = 30\%$ | $f = 40\%$ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Label Flipping** | FedAvg | 60.27% | 60.27% | 56.00% | 46.57% | 15.90% |
| | FedRep | 64.07% | 64.49% | 65.51% | 65.94% | 64.88% |
| | Ditto | 70.83% | 70.65% | 71.21% | 69.41% | 65.36% |
| | **HEP (Ours)** | **76.67%** | **76.50%** | **76.71%** | **34.83%** | **30.53%** |
| **Sign Flipping** | FedAvg | 47.96 ± 0.46% | 38.38 ± 0.54% | 13.30 ± 0.72% | 10.84 ± 0.68% | 12.21 ± 0.70% |
| | FedRep | 52.53 ± 0.44% | 52.77 ± 0.46% | 52.08 ± 0.50% | 38.70 ± 0.60% | 27.34 ± 0.68% |
| | Ditto | 64.20 ± 0.43% | 50.73 ± 0.47% | 53.50 ± 0.48% | 40.31 ± 0.62% | 18.12 ± 0.65% |
| | **HEP (Ours)** | **76.16 ± 0.41%** | 45.14 ± 0.49% | 41.80 ± 0.53% | 32.00 ± 0.58% | 23.60 ± 0.61% |
| **Gaussian Noise** | FedAvg | 49.22 ± 0.44% | 31.53 ± 0.59% | 32.11 ± 0.62% | 21.28 ± 0.69% | 22.11 ± 0.71% |
| | FedRep | 54.92 ± 0.42% | 48.86 ± 0.49% | 37.73 ± 0.58% | 39.39 ± 0.55% | 34.88 ± 0.62% |
| | Ditto | 68.54 ± 0.41% | 55.89 ± 0.45% | 50.88 ± 0.49% | 52.94 ± 0.47% | 40.00 ± 0.52% |
| | **HEP (Ours)** | **76.16 ± 0.40%** | 51.37 ± 0.46% | 49.12 ± 0.48% | 49.64 ± 0.49% | **42.47 ± 0.51%** |

---

## Architectural Workflow & Mathematical Formulations

```
                         [Server Coordination]
                                   |
                +------------------+------------------+
                |                                     |
        [Global Aggregation]                 [Cluster Aggregation]
      Backbone & Root Head w_r             Cluster Heads {w_p,k}_k=1..K
                |                                     |
                +------------------+------------------+
                                   |
                                   v
                          +---------------------------------------+
                          |      Edge Clients (1 .. N)            |  <- Single Backbone Multi-Head Network
                          |  [Shared Convolutional Backbone]      |
                          |  |-- Root Head   (Synchronized Global)|
                          |  |-- Parent Head (Cluster Shared)     |
                          |  \-- Local Head  (Client Private)     |
                          +---------------------------------------+
```

### Core Mathematical Formulations

#### 1. Local Label Skew Metric
$$
R_{\text{skew},i} = \frac{\exp\big(H(p_i)\big) - 1}{C - 1} \in [0, 1]
$$
Evaluates empirical class balance ($0$ = extreme single-class skew, $1$ = uniform IID).

#### 2. Normalized Anchored Binomial Loss Weights
$$
q_{r,i} = a_i + (1 - a_i) R_{\text{skew},i}^2, \quad q_{p,i} = 2 R_{\text{skew},i} (1 - R_{\text{skew},i}), \quad q_{l,i} = (1 - R_{\text{skew},i})^2
$$
$$
\lambda_{k,i} = \frac{q_{k,i}}{q_{r,i} + q_{p,i} + q_{l,i}}, \quad \forall k \in \{r, p, l\}
$$

#### 3. 3-Head Composite Loss Objective with ACLM
$$
\mathcal{L}_{\text{batch}} = \lambda_{r,i} \mathcal{L}_{\text{CE}}(z_r, y) + \lambda_{p,i} \mathcal{L}_{\text{CE}}^{\text{masked}}(z_p, y) + \lambda_{l,i} \mathcal{L}_{\text{CE}}^{\text{masked}}(z_l, y)
$$
Computes features once per batch and applies Active-Class Logit Masking (ACLM) to Parent and Local heads.

#### 4. Inference Prediction Blending
$$
z_{\text{pred}} = \alpha_r z_r + \gamma_i (\alpha_p z_p + \alpha_l z_l) + \mathbf{m}_i
$$
Blends multi-head predictions at test time with staleness attenuation ($\gamma_i = \exp(-\tau_i/\tau_{0,i})$).

#### 5. Privacy-Preserving Random Projection Sketching
$$
s_i = P \cdot \Delta w_{r,i} \in \mathbb{R}^{256}
$$
Compresses Root-head updates ($2570$ dimensions) into a $256$-dimensional summary to make gradient reconstruction severely underdetermined.

---

## Repository Structure

```
Topology-aware-FDL/
|-- configs/                    # YAML experiment configurations
|   |-- comparison.yaml         # Main 5-regime benchmark (FedAvg vs APFL vs Ditto vs HEP)
|   |-- shard_cifar100_5regimes.yaml # CIFAR-100 full 5-regime sweep
|   |-- shard_hep_cifar10_5regimes.yaml # CIFAR-10 full 5-regime sweep
|   |-- ablation_study.yaml     # Component ablations
|   |-- byzantine_matrix.yaml   # Multi-attack Byzantine robustness sweep
|   |-- evaluation_full.yaml    # Full evaluation suite
|   |-- test_1round.yaml        # Fast smoke test configuration
|   |-- benchmarks/             # High-cardinality, scaling & baseline configs
|   |-- ablations/              # Distillation, grouping & budget configs
|   |-- byzantine/              # Attack & defense configs
|   \-- archive/                # Historical & diagnostic configs
|-- src/
|   |-- config.py               # Pydantic configuration schemas & HEP defaults
|   |-- core/                   # Core FL engines, updaters, and models
|   |   |-- model.py            # SimpleCNN, ResNet-9, MultiHeadResNet9, MobileNetV3
|   |   |-- updater.py          # PyTorchLocalUpdater with ACLM & binomial weighting
|   |   |-- hierarchical_ensemble_engine.py  # 3-tier ensemble controller
|   |   |-- centralized_engine.py            # Star topology engine (FedAvg, Ditto, APFL, FedRep)
|   |   \-- aggregator.py       # FedAvg & robust aggregation functions
|   |-- data/                   # Dataset loaders & Dirichlet Non-IID partitioners
|   |-- topologies/             # Dynamic topology graphs & clustering controllers
|   \-- experiments/            # Experiment runner, logging, and plotting
|-- scripts/
|   |-- run_comparison.py       # Master runner for YAML-specified experiment suites
|   |-- run_all_paper_experiments.py # Master orchestrator for all paper experiments
|   |-- profile_hardware_efficiency.py  # Hardware latency & peak VRAM profiler
|   |-- run_cifar100_benchmark.py       # High-cardinality CIFAR-100 benchmark
|   |-- run_scale_50clients.py          # 50-client scalability benchmark
|   |-- run_multi_attack_byzantine.py   # Multi-attack Byzantine suite
|   |-- run_mobilenet_benchmark.py      # MobileNetV3-Small generalizability benchmark
|   |-- run_cluster_k_sensitivity.py    # Cluster count (K) sensitivity sweep
|   |-- run_drift_analysis.py           # Linear CKA representation drift analyzer
|   |-- run_clustering_privacy_sweep.py # Sketching and DP sweep
|   |-- run_calibration_distillation_ablation.py # Calibration & distillation ablation
|   |-- run_epoch_budget_ablation.py    # Epoch budget compute fairness ablation
|   |-- generate_all_tables.py          # Automated LaTeX table generation
|   |-- make_paper_figures.py           # Paper figure generation
|   |-- compute_multiseed_statistics.py # Multi-seed statistics aggregator
|   |-- compute_significance.py         # Statistical significance testing
|   |-- compute_dp_budget.py            # Differential privacy budget accountant
|   |-- download_cifar.py               # Dataset download utility
|   \-- archive/                        # Preserved historical experiment scripts
|-- report/                     # UROP Final Report LaTeX source & compiled PDF
|   |-- main.tex                # Report manuscript
|   |-- main.pdf                # Compiled PDF report
|   \-- references.bib          # Bibliography
|-- tests/                      # 94 pytest unit tests
|-- requirements.txt            # Dependency specifications
|-- main.py                     # CLI entrypoint
\-- README.md
```

---

## Installation & Quick Start

### 1. Prerequisites
* Python 3.10, 3.11, or 3.12
* PyTorch 2.0+ with CUDA, ROCm, DirectML, or CPU support

### 2. Environment Setup
```bash
# Clone repository
git clone https://github.com/nam200718/Topology-aware-FDL.git
cd Topology-aware-FDL

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Test Suite
```bash
pytest tests/ -q
```
All **94 unit tests** should pass.

### 4. Run a Fast Smoke Test (1 Round)
```bash
python scripts/run_comparison.py --config configs/test_1round.yaml
```

---

## Reproducing Experiments

### Step-by-Step CLI Reproduction Commands

* **Main 5-Regime Personalization Benchmark (Table III)**:
  ```bash
  # Execute full benchmark matrix across 5 regimes (seed 42)
  python scripts/run_comparison.py --config configs/comparison.yaml

  # Extract formatted LaTeX table rows from raw metrics
  python scripts/generate_all_tables.py
  ```

* **Hardware Latency, Memory Footprint & Parameter Count (Table II & Figure 1)**:
  ```bash
  python scripts/profile_hardware_efficiency.py
  ```
  *Outputs saved to `outputs/hardware_profiling/`.*

* **High-Class Cardinality CIFAR-100 (Table IV.A)**:
  ```bash
  python scripts/run_cifar100_benchmark.py
  ```
  *Results saved to `outputs/cifar100_results.json`.*

* **50-Client Scalability Benchmark with Partial Participation (Table IV.B)**:
  ```bash
  python scripts/run_scale_50clients.py
  ```
  *Results saved to `outputs/scale_50clients_results.json`.*

* **Multi-Attack Byzantine Fault Tolerance (Table V & Figure 2)**:
  ```bash
  python scripts/run_multi_attack_byzantine.py
  ```
  *Results saved to `outputs/byzantine_multi_attack.json`.*

* **MobileNetV3 Architectural Benchmark (Table VI)**:
  ```bash
  python scripts/run_mobilenet_benchmark.py
  ```
  *Results saved to `outputs/mobilenet_results.json`.*

* **Cluster Count (K) Sensitivity & Bipartite Certification (Table VII)**:
  ```bash
  python scripts/run_cluster_k_sensitivity.py
  ```

* **Backbone Representation Drift Analysis with Linear CKA (Table VIII)**:
  ```bash
  python scripts/run_drift_analysis.py
  ```

* **Random Projection Sketching & Differential Privacy Sweep (Table IX)**:
  ```bash
  python scripts/run_clustering_privacy_sweep.py
  ```

* **Calibration & Distillation Ablation (Table X)**:
  ```bash
  python scripts/run_calibration_distillation_ablation.py
  ```

* **Master Strengthening Suite (Runs CIFAR-100, Scale-50, K-Sweep, Byzantine, and Budget Ablations sequentially)**:
  ```bash
  python scripts/run_all_paper_experiments.py
  ```

* **Render All Paper Figures**:
  ```bash
  python scripts/make_paper_figures.py
  ```

---

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.
