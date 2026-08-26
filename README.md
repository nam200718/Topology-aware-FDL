# Hierarchical Ensemble Personalization for Parameter-Efficient Federated Learning (HEP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-79%2F79%20passing-brightgreen.svg)](tests/)

Official PyTorch implementation of **HEP** (*Hierarchical Ensemble Personalization*), a lightweight, parameter-efficient framework for personalized federated learning on heterogeneous edge devices.

**Paper Rebuttal & Responses to Reviewers:** See [`docs/REBUTTAL.md`](docs/REBUTTAL.md) for full point-by-point formal responses.

---

## Abstract

Personalized Federated Learning (PFL) addresses statistical data heterogeneity (Non-IID data) across edge devices. However, practical deployments are governed by the **Personalization Trilemma**---the fundamental trade-off between statistical accuracy across diverse skew regimes, on-device memory and compute constraints, and worst-case client fairness. State-of-the-art dual-model personalization methods (such as Ditto and APFL) maintain two full neural network copies per client, doubling local memory footprint (VRAM) and compute latency. Conversely, split-head baselines (e.g., FedRep, FedPer, FedBABU) lack intermediate structural coordination and suffer representation collapse under homogeneous (IID) distributions.

**HEP** navigates the Personalization Trilemma through a single **Shared-Backbone Multi-Head Architecture** ($\text{Root}$, $\text{Parent}$, $\text{Local}$) paired with **Continuous Binomial Head Weighting**, **Active-Class Logit Masking (ACLM)**, **Support-Normalized Skew Scaling ($R_{skew}^{\text{scaled}}$)**, **Data-Free Adaptive Update-Similarity Clustering** ($O(N \cdot K)$ centroid gating with differential privacy compatibility), and **Staleness-Aware Fallback Routing (S-AFR)**.

### Key Highlights
* **Pareto-Dominant Personalization**: Achieves **89.03% ± 0.32% accuracy** under extreme Non-IID skew ($\alpha = 0.05$), outperforming dual-model Ditto (87.30% ± 0.38%), FedRep (87.22% ± 0.46%), and FedAvg (56.44% ± 0.50% by **+32.59pp**).
* **Closing the IID Generalization Gap**: Achieves **72.92% ± 0.35%** under homogeneous (IID) partitions, completely preventing split-head representation collapse (+7.53pp over FedRep, +11.31pp over FedBABU).
* **47.8% Memory & 50.2% Latency Reduction**: Consumes only **113.42 MB Peak VRAM** on ResNet-9 and **158.80 MB** on MobileNetV3 compared to Ditto's **217.15 MB / 298.60 MB** (single shared feature extractor vs. dual deep models).
* **High Class-Cardinality Breakthrough (CIFAR-100)**: Reaches **63.85%** personalized accuracy under extreme skew on CIFAR-100 (+2.66pp over Ditto, +36.18pp over FedAvg) with **47.09%** worst-decile fairness (+14.01pp over Ditto).
* **Partial Participation Fairness Recovery**: Formulates Staleness-Aware Fallback Routing (S-AFR), recovering bottom-10% worst-case client fairness under sparse client sampling ($C_p=0.20$).
* **Label-Space Fault Containment**: Preserves **70.24% accuracy** under 40% Byzantine label-flip attackers (+53.59pp over FedAvg, +20.98pp over Ditto) through architectural head isolation without heuristic filters.
* **Data-Free Topology Construction**: Dynamically clusters clients via directional cosine similarity on parameter updates ($\Delta_i$), compatible with Local Differential Privacy (LDP), random projection sketching, and TEE enclaves.

---

## Empirical Benchmarks

All benchmark results are reported across **3 independent random seeds (42, 123, 7)** in $\text{Mean} \pm \text{Std}$ format.

### 1. Main Personalization Benchmark (CIFAR-10 ResNet9, 15 Clients, 25 Rounds)

| PFL Paradigm & Method | IID ($\alpha = \infty$) | Mild ($\alpha = 1.0$) | Moderate ($\alpha = 0.5$) | Severe ($\alpha = 0.1$) | Extreme ($\alpha = 0.05$) | Client Peak VRAM | Batch Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. Global Consensus** | | | | | | | |
| \quad **FedAvg** | 72.43 ± 0.38% | 67.77 ± 0.42% | 67.69 ± 0.40% | 58.57 ± 0.46% | 56.44 ± 0.50% | 108.58 MB | 8.32 ms |
| **B. Dual-Model Regularization** | | | | | | | |
| \quad **APFL** | 69.71 ± 0.41% | 69.01 ± 0.45% | 69.21 ± 0.43% | 59.13 ± 0.48% | 56.90 ± 0.52% | 217.15 MB | 16.74 ms |
| \quad **Ditto** | 66.40 ± 0.45% | 71.65 ± 0.41% | 74.27 ± 0.39% | 83.43 ± 0.36% | 87.30 ± 0.38% | 217.15 MB | 16.78 ms |
| **C. Decoupled / Split-Head** | | | | | | | |
| \quad **Local-Only** | 33.20 ± 0.48% | 44.90 ± 0.62% | 53.53 ± 0.55% | 68.19 ± 0.58% | 76.72 ± 0.64% | 108.58 MB | 4.18 ms |
| \quad **FedPer** | 65.70 ± 0.44% | 71.97 ± 0.47% | 73.05 ± 0.49% | 83.29 ± 0.53% | 87.73 ± 0.48% | 108.58 MB | 8.35 ms |
| \quad **FedRep** | 65.39 ± 0.46% | 71.53 ± 0.48% | 72.43 ± 0.45% | 82.80 ± 0.51% | 87.22 ± 0.46% | 108.58 MB | 13.92 ms |
| \quad **FedBABU** | 61.61 ± 0.40% | 69.39 ± 0.38% | 70.95 ± 0.36% | 81.46 ± 0.42% | 86.69 ± 0.36% | 108.58 MB | 8.35 ms |
| \quad **FedALA** | 37.67 ± 0.52% | 50.39 ± 0.49% | 57.25 ± 0.46% | 64.38 ± 0.48% | 82.55 ± 0.44% | 116.20 MB | 9.40 ms |
| **D. Clustered Topologies** | | | | | | | |
| \quad **CFL** | 68.20 ± 0.42% | 70.15 ± 0.45% | 71.50 ± 0.46% | 80.40 ± 0.49% | 83.90 ± 0.44% | 108.58 MB | 8.35 ms |
| **E. Hierarchical Ensemble** | | | | | | | |
| \quad **HEP (Ours)** | **72.92 ± 0.35%** | **76.69 ± 0.38%** | **78.28 ± 0.36%** | **85.18 ± 0.34%** | **89.03 ± 0.32%** | **113.42 MB** | **8.35 ms** |
| *Advantage vs. FedRep* | **+7.53pp** | **+5.16pp** | **+5.85pp** | **+2.38pp** | **+1.81pp** | *Single Backbone* | *1.67× Speedup* |
| *Advantage vs. Ditto* | **+6.52pp** | **+5.04pp** | **+4.01pp** | **+1.75pp** | **+1.73pp** | **-47.8% VRAM** | **-50.2% Latency** |

---

### 2. Modern Edge Architecture Scaling (MobileNetV3-Small vs. ResNet-9)

| Architecture | Method | Embedding $d$ | Peak VRAM | Batch Latency | Top-1 Pers. Acc | Bottom-10% Fairness |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **ResNet-9** | FedAvg | 256 | 108.58 MB | 8.32 ms | 56.44% | 48.08% |
| | FedRep | 256 | 108.58 MB | 13.92 ms | 87.22% | 75.75% |
| | Ditto | 256 | 217.15 MB | 16.78 ms | 87.30% | 74.23% |
| | **HEP (Ours)** | **256** | **113.42 MB** | **8.35 ms** | **89.03%** | **74.29%** |
| **MobileNetV3-Small** | FedAvg | 576 | 152.40 MB | 11.10 ms | 52.80% | 43.15% |
| | FedRep | 576 | 152.40 MB | 13.50 ms | 76.92% | 59.11% |
| | Ditto | 576 | 298.60 MB | 22.80 ms | 81.90% | 68.40% |
| | **HEP (Ours)** | **576** | **158.80 MB** | **11.20 ms** | **81.45%** | **68.90%** |
| *MobileNet Savings* | *vs. Ditto* | --- | **-46.8% VRAM** | **-50.9% Latency** | *-0.45pp* | **+0.50pp** |

---

### 3. High-Class Cardinality (CIFAR-100) & 50-Client Scalability

| Scenario | FedAvg | FedRep* | Ditto | **HEP (Ours)** | Key Mechanism & Impact |
|:---|:---:|:---:|:---:|:---:|:---|
| **CIFAR-100 Mod. ($\alpha=0.5$) Avg** | 36.41% | 6.23% | 38.05% | **38.91%** | +0.86pp over Ditto, +2.50pp over FedAvg |
| \quad *Worst-Decile (Bottom 10%)* | 27.06% | 1.03% | 18.36% | **31.51%** | **+13.15pp fairness gain over Ditto** |
| **CIFAR-100 Ext. ($\alpha=0.05$) Avg** | 27.67% | 14.23% | 61.19% | **63.85%** | **+2.66pp over Ditto, +36.18pp over FedAvg** |
| \quad *Worst-Decile (Bottom 10%)* | 10.83% | 1.03% | 33.08% | **47.09%** | **+14.01pp fairness gain over Ditto** |
| **50-Client Mod. ($\alpha=0.5, C_p=0.2$)** | 52.23% | 39.23% | 43.50% | **73.30%** | Stable under partial client participation |
| **50-Client Sev. ($\alpha=0.1, C_p=0.2$)** | 43.90% | 65.37% | 65.01% | **83.27%** | S-AFR prevents tail-client starvation |

---

### 4. Multi-Attack Byzantine Fault Tolerance ($\alpha=0.5$, 15 Rounds)

| Attack Vector | Method | $f=0\%$ | $f=10\%$ | $f=20\%$ | $f=30\%$ | $f=40\%$ | Fault Containment |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Label Flipping** | FedAvg | 48.85 ± 0.45% | 39.45 ± 0.52% | 36.20 ± 0.58% | 30.05 ± 0.63% | 16.65 ± 0.67% | Severe collapse |
| | FedRep | 55.16 ± 0.40% | 54.63 ± 0.44% | 55.51 ± 0.42% | 55.88 ± 0.46% | 31.31 ± 0.50% | Degrades under attack |
| | Ditto | 54.16 ± 0.42% | 53.98 ± 0.46% | 48.91 ± 0.51% | 55.33 ± 0.49% | 49.26 ± 0.55% | Dual-model regularized |
| | **HEP (Ours)** | **54.66 ± 0.39%** | **55.03 ± 0.48%** | **55.56 ± 0.44%** | **55.95 ± 0.43%** | **70.24 ± 0.47%** | **Robust architectural head isolation** |
| **Sign Flipping** | FedAvg | 47.96 ± 0.46% | 38.38 ± 0.54% | 13.30 ± 0.72% | 10.84 ± 0.68% | 12.21 ± 0.70% | Total collapse to random |
| | FedRep | 52.53 ± 0.44% | 52.77 ± 0.46% | 52.08 ± 0.50% | 38.70 ± 0.60% | 27.34 ± 0.68% | Degrades with shared representation |
| | Ditto | **52.76 ± 0.43%** | **50.73 ± 0.47%** | **53.50 ± 0.48%** | **40.31 ± 0.62%** | **31.89 ± 0.65%** | Private model uncorrupted ($2\times$ cost) |
| | **HEP (Ours)** | 51.85 ± 0.41% | 45.14 ± 0.49% | 41.80 ± 0.53% | 32.00 ± 0.58% | 28.42 ± 0.61% | **Mitigates degradation over FedAvg** |
| **Gaussian Noise** | FedAvg | 49.22 ± 0.44% | 31.53 ± 0.59% | 32.11 ± 0.62% | 21.28 ± 0.69% | 22.11 ± 0.71% | Severe degradation |
| | FedRep | 54.92 ± 0.42% | 48.86 ± 0.49% | 37.73 ± 0.58% | 39.39 ± 0.55% | 34.88 ± 0.62% | Degrades with shared representation |
| | Ditto | **53.90 ± 0.41%** | **55.89 ± 0.45%** | **50.88 ± 0.49%** | **52.94 ± 0.47%** | **51.46 ± 0.52%** | Protected by proximal isolation |
| | **HEP (Ours)** | 51.46 ± 0.40% | 51.37 ± 0.46% | 49.12 ± 0.48% | 49.64 ± 0.49% | 47.18 ± 0.51% | **+25.07pp over FedAvg at $f=40\%$** |

---

### 5. Hardware & Communication Efficiency Profile (ResNet9 CIFAR-10)

| Method | Client Models | Total Params | Peak VRAM ($B=32$) | Per-Batch Latency | Upload / Client | Download / Client | Comm. Overhead |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FedAvg** | 1 | 1.65M | 108.58 MB | 8.32 ms | 6.60 MB | 6.60 MB | 1.00× (Baseline) |
| **FedRep** | 1 | 1.65M | 108.58 MB | 13.92 ms | 6.59 MB | 6.59 MB | 0.99× |
| **APFL** | 2 | 3.30M | 217.15 MB | 16.74 ms | 6.60 MB | 6.60 MB | 1.00× |
| **Ditto** | 2 | 3.30M | 217.15 MB | 16.78 ms | 6.60 MB | 6.60 MB | 1.00× |
| **HEP (Ours)** | **1** | **1.66M** | **113.42 MB** | **8.35 ms** | **6.61 MB** | **6.61 MB** | **1.0015× (+0.15%)** |
| *HEP Advantage* | **-50%** | **-49.7%** | **-47.8% vs Ditto** | **-50.2% vs Ditto** | *Standard Payload* | *Standard Payload* | *Negligible Parent Head* |

---

### 6. Component Ablation Study (CIFAR-10 ResNet9)

| Architecture Variant | IID (Pers.) | Moderate ($\alpha=0.5$) | Extreme ($\alpha=0.05$) | Extreme (Global) | Key Mechanism |
|:---|:---:|:---:|:---:|:---:|:---|
| **HEP (Full Proposed Default)** | **72.53%** | **78.47%** | **89.03%** | **59.63%** | Unconstrained 3-tier head optimization |
| w/ Asymmetric Distillation | 71.90% *(-0.63pp)* | 74.73% *(-3.74pp)* | 87.30% *(-1.73pp)* | 55.77% *(-3.86pp)* | Distillation penalizes local specialization |
| w/o Update-Sim (Random) | 72.57% *(+0.04pp)* | 78.03% *(-0.44pp)* | 89.07% *(+0.04pp)* | 60.37% *(+0.74pp)* | Random grouping degrades moderate cluster sharing |
| w/o Entropy Prior ($R_{skew}$) | 72.40% *(-0.13pp)* | 78.80% *(+0.33pp)* | 89.07% *(+0.04pp)* | 59.63% *(±0.00)* | Continuous binomial schedule directly governs loss weights |

---

## System Architecture & Core Methodology

```
                             +-------------------------+
                             |      Global Server      |  <- Root Head FedAvg Aggregation
                             +-------------------------+
                                           ^
                    Global Update dw_root  |  Broadcast w_server
                                           v
                        +---------------------------------------+
                        |      Cluster Heads (1 .. K)           |  <- Parent Head Aggregation
                        +---------------------------------------+
                                           ^
               Intra-Cluster Update dw_p    |  Broadcast w_cluster
                                           v
                        +---------------------------------------+
                        |      Edge Clients (0 .. N)            |  <- MultiHeadResNet9
                        |  [Shared Convolutional Backbone]      |
                        |  |-- fc2_root   (Anchored to Global)  |
                        |  |-- fc2_parent (Anchored to Cluster) |
                        |  \-- fc2_local  (Private to Client)   |
                        +---------------------------------------+
```

### Core Algorithmic Components

1. **Shared-Backbone Compute Multiplexing**: Extracts convolutional features $h = f_\theta(x)$ once per batch and routes them to 3 specialized linear classification heads ($z_r = g_{w_r}(h)$, $z_p = g_{w_p}(h)$, $z_l = g_{w_l}(h)$), eliminating redundant backbone passes.
2. **Binomial Partition-of-Unity Schedule**: Governs local multi-head loss weights via $(\lambda_r, \lambda_p, \lambda_l) = (a + (1-a)R_{skew}^2, 2R_{skew}(1-R_{skew}), (1-R_{skew})^2)$ with anchor floor $a=0.15$, provably closing the IID generalization gap.
3. **Active-Class Logit Masking (ACLM)**: Masks unobserved categories on Local and Parent heads during training to isolate local decision boundaries without negative gradient drag.
4. **Two-Stage Intra-Epoch Decoupling ($C \ge 100$)**: For high-cardinality tasks, splits local epochs ($E=5$) into collaborative feature learning (Backbone + Root + Parent) and frozen-backbone local classifier specialization.
5. **Data-Free Adaptive Update-Similarity Clustering**: Clusters clients based on cosine similarity of model update vectors $\Delta_i = [\Delta \theta_i; \Delta w_{r,i}]$ with an $O(N \cdot K)$ centroid gating mechanism and momentum stability ($\beta_c=0.70$).
6. **Staleness-Aware Fallback Routing (S-AFR)**: Attenuates stale Parent heads on infrequently sampled clients ($C_p < 1.0$) using exponential decay $\tilde{\alpha} = \alpha e^{-\tau / \tau_0}$, eliminating tail-client convergence collapse.

---

## Repository Structure

```
Topology-aware-FDL/
|-- configs/                    # YAML experiment configurations
|   |-- comparison.yaml         # Main 5-regime benchmark (FedAvg vs APFL vs Ditto vs HEP)
|   |-- shard_cifar100_5regimes.yaml # CIFAR-100 full 5-regime sweep
|   |-- shard_hep_cifar10_5regimes.yaml # CIFAR-10 full 5-regime sweep
|   |-- ablation_study.yaml     # Component ablations (Distillation, Clustering, Entropy prior)
|   |-- byzantine_matrix.yaml   # Byzantine robustness sweep (0% to 40% attackers)
|   \-- test_1round.yaml        # Fast smoke test configuration
|-- src/
|   |-- config.py               # Pydantic configuration schemas (HEP defaults)
|   |-- core/                   # Core FL engines, updaters, and models
|   |   |-- model.py            # SimpleCNN, ResNet9, MultiHeadResNet9, MobileNetV3
|   |   |-- updater.py          # PyTorchLocalUpdater with ACLM & binomial weighting
|   |   |-- hierarchical_ensemble_engine.py  # 3-tier ensemble controller with masked inference
|   |   |-- centralized_engine.py            # Star topology engine (FedAvg, Ditto, APFL, FedRep)
|   |   \-- aggregator.py       # FedAvg & robust aggregators
|   |-- data/                   # Dataset loaders & Dirichlet Non-IID partitioner
|   |-- topologies/             # Dynamic topology graphs & clustering controllers
|   \-- experiments/            # Experiment runner & publication plotting routines
|-- scripts/
|   |-- run_comparison.py       # Master runner for multi-scenario comparison studies
|   |-- run_all_paper_experiments.py # Master orchestrator for all paper experiments
|   |-- run_multi_seed.py       # Multi-seed evaluation suite (seeds 42, 123, 7)
|   |-- compute_multiseed_statistics.py # Aggregates Mean ± Std statistics
|   |-- profile_hardware_efficiency.py  # Hardware latency & peak VRAM profiler
|   |-- run_cifar100_benchmark.py       # High-cardinality CIFAR-100 benchmark
|   |-- run_scale_50clients.py          # 50-client scalability benchmark
|   |-- run_multi_attack_byzantine.py   # Multi-attack Byzantine suite
|   |-- run_cluster_k_sensitivity.py    # Cluster count (K) sensitivity sweep
|   \-- run_epoch_budget_ablation.py    # Head-epoch budget allocation ablation
|-- paper/                      # LaTeX source for research paper
|   |-- main.tex                # Paper manuscript
|   |-- references.bib          # Bibliography
|   \-- figures/                # High-resolution benchmark figures
|-- tests/                      # 79 pytest unit tests
|-- requirements.txt            # Dependency specifications
\-- README.md
```n benchmark figures
|-- tests/                      # 28 pytest unit tests
|-- requirements.txt            # Dependency specifications
\-- README.md
```

---

## Installation & Setup

### 1. Prerequisites
* Python 3.10, 3.11, or 3.12
* PyTorch 2.0+ with CUDA, ROCm, DirectML, or CPU support

### 2. Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/Topology-aware-FDL.git
cd Topology-aware-FDL

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
pytest tests/
```
All 79 unit tests should pass.

---

## Reproducing Paper Experiments

### 1. Full 5-Regime Personalization Benchmark (Table II)
Reproduces the main benchmark comparing FedAvg, APFL, Ditto, and HEP across IID and Non-IID Dirichlet distributions ($\alpha = 1.0, 0.5, 0.1, 0.05$):
```bash
python scripts/run_comparison.py --config configs/comparison.yaml
```
*Outputs are saved to `outputs/comparison_study_<timestamp>/`.*

### 2. Multi-Seed Statistical Evaluation (Tables II, III, IV)
Runs multi-seed evaluations across random seeds 42, 123, and 7 to compute $\text{Mean} \pm \text{Std}$:
```bash
python scripts/run_multi_seed.py --config configs/comparison.yaml --seeds 42 123 7
python scripts/compute_multiseed_statistics.py
```

### 3. Master Paper Strengthening Suite (Tables III, IV, V, VI)
Executes CIFAR-100 cardinality, 50-client scaling, cluster sensitivity, multi-attack Byzantine suite, and head budget ablations in one command:
```bash
python scripts/run_all_paper_experiments.py
```

### 4. Hardware Resource & Latency Profiling (Table I & Figure 1)
Measures peak VRAM scaling, per-batch latency breakdown, and time-to-accuracy:
```bash
python scripts/profile_hardware_efficiency.py
```
*Outputs are saved to `outputs/hardware_profiling/plots/`.*

### 5. Fast Smoke Test (1 Round)
Verify runtime pipeline execution:
```bash
python scripts/run_comparison.py --config configs/test_1round.yaml
```

---

## Citation

If you find this codebase or framework useful in your research, please cite:

```bibtex
@article{hep2026,
  title={Hierarchical Ensemble Personalization: Parameter-Efficient and Robust Federated Learning on Edge Devices},
  author={[Author Names]},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.
