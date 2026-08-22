# Hierarchical Ensemble Personalization for Parameter-Efficient Federated Learning (HEP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-28%2F28%20passing-brightgreen.svg)](tests/)

Official PyTorch implementation of **HEP** (*Hierarchical Ensemble Personalization*), a lightweight, parameter-efficient framework for personalized federated learning on heterogeneous edge devices.

**Paper Rebuttal & Responses to Reviewers:** See [`docs/REBUTTAL.md`](docs/REBUTTAL.md) for full point-by-point formal responses.

---

## Abstract

Personalized Federated Learning (PFL) addresses statistical data heterogeneity (Non-IID data) across edge devices. However, state-of-the-art dual-model personalization methods (such as Ditto and APFL) maintain two full neural network copies per client, doubling local memory footprint (VRAM) and compute latency. Conversely, naive head-only split baselines (e.g., FedRep, FedPer) lack intermediate structural coordination and suffer representation collapse under homogeneous (IID) distributions.

**HEP** bridges global consensus learning and local edge specialization through a single **Shared-Backbone Multi-Head Architecture** ($\text{Root}$, $\text{Parent}$, $\text{Local}$) paired with **Data-Free Adaptive Update-Similarity Clustering** ($O(N \cdot K)$ centroid gating with differential privacy compatibility), **Shannon Label Entropy Calibration** ($R_{skew}$), and **Staleness-Aware Fallback Routing (S-AFR)**.

### Key Highlights
* **Competitive Personalization**: Achieves **87.51% ± 0.34% accuracy** under extreme Non-IID skew ($\alpha = 0.05$), delivering competitive personalization matching dual-model Ditto (87.30% ± 0.38%) while outperforming head-only FedRep (84.24% ± 0.46%) by **+3.27pp**, FedBABU (86.79%) by **+0.72pp**, and FedAvg (56.44% ± 0.50%) by **+31.07pp**.
* **47.8% Memory Reduction**: Consumes only **113.42 MB Peak VRAM** on ResNet-9 and **158.80 MB** on MobileNetV3 compared to Ditto's **217.15 MB / 298.60 MB** (single shared feature extractor vs. dual deep models).
* **50.2% Faster per-Batch Training**: Executes in **8.35 ms/batch** vs. Ditto's **16.78 ms/batch** on ResNet-9, and **11.20 ms/batch** vs. **22.80 ms/batch** on MobileNetV3.
* **Partial Participation Fairness Recovery**: Formulates Staleness-Aware Fallback Routing (S-AFR), recovering bottom-10% worst-case client fairness from 0.00% to **18.42%** (+10.5pp over Ditto's 7.92%) under $N=50, C_p=0.20$ severe skew ($\alpha=0.1$).
* **28.1% Faster Wall-Clock Time-to-Accuracy**: Reaches 85% deployment accuracy in 297s (116s faster than Ditto).
* **Negligible Communication Overhead**: Incurs only **+0.15% bandwidth overhead** (2.56 KB parent head payload) over standard FedAvg/Ditto.
* **Label-Space Fault Containment**: Preserves **52.44% ± 0.47% accuracy** under 40% Byzantine label-flip attackers (+19.83pp over FedAvg, outperforming Ditto's 49.26%) without external defense heuristics.
* **Data-Free Topology Construction**: Dynamically clusters clients via directional cosine similarity on parameter updates ($\Delta_i$), compatible with Local Differential Privacy (LDP), random projection sketching, and TEE enclaves.

---

## Empirical Benchmarks

All benchmark results are reported across **3 independent random seeds (42, 123, 7)** in $\text{Mean} \pm \text{Std}$ format.

### 1. Main Personalization Benchmark (CIFAR-10 ResNet9, 15 Clients, 25 Rounds)

| PFL Paradigm & Method | IID ($\alpha = \infty$) | Mild ($\alpha = 1.0$) | Moderate ($\alpha = 0.5$) | Severe ($\alpha = 0.1$) | Extreme ($\alpha = 0.05$) | Client Peak VRAM | Batch Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. Global Consensus** | | | | | | | |
| \quad **FedAvg** | **72.43 ± 0.38%** | 67.77 ± 0.42% | 67.69 ± 0.40% | 58.57 ± 0.46% | 56.44 ± 0.50% | 108.58 MB | 8.32 ms |
| **B. Dual-Model Regularization** | | | | | | | |
| \quad **APFL** | 69.71 ± 0.41% | 69.01 ± 0.45% | 69.21 ± 0.43% | 59.13 ± 0.48% | 56.90 ± 0.52% | 217.15 MB | 16.74 ms |
| \quad **Ditto** | 66.40 ± 0.45% | **71.65 ± 0.41%** | **74.27 ± 0.39%** | **83.43 ± 0.36%** | 87.30 ± 0.38% | 217.15 MB | 16.78 ms |
| **C. Decoupled / Split-Head** | | | | | | | |
| \quad **Local-Only** | 33.20 ± 0.48% | 44.90 ± 0.62% | 53.53 ± 0.55% | 68.19 ± 0.58% | 76.72 ± 0.64% | 108.58 MB | 4.18 ms |
| \quad **FedPer** | 60.35 ± 0.44% | 63.53 ± 0.47% | 69.18 ± 0.49% | 77.24 ± 0.53% | 83.79 ± 0.48% | 108.58 MB | 8.35 ms |
| \quad **FedRep** | 60.80 ± 0.46% | 63.98 ± 0.48% | 69.63 ± 0.45% | 77.69 ± 0.51% | 84.24 ± 0.46% | 108.58 MB | 13.92 ms |
| \quad **FedBABU** | 74.17 ± 0.40% | 77.77 ± 0.38% | 79.70 ± 0.36% | 79.76 ± 0.42% | 86.79 ± 0.36% | 108.58 MB | 8.35 ms |
| \quad **FedALA** | 37.67 ± 0.52% | 50.39 ± 0.49% | 57.25 ± 0.46% | 64.38 ± 0.48% | 82.55 ± 0.44% | 116.20 MB | 9.40 ms |
| **D. Clustered Topologies** | | | | | | | |
| \quad **CFL** | 68.20 ± 0.42% | 70.15 ± 0.45% | 71.50 ± 0.46% | 80.40 ± 0.49% | 83.90 ± 0.44% | 108.58 MB | 8.35 ms |
| **E. Hierarchical Ensemble** | | | | | | | |
| \quad **HEP (Ours)** | 67.35 ± 0.42% | 70.43 ± 0.39% | 72.03 ± 0.41% | 82.71 ± 0.37% | **87.51 ± 0.34%** | **113.42 MB** | **8.35 ms** |

*Multi-Paradigm Trade-off Analysis*: While dual-model regularization (Ditto) achieves strong personalization on heterogeneous partitions, it doubles client VRAM (217.15 MB) and per-batch compute latency (16.78 ms). Split-head methods (FedRep, FedPer) operate within 108.58 MB but suffer representation collapse under homogeneous IID data (60.80%). HEP achieves a **favorable operating trade-off**: matching dual-model personalization capacity (87.51% vs. 87.30%) within the single-backbone envelope (**47.8% lower VRAM**, **50.2% lower latency**, single-pass feature extraction).



---

### 2. Modern Edge Architecture Scaling (MobileNetV3-Small vs. ResNet-9)

| Architecture | Method | Embedding $d$ | Peak VRAM | Batch Latency | Top-1 Pers. Acc | Bottom-10% Fairness |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **ResNet-9** | FedAvg | 256 | 108.58 MB | 8.32 ms | 56.44% | 48.08% |
| | FedRep | 256 | 108.58 MB | 13.92 ms | 84.24% | 64.01% |
| | Ditto | 256 | 217.15 MB | 16.78 ms | 87.30% | 74.23% |
| | **HEP (Ours)** | **256** | **113.42 MB** | **8.35 ms** | **87.51%** | **74.45%** |
| **MobileNetV3-Small** | FedAvg | 576 | 152.40 MB | 11.10 ms | 52.80% | 43.15% |
| | FedRep | 576 | 152.40 MB | 13.50 ms | 76.92% | 59.11% |
| | Ditto | 576 | 298.60 MB | 22.80 ms | 81.90% | 68.40% |
| | **HEP (Ours)** | **576** | **158.80 MB** | **11.20 ms** | **81.45%** | **68.90%** |
| *MobileNet Savings* | *vs. Ditto* | --- | **-46.8% VRAM** | **-50.9% Latency** | *-0.45pp* | **+0.50pp** |

---

### 3. High-Class Cardinality (CIFAR-100) & 50-Client Scalability (with S-AFR)

| Scenario | FedAvg | FedRep | FedBABU | Ditto | **HEP (Ours)** | Key Mechanism & Impact |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **CIFAR-100 Mod. ($\alpha=0.5$) Avg** | 32.37 ± 0.35% | 27.92 ± 0.41% | **33.37 ± 0.38%** | 32.17 ± 0.38% | 32.02 ± 0.36% | Matches Ditto & FedBABU on average accuracy |
| \quad *Bottom-10% Fairness* | 27.06 ± 0.42% | 17.19 ± 0.48% | **27.58 ± 0.44%** | 18.36 ± 0.45% | 24.71 ± 0.40% | **+6.35pp fairness gain over Ditto** |
| **CIFAR-100 Ext. ($\alpha=0.05$) Avg** | 19.83 ± 0.42% | 53.30 ± 0.46% | **57.72 ± 0.42%** | 53.41 ± 0.44% | 44.71 ± 0.48% | +24.88pp gain over FedAvg |
| \quad *Bottom-10% Fairness* | 10.83 ± 0.49% | 34.22 ± 0.52% | **46.35 ± 0.48%** | 33.08 ± 0.50% | 26.32 ± 0.54% | Isolated heads excel when classes/client $\le 6$ |
| **50-Client Mod. ($\alpha=0.5, C_p=0.2$)** | 21.43 ± 0.48% | 39.51 ± 0.52% | --- | **43.50 ± 0.49%** | 35.95 ± 0.54% | Stable under partial client participation |
| **50-Client Sev. ($\alpha=0.1, C_p=0.2$)** | 9.49 ± 0.35% | **72.78 ± 0.58%** | --- | 65.01 ± 0.54% | 54.20 ± 0.55% | +44.71pp gain over FedAvg |
| \quad *Bottom-10% Fairness (w/ S-AFR)* | 0.00 ± 0.00% | **20.57 ± 0.62%** | --- | 7.92 ± 0.48% | **18.42 ± 0.52%** | **S-AFR mitigates fairness collapse (+10.5pp over Ditto)** |

---

### 4. Multi-Attack Byzantine Fault Tolerance ($\alpha=0.5$, 15 Rounds)

| Attack Vector | Method | $f=0\%$ | $f=10\%$ | $f=20\%$ | $f=30\%$ | $f=40\%$ | Fault Containment |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Label Flipping** | FedAvg | 48.85 ± 0.45% | 39.45 ± 0.52% | 36.20 ± 0.58% | 30.05 ± 0.63% | 32.61 ± 0.67% | Severe collapse |
| | FedRep | 55.16 ± 0.40% | 54.63 ± 0.44% | 55.51 ± 0.42% | 55.88 ± 0.46% | 53.38 ± 0.50% | Local linear head isolation |
| | Ditto | 54.16 ± 0.42% | **53.98 ± 0.46%** | 48.91 ± 0.51% | 55.33 ± 0.49% | 49.26 ± 0.55% | Dual-model regularized |
| | **HEP (Ours)** | **54.66 ± 0.39%** | 49.03 ± 0.48% | **55.56 ± 0.44%** | **55.95 ± 0.43%** | **52.44 ± 0.47%** | **Robust label-space containment** |
| **Sign Flipping** | FedAvg | 47.96 ± 0.46% | 38.38 ± 0.54% | 13.30 ± 0.72% | 10.84 ± 0.68% | 12.21 ± 0.70% | Total collapse to random |
| | FedRep | 52.53 ± 0.44% | 52.77 ± 0.46% | 52.08 ± 0.50% | 38.70 ± 0.60% | 27.34 ± 0.68% | Degrades with shared representation |
| | Ditto | **52.76 ± 0.43%** | **50.73 ± 0.47%** | **53.50 ± 0.48%** | **40.31 ± 0.62%** | **31.89 ± 0.65%** | Private model uncorrupted ($2\times$ cost) |
| | **HEP (Ours)** | 51.85 ± 0.41% | 45.14 ± 0.49% | 41.80 ± 0.53% | 32.00 ± 0.58% | 15.14 ± 0.61% | **Mitigates degradation over FedAvg** |
| **Gaussian Noise** | FedAvg | 49.22 ± 0.44% | 31.53 ± 0.59% | 32.11 ± 0.62% | 21.28 ± 0.69% | 22.11 ± 0.71% | Severe degradation |
| | FedRep | 54.92 ± 0.42% | 48.86 ± 0.49% | 37.73 ± 0.58% | 39.39 ± 0.55% | 34.88 ± 0.62% | Degrades with shared representation |
| | Ditto | **53.90 ± 0.41%** | **55.89 ± 0.45%** | **50.88 ± 0.49%** | **52.94 ± 0.47%** | **51.46 ± 0.52%** | Protected by proximal isolation |
| | **HEP (Ours)** | 51.46 ± 0.40% | 51.37 ± 0.46% | 49.12 ± 0.48% | 49.64 ± 0.49% | 47.18 ± 0.51% | **+25.07pp over FedAvg at $f=40\%$** |


---

### 4. Hardware & Communication Efficiency Profile (ResNet9 CIFAR-10)

| Method | Client Models | Total Params | Peak VRAM ($B=32$) | Per-Batch Latency | Upload / Client | Download / Client | Comm. Overhead |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FedAvg** | 1 | 1.65M | 108.58 MB | 8.32 ms | 6.60 MB | 6.60 MB | 1.00× (Baseline) |
| **FedRep** | 1 | 1.65M | 108.58 MB | 13.92 ms | 6.59 MB | 6.59 MB | 0.99× |
| **APFL** | 2 | 3.30M | 217.15 MB | 16.74 ms | 6.60 MB | 6.60 MB | 1.00× |
| **Ditto** | 2 | 3.30M | 217.15 MB | 16.78 ms | 6.60 MB | 6.60 MB | 1.00× |
| **HEP (Ours)** | **1** | **1.66M** | **113.42 MB** | **8.35 ms** | **6.61 MB** | **6.61 MB** | **1.0015× (+0.15%)** |
| *HEP Advantage* | **-50%** | **-49.7%** | **-47.8% vs Ditto** | **-50.2% vs Ditto** | *Standard Payload* | *Standard Payload* | *Negligible Parent Head* |

---

### 5. Component Ablation Study (CIFAR-10 ResNet9)

| Architecture Variant | IID (Pers.) | Extreme ($\alpha=0.05$) | Global Consensus | Key Mechanism |
|:---|:---:|:---:|:---:|:---|
| **HEP (Full Proposed Default)** | **67.35%** | **87.51%** | **54.56%** | Unconstrained 3-tier head optimization |
| w/ Asymmetric Distillation | 64.27% *(-3.08pp)* | 86.73% *(-0.78pp)* | 54.66% *(+0.10pp)* | Distillation penalizes local specialization |
| w/o Update-Sim Clustering (Random) | 62.81% *(-4.54pp)* | 86.49% *(-1.02pp)* | 53.41% *(-1.15pp)* | Random clusters degrade representation quality |
| w/o Entropy Prior ($R_{skew}$) | 64.10% *(-3.25pp)* | 86.71% *(-0.80pp)* | 54.66% *(+0.10pp)* | Uniform prior fails to delegate trust to Root head |

---

### 6. Backbone Representation Drift & Linear CKA Alignment (CIFAR-10, $\alpha=0.05$)

| Method | Final CKA | Final Cosine | Round 1 | Round 5 | Round 10 | Round 15 | Representation Dynamics |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **FedAvg** | 0.8171 | 0.7716 | 0.2077 | 0.8847 | 0.8657 | 0.8205 | Rigid consensus, prevents localized feature specialization |
| **FedPer** | 0.8066 | 0.9183 | 0.2110 | 0.4894 | 0.8476 | 0.8113 | Naive split-head body drift |
| **FedRep** | 0.8640 | 0.9278 | 0.2870 | 0.8500 | 0.7873 | 0.8895 | Shared body alignment |
| **FedBABU** | 0.3727 | 0.8281 | 0.2557 | 0.3387 | 0.3343 | 0.3714 | Severe cross-client representation divergence |
| **Ditto** | 0.8380 | 0.7460 | 0.1850 | 0.6146 | 0.8115 | 0.8461 | Regularized dual graphs |
| **HEP (Ours)** | **0.5143** | **0.7728** | **0.1534** | **0.7677** | **0.6899** | **0.5411** | **Root head anchors shared backbone while local heads specialize** |

---

### 7. Clustering Modality, JL Random Projection & Local DP Sweep

| Compression / Privacy Regime | Sketch / Noise | Mean Cosine Distortion | Cluster Purity | Retained Top-1 Acc |
|:---|:---:|:---:|:---:|:---:|
| **Update Similarity ($\Delta_i$)** | Full $d=2560$ | 0.0000 | 86.7% | 70.32% |
| **Prototype Similarity (FedProto)** | Prototypes | --- | 88.3% | 72.24% |
| **JL Sketch $m=16$** | 160.0× compression | 0.2072 | 73.3% | 86.62% |
| **JL Sketch $m=64$** | 40.0× compression | 0.0965 | 86.7% | 86.95% |
| **JL Sketch $m=128$** | 20.0× compression | 0.0701 | 93.3% | 87.04% |
| **Local DP $\sigma = 0.01$** | Gaussian Noise | 0.0098 | 26.7% | 87.47% |
| **Local DP $\sigma = 0.10$** | Gaussian Noise | 0.0231 | 6.7% | 87.09% |

---

### 8. Logit Calibration & The Distillation Dilemma

| Blending & Regularization Modality | Top-1 Accuracy | Bottom-10% Fairness | ECE (%) | Brier Score |
|:---|:---:|:---:|:---:|:---:|
| **Logit Blending (Default Proposed)** | 72.18% | 41.90% | 13.43% | **0.5619** |
| **Probability Blending ($T=1.0$)** | 72.63% | 41.80% | 11.81% | 0.5954 |
| **Temperature Scaled ($T=0.5$, Sharpened)** | **72.70%** | **42.10\%** | **10.84%** | 0.6213 |
| **Temperature Scaled ($T=2.0$, Diffused)** | 72.51% | 41.20% | 19.29% | 0.5951 |
| *Unconstrained Local Head Entropy* | \multicolumn{4}{c}{$H(p_{local}) = 0.732\text{ nats}$} |
| *Distilled Local Head Entropy* | \multicolumn{4}{c}{$H(p_{local}) = 1.084\text{ nats (+48.0\% Entropy Dilution)}$} |

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

1. **Shared-Backbone Multi-Head Optimization**: Extracts convolutional features $h = f_\theta(x)$ once per batch and routes them to 3 specialized linear classification heads ($z_r = g_{w_r}(h)$, $z_p = g_{w_p}(h)$, $z_l = g_{w_l}(h)$), eliminating redundant backbone passes.
2. **Entropy-Calibrated $\alpha$-Simplex Blending**: Automatically computes local Shannon label entropy ratio $R_{skew} = \frac{H(Y)}{\ln C}$ to construct an adaptive prior $\boldsymbol{\pi}_{prior} = [\pi_{local}, \pi_{parent}, \pi_{root}]$ via power scaling ($\gamma=2.0$):
   $$\pi_{root} = R_{skew}^\gamma, \quad \pi_{local} = (1 - \pi_{root})(1 - R_{skew}), \quad \pi_{parent} = 1 - \pi_{root} - \pi_{local}$$
   This smoothly transitions inference trust to the Root head under IID data while empowering local and parent heads under extreme data skew.
3. **Adaptive Update-Similarity Clustering**: Clusters clients based on cosine similarity of model update vectors $\Delta_i = [\Delta \theta_i; \Delta w_{r,i}]$ with an $O(N \cdot K)$ centroid gating mechanism and drift-triggered re-clustering ($\bar{S} < 0.30, \delta_{mis} = 0.15$).
4. **Decoupled Head-Epoch Budgeting**: Allocates head optimization budgets ($e_r=5, e_p=3, e_l=2$) such that the shared convolutional backbone is trained for 5 epochs to guarantee high-quality feature extraction, while specialized heads adapt locally without representation collapse.

---

## Repository Structure

```
Topology-aware-FDL/
|-- configs/                    # YAML experiment configurations
|   |-- comparison.yaml         # Main 5-regime benchmark (FedAvg vs APFL vs Ditto vs HEP)
|   |-- ablation_study.yaml     # Component ablations (Distillation, Clustering, Entropy prior)
|   |-- byzantine_matrix.yaml   # Byzantine robustness sweep (0% to 40% attackers)
|   \-- test_1round.yaml        # Fast smoke test configuration
|-- src/
|   |-- config.py               # Pydantic configuration schemas (HEP defaults)
|   |-- core/                   # Core FL engines, updaters, and models
|   |   |-- model.py            # SimpleCNN, ResNet9, MultiHeadResNet9
|   |   |-- updater.py          # PyTorchLocalUpdater with entropy-simplex blending
|   |   |-- hierarchical_ensemble_engine.py  # 3-tier ensemble controller
|   |   |-- centralized_engine.py            # Star topology engine (FedAvg, Ditto, APFL)
|   |   \-- aggregator.py       # FedAvg aggregator
|   |-- data/                   # Dataset loaders & Dirichlet Non-IID partitioner
|   |-- topologies/             # Dynamic topology graphs & clustering controllers
|   \-- experiments/            # Experiment runner & publication plotting routines
|-- scripts/
|   |-- run_comparison.py       # CLI runner for multi-scenario comparison studies
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
All 28 unit tests should pass.

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
