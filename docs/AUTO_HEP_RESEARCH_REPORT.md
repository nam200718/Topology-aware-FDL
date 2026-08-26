# An Architectural Evaluation and Systems Analysis of Auto-HEP in Edge Federated Learning

**Document Type:** Academic Research Report  
**Subject:** Federated Learning, Parameter-Efficient Edge Optimization, Multi-Head Architectures  
**Code Repository Reference:** `Topology-aware-FDL`  

---

## Abstract

Personalized Federated Learning (PFL) seeks to address data heterogeneity across decentralized edge devices while operating within constrained on-device hardware budgets. This report provides an architectural evaluation and systems analysis of the **Auto-HEP** (*Automated Hierarchical Ensemble Personalization*) framework. 

Auto-HEP implements a **shared-backbone compute multiplexing** design that couples a single convolutional feature extractor with multiple lightweight linear classification heads. In measured benchmarks on CIFAR-10 ResNet-9, this design achieves a peak VRAM allocation of **113.42 MB** (a 47.8% reduction compared to the 217.15 MB required by dual-model Ditto) and an average per-batch training latency of **8.35 ms** (a 50.2% improvement over Ditto's 16.78 ms). 

This report examines the implemented algorithmic components—including the entropy-based label skew metric ($R_{skew}$), binomial loss scheduling, and Active-Class Logit Masking (ACLM)—and evaluates the empirical evidence regarding the system's 3-tier hierarchy. Analysis of the published ablation data indicates that a streamlined **2-Tier Bipartite architecture** (Global Root Head + Private Local Head) performs within $\pm 0.44\text{ pp}$ of the 3-tier system across all evaluated heterogeneity regimes, while eliminating sketch transmission, server-side cluster tracking, and intermediate routing noise. Additionally, we analyze observed trade-offs in worst-case client fairness, where split-head FedRep retains higher bottom-10% accuracy under extreme skew (76.85% vs. Auto-HEP's 74.29%), and discuss the privacy boundaries of the communication pipeline.

---

## 1. Introduction & Problem Setting

### 1.1 Federated Learning under Statistical Skew
In Federated Learning (FL), $N$ edge clients collaboratively train machine learning models using local data partitions $\{\mathcal{D}_i\}_{i=1}^N$ coordinated by a central server. In edge scenarios, local datasets often exhibit statistical label heterogeneity (Non-IID data), commonly parameterized via a symmetric Dirichlet distribution $\text{Dir}(\alpha)$. Lower values of $\alpha$ (e.g., $\alpha=0.05$) result in severe label imbalance, where clients observe only a small subset of total classes, while $\alpha \to \infty$ corresponds to uniform (IID) sampling.

### 1.2 The Personalization Trilemma
Deploying federated models on edge devices involves managing trade-offs across three primary constraints:
1. **Accuracy Across Skew Regimes:** Maintaining stable performance across uniform (IID) data, moderate skew, and extreme skew ($\alpha=0.05$).
2. **On-Device Hardware Footprint:** Restricting peak VRAM usage and per-batch computation passes to accommodate memory-limited hardware (such as edge GPUs and microcontrollers).
3. **Worst-Case Client Fairness:** Ensuring that tail-decile clients (the bottom 10% of clients by accuracy) achieve acceptable performance under decentralized training.

### 1.3 Baseline Paradigm Comparison

Existing PFL approaches manage these constraints through different structural strategies:

- **Global Consensus (e.g., FedAvg):** Trains a single shared model. Memory overhead is low ($1\times$), but accuracy degrades under extreme label skew due to client gradient drift.
- **Dual-Model Regularization (e.g., Ditto):** Trains two complete neural networks per client (a global model and a private local model) linked by an $L_2$ proximal penalty. This achieves high personalization accuracy under skew but doubles peak VRAM footprint and forward/backward passes per batch.
- **Split-Head Architectures (e.g., FedRep, FedBABU):** Shares a global feature extractor while keeping private local classification heads. While memory-efficient, standard split-head methods can experience performance drops when evaluated on homogeneous (IID) data because local heads lack global multi-class regularization.
- **Clustered Federated Learning (e.g., CFL):** Groups clients by gradient similarity to train cluster-level models, but lacks on-device head specialization for individual client distributions.

---

## 2. Implemented Architecture & Mathematical Formulations

Auto-HEP structures client-side computation around a single shared feature extractor paired with lightweight linear heads. Below are the specific mathematical equations implemented in the codebase.

```
+-------------------------------------------------------------+
|                      Client Local Batch                     |
+-------------------------------------------------------------+
                              |
                              v
                +----------------------------+
                |   Shared Feature Extractor |
                |          h = f_θ(X)        |
                +----------------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
    +-------------------+           +-------------------+
    |  Global Root Head |           | Private Local Head|
    |    z_r = w_r · h  |           |   z_l = w_l · h   |
    +-------------------+           +-------------------+
              |                               | (ACLM Masking)
              v                               v
    +-------------------+           +-------------------+
    | Loss: L_CE(z_r, Y)|           |Loss: L_ACLM(z_l,Y)|
    |   Weight: λ_r     |           |   Weight: λ_l     |
    +-------------------+           +-------------------+
              \                               /
               \                             /
                +-------------> <-----------+
                               |
                               v
                    Composite Loss: L_batch
              (Single Backward Pass on θ, w_r, w_l)
```

### 2.1 Label Skew Metric ($R_{skew}$)
To quantify local class imbalance without manual threshold tuning, the codebase implements the normalized order-1 Hill number (exponential of Shannon entropy):

$$H(p_i) = -\sum_{c=1}^C p_{i,c} \ln p_{i,c}$$

$$R_{skew,i} = \frac{\exp(H(p_i)) - 1}{C - 1}$$

**Variable Definitions:**
- $p_i \in \mathbb{R}^C$: Empirical class probability distribution on client $i$, where $p_{i,c} = \frac{|\mathcal{D}_{i,c}|}{|\mathcal{D}_i|}$.
- $C$: Total number of global classes (e.g., $C=10$ for CIFAR-10, $C=100$ for CIFAR-100).
- $R_{skew,i} \in [0, 1]$: Computed skew metric.

**Operational Purpose:**
- When client $i$ possesses data from only one class, $H(p_i) = 0 \implies R_{skew,i} = 0.0$.
- When client $i$ observes all $C$ classes uniformly, $H(p_i) = \ln(C) \implies R_{skew,i} = 1.0$.
- For any subset of $k$ uniform classes, $R_{skew,i} = \frac{k-1}{C-1}$, providing a linear mapping of class support to $[0, 1]$.

### 2.2 Loss Weight Calibration
During training, the client optimizes a composite loss over its active heads. The loss weights are determined by a polynomial partition of unity based on $R_{skew,i}$ and a dynamic anchor floor $a_i$:

$$a_i = \max\left(\frac{1}{2K}, \frac{|\mathcal{Y}_i|}{C}\right)$$

where $|\mathcal{Y}_i|$ is the number of active classes observed by client $i$, and $K$ is the number of clusters.

#### 3-Tier Weight Formulation:
$$\lambda_{r,i} = a_i + (1 - a_i) R_{skew,i}^2$$
$$\lambda_{p,i} = 2 R_{skew,i} (1 - R_{skew,i})$$
$$\lambda_{l,i} = (1 - R_{skew,i})^2$$

Normalized such that:
$$\hat{\lambda}_{k,i} = \frac{\lambda_{k,i}}{\lambda_{r,i} + \lambda_{p,i} + \lambda_{l,i}} \quad \text{for } k \in \{r, p, l\}$$

#### 2-Tier Bipartite Weight Formulation:
$$\lambda_{r,i} = a_i + (1 - a_i) R_{skew,i}^2$$
$$\lambda_{l,i} = (1 - R_{skew,i})^2$$

Normalized such that:
$$\hat{\lambda}_{r,i} = \frac{\lambda_{r,i}}{\lambda_{r,i} + \lambda_{l,i}}, \quad \hat{\lambda}_{l,i} = \frac{\lambda_{l,i}}{\lambda_{r,i} + \lambda_{l,i}}$$

**Operational Purpose:**
- Under uniform data ($R_{skew} = 1.0$), $\hat{\lambda}_r = 1.0$ and $\hat{\lambda}_l = 0.0$, directing training updates into the shared global head (matching FedAvg).
- Under extreme skew ($R_{skew} \to 0.0$), $\hat{\lambda}_l \approx \frac{1}{1+a_i}$ and $\hat{\lambda}_r \approx \frac{a_i}{1+a_i}$, concentrating optimization on the private local head while maintaining an anchor gradient on the shared feature extractor.

### 2.3 Active-Class Logit Masking (ACLM)
On high-cardinality datasets ($C=100$) where clients observe only a small subset of classes ($\mathcal{Y}_i \subset \{1, \dots, C\}$), standard Cross-Entropy loss penalizes unobserved logits, inducing negative gradient updates on absent classes. ACLM restricts Cross-Entropy evaluation on the Local head strictly to observed classes:

$$\mathcal{L}_{\text{ACLM}}(z_l, y) = -\log \left( \frac{\exp(z_{l, y})}{\sum_{c \in \mathcal{Y}_i} \exp(z_{l, c})} \right)$$

In implementation, logits corresponding to unobserved classes ($c \notin \mathcal{Y}_i$) are masked by setting $z_{l, c} = -10^9$ before computing the softmax denominator. The Root head remains unmasked to preserve global multi-class decision boundaries.

### 2.4 Inference Prediction Blending
During local evaluation, predictions from the active heads are combined using an interpolation scalar $\alpha_r \in [0, 1]$:

$$z_{blend} = \left[ (1 - \alpha_r) \hat{z}_l + \alpha_r \hat{z}_r \right] \odot \mathbf{m}_i$$

where $\mathbf{m}_i \in \{-\infty, 0\}^C$ masks unobserved classes ($m_{i,c}=0$ for $c \in \mathcal{Y}_i$, $m_{i,c}=-\infty$ for $c \notin \mathcal{Y}_i$), ensuring high-entropy background Root logits do not dilute local predictions.

---

## 3. Systems-Level Efficiency & Measured Profiling

Table 1 summarizes measured hardware resource allocations on CIFAR-10 ResNet-9 at batch size $B=32$.

### Table 1: Measured Hardware Footprint across PFL Paradigms
| Method | Models / Client | Total Parameters | Peak VRAM | Per-Batch Latency | Passes / Batch | Uplink / Downlink Payload |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **FedAvg** | 1 | 1.65M | 108.58 MB | 8.32 ms | 3 | 6.60 MB / 6.60 MB |
| **Ditto** | 2 | 3.30M | 217.15 MB | 16.78 ms | 6 | 6.60 MB / 6.60 MB |
| **FedRep** | 1 | 1.65M | 108.58 MB | 13.92 ms | 5 | 6.59 MB / 6.59 MB |
| **Auto-HEP (3-Tier)** | 1 | 1.66M | 113.42 MB | 8.35 ms | 5 | 6.61 MB / 6.61 MB |
| **Bipartite Auto-HEP (2-Tier)** | 1 | 1.65M | **110.10 MB** | **8.30 ms** | 4 | **6.60 MB / 6.60 MB** |

### Systems Analysis:
1. **Memory Allocation:** Dual-model methods (Ditto) maintain two independent model graphs and optimizer states, requiring 217.15 MB peak VRAM. By sharing the feature extractor $\theta$ across heads, Auto-HEP allocates 113.42 MB (a **47.8% reduction**). The linear heads add only 10 KB per head ($0.15\%$ of model parameters), resulting in minimal memory overhead over FedAvg (+4.84 MB).
2. **Latency Breakdown:** Evaluating the convolutional backbone once per batch yields an 8.35 ms latency for Auto-HEP compared to 16.78 ms for Ditto (a **50.2% reduction**).

---

## 4. Empirical Evaluation & The 2-Tier Bipartite Simplification

### 4.1 Benchmark Comparison across Heterogeneity Regimes
Table 2 reports the benchmark results across five Dirichlet heterogeneity regimes on CIFAR-10 ResNet-9.

### Table 2: Multi-Paradigm Personalization Benchmark (CIFAR-10 ResNet-9)
| Method | IID ($\alpha=\infty$) Avg / B-10 | Mild ($\alpha=1.0$) Avg / B-10 | Moderate ($\alpha=0.5$) Avg / B-10 | Severe ($\alpha=0.1$) Avg / B-10 | Extreme ($\alpha=0.05$) Avg / B-10 |
|---|:---:|:---:|:---:|:---:|:---:|
| **FedAvg** | 72.43% / 61.88% | 67.77% / 57.94% | 67.69% / 57.77% | 58.57% / 50.09% | 56.44% / 48.08% |
| **Ditto** | 67.93% / 63.50% | 72.54% / 64.88% | 73.94% / 62.21% | 83.13% / 65.41% | 88.33% / 70.00% |
| **FedRep** | 61.79% / 57.50% | 68.87% / 62.93% | 70.67% / 58.13% | 81.45% / 64.32% | 87.07% / **76.85%** |
| **FedBABU** | 72.10% / 66.75% | 75.67% / 69.61% | 76.02% / 31.04% | 84.43% / 59.73% | 88.39% / 70.00% |
| **Auto-HEP (3-Tier)** | **72.92%** / 66.50% | **76.69%** / 69.76% | **78.28%** / **69.89%** | **85.18%** / **71.49%** | **89.03%** / 74.29% |

### 4.2 Analysis of Intermediate Cluster Head Redundancy
A key question is whether the intermediate Parent (cluster) head is necessary to achieve these results. Table 3 presents the Gate B ablation data from Table VIII of the paper, comparing the full 3-tier model ($K=3$) against a 2-tier bipartite baseline ($K=1$, Root + Local).

### Table 3: Cluster Count Sensitivity ($K=1$ vs. $K=3$ Ablation)
| Heterogeneity Regime | Full 3-Tier ($K=3$) | Bipartite Baseline ($K=1$) | Delta ($K=1 - K=3$) | Observation |
|---|:---:|:---:|:---:|---|
| **IID ($\alpha=\infty$)** | 72.92% | **73.20%** | +0.28 pp | Bipartite matches 3-tier |
| **Mild ($\alpha=1.0$)** | 76.69% | **77.13%** | +0.44 pp | Bipartite matches 3-tier |
| **Moderate ($\alpha=0.5$)** | 78.28% | **78.37%** | +0.09 pp | Bipartite matches 3-tier |
| **Severe ($\alpha=0.1$)** | **85.18%** | 85.11% | -0.07 pp | Within measurement noise |
| **Extreme ($\alpha=0.05$)** | 89.03% | **89.47%** | +0.44 pp | Bipartite matches 3-tier |

**Key Findings:**
1. Across all five regimes, the difference between $K=1$ and $K=3$ remains within $\pm 0.44\text{ pp}$.
2. Under extreme skew and IID conditions, the binomial loss weight $\lambda_p = 2R(1-R)$ automatically decays to zero, rendering the Parent head inactive.
3. In Table IX of the paper, replacing update-similarity clustering with uniform random cluster assignment degrades accuracy by $<0.4\text{ pp}$.
4. In Table XI, when differential privacy noise on sketches increases to $\sigma=1.0$ (reducing cluster assignment stability to 20%), overall test accuracy decreases by only 0.57%.

**Architectural Takeaway:** These ablation results indicate that the empirical performance of Auto-HEP is primarily driven by shared-backbone multiplexing and entropy-based loss weighting, while dynamic clustering mechanisms provide marginal empirical impact.

---

## 5. Methodological Observations & Trade-offs

### 5.1 Hyperparameter Dependencies
While Auto-HEP derives loss weights dynamically from label distributions, several global configuration values remain fixed:
- **Sketch Dimension:** $m = 64$ fixed for projection matrices.
- **Gradient Clipping Bound:** $C_g = 1.0$ for differential privacy noise calibration.
- **Inference Temperature Clamps:** Clamped to $[0.2, 2.0]$.
- **Fallback Staleness Constant:** $\tau_0 = 4.0$ when sampling histories are unavailable.

The framework is therefore more precisely described as **parameter-reduced** rather than fully parameter-free.

### 5.2 Privacy Architecture & Information Boundaries
Auto-HEP utilizes a two-channel communication design:
- **Channel 1 (Routing Sketches):** Transmits an $m=64$ sketch $s_i = P \Delta_i^{head} \in \mathbb{R}^{64}$ in the clear to the central server.
- **Channel 2 (Model Aggregation):** Transmits full backbone and root head parameters $(\theta_i, w_{r,i})$ via Secure Aggregation (SecAg).

**Analytical Observation:**  
Projecting head updates ($d_{head}=2560$) to $m=64$ constraints creates an underdetermined linear system with $\dim(\ker(P)) = 2496$ free dimensions, preventing exact pixel-level gradient inversion (DLG). However, because sketch projections preserve cosine distances, directional sketch angles remain correlated with client label distributions and cluster membership.

### 5.3 Worst-Case Client Fairness Under Extreme Skew
Under extreme label skew ($\alpha=0.05$), split-head FedRep achieves a bottom-10% client fairness of **76.85%**, compared to Auto-HEP's **74.29%** (Table 2). While Auto-HEP achieves higher average accuracy across all clients (89.03% vs. 87.07%), FedRep's complete decoupling of private heads provides higher tail-client accuracy in this specific regime.

---

## 6. Summary of Validation Script Checks

The companion verification script ([`scripts/verify_theoretical_claims.py`](file:///C:/Users/IT/Topology-aware-FDL/scripts/verify_theoretical_claims.py)) was executed to deterministically check the implemented formulas:

1. **Skew Metric Invariance ($R_{skew}$):** Confirmed that single-class inputs yield $R_{skew} = 0.000000$ and uniform inputs yield $R_{skew} = 1.000000$ across both $C=10$ and $C=100$.
2. **Loss Weight Partition of Unity:** Confirmed that $\sum \hat{\lambda}_k = 1.0$ across 1,000 discrete evaluation points (maximum numerical deviation $< 2.3 \times 10^{-16}$).
3. **Dimensionality Identity:** Confirmed head parameter dimensions ($d_{head}=2560$) and null-space dimensions ($2496$) for ResNet-9 CIFAR-10.
4. **Gate B Ablation Consistency:** Re-evaluated the five regime deltas between $K=1$ and $K=3$, confirming all differences satisfy $|\Delta| \le 0.44\text{ pp}$.

---

## 7. Conclusion

Auto-HEP presents a practical systems engineering solution for edge federated learning. Its primary contribution is **shared-backbone compute multiplexing**, which reduces peak VRAM footprint by 47.8% and per-batch latency by 50.2% relative to dual-model approaches, while preventing the IID generalization collapse observed in decoupled split-head baselines. 

Analysis of published ablations demonstrates that a simplified **2-Tier Bipartite architecture** (Global Root Head + Private Local Head) captures the empirical benefits of the system while reducing implementation complexity, communication overhead, and server-side state tracking.
