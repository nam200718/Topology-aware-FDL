# AAMAS 2027 Submission Metadata & Abstract

**Target Conference**: The 26th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2027)  
**Track**: *Distributed and Collaborative Machine Learning / Robust Distributed Systems*  
**Abstract Submission Deadline**: October 1, 2026 (23:59 AoE)  
**Full Paper Deadline**: October 8, 2026 (23:59 AoE)  
**Submission Portal**: AAMAS 2027 CMT / EasyChair  

---

## Submission Metadata

- **Title**: *FedHEP: Efficient Hierarchical Ensemble Personalization in Federated Learning*
- **Framework Moniker**: **FedHEP** (*Federated Hierarchical Ensemble Personalization*)
- **Authors**:
  - **Nghiem Duc Khanh Nam**\* (College of Engineering & Computer Science, VinUniversity, Vietnam) — `nam.ndk@vinuni.edu.vn`
  - **Hung Anh Nguyen**\* (College of Engineering & Computer Science, VinUniversity, Vietnam) — `anh.nh@vinuni.edu.vn`
  - **Leandro Soriano Marcolino** (School of Computing & Communications, Lancaster University, UK) — `l.marcolino@lancaster.ac.uk`  
  *\* Equal contribution.*
- **Primary Area**: *Distributed Machine Learning / Cooperative Learning in Multi-Node Systems*
- **Secondary Area**: *Fault Tolerance, Safety, and Trustworthiness in Distributed Systems*
- **Keywords**: Federated Learning, Multi-Scale Personalization, Byzantine Fault Tolerance, Statistical Heterogeneity, Tail Fairness, Edge Computing.

---

## CMT / EasyChair Portal Plain Text Abstract
*(Word Count: ~150 words — strictly under the 250-word portal ceiling)*

```text
Federated Learning (FL) on resource-constrained edge devices faces a fundamental trilemma: mitigating client drift under non-IID data, enabling personalization without dual-model memory overhead, and defending against Byzantine poisoning without penalizing honest specialized clients. We introduce FedHEP (Federated Hierarchical Ensemble Personalization), a single-backbone framework coordinating three representation tiers—global consensus, collaborative peer clusters, and private local heads. First, Active-Class Logit Masking (ACLM) with skew-calibrated loss weighting eliminates gradient interference from unobserved categories without duplicating feature extractors. Second, low-dimensional Random Projections (RP) enable privacy-preserving cluster discovery among peers. Third, Subspace-Constrained Cosine Filtering (SCCF) coupled with Temporal Trust Tracking (TTT) enhances Byzantine resilience under moderate corruption by leveraging active-class alignment to reduce false-positive rejection of specialized clients. Experiments on CIFAR-100 with ResNet-9 across five Dirichlet non-IID regimes, multi-attack Byzantine benchmarks, and 50-client scaling show that FedHEP improves personalization accuracy and worst-decile tail fairness over competitive baselines. Simulated on MobileNetV3 confirms single-model memory footprint and latency efficiency on resource-constrained edge platforms.
```

**TL;DR:** A single-backbone hierarchical ensemble enabling three-tier multi-head personalization for non-IID federated edge networks.

---

## Formatted Abstract (LaTeX)

```latex
\begin{abstract}
Federated Learning (FL) on resource-constrained edge devices faces a fundamental trilemma: mitigating client drift under non-IID data, enabling personalization without dual-model memory overhead, and defending against Byzantine poisoning without penalizing honest specialized clients. We introduce \textbf{\fedhep{}} (\textit{Federated Hierarchical Ensemble Personalization}), a single-backbone framework coordinating three representation tiers---global consensus, collaborative peer clusters, and private local heads. First, Active-Class Logit Masking (\textbf{ACLM}) with skew-calibrated loss weighting eliminates gradient interference from unobserved categories without duplicating feature extractors. Second, low-dimensional Random Projections (\textbf{RP}) enable privacy-preserving cluster discovery among peers. Third, Subspace-Constrained Cosine Filtering (\textbf{SCCF}) coupled with Temporal Trust Tracking (\textbf{TTT}) enhances Byzantine resilience under moderate corruption by leveraging active-class alignment to reduce false-positive rejection of specialized clients. Experiments on CIFAR-100 across five Dirichlet non-IID regimes, multi-attack Byzantine benchmarks, and 50-client scaling show that \fedhep{} improves personalization accuracy and worst-decile tail fairness over competitive baselines while maintaining single-model edge efficiency on MobileNetV3. Source code is provided in the supplementary material.
\end{abstract}
```

---

## Methodological Summary

### 1. Three Practical Challenges Addressed
1. **Client Drift**: Extreme statistical non-IID distributions degrade monolithic consensus models.
2. **On-Device Resource Bloat**: Dual-model architectures double parameter footprint and backpropagation VRAM.
3. **Robustness vs. Heterogeneity Dilemma**: Euclidean and global-cosine robust aggregators penalize non-IID specialization as malicious anomalies.

### 2. FedHEP Core Architectural Mechanics
1. **Single-Backbone 3-Tier Hierarchy**:
   A single shared feature extractor backbone with branched Root (global server), Parent (cluster/peer level), and Local (client on-device) linear classification heads.
2. **Local Label Skew Metric ($R_{skew}$)**:
   Measures empirical local class entropy to smoothly parameterize edge specialization.
3. **Anchored Dynamic Binomial Loss Weighting**:
   Exact continuous partition-of-unity ($\alpha_r + \alpha_p + \alpha_l = 1.0$) with data-support floor $a_i = \max(1/(2K), |\mathcal{Y}_i|/C)$ preventing under-anchoring on sparse edge devices.
4. **Active-Class Logit Masking (ACLM)**:
   Restricts loss computation on Parent and Local heads strictly to observed active classes, eliminating negative gradient drag on the shared backbone.
5. **Information-Reducing Random Projections (RP / Parameter Sketches, $m=256$)**:
   Projects client head update deltas into 256-dimensional random orthogonal sketches, bounding geometric distance distortion by the Johnson-Lindenstrauss lemma while keeping gradient reconstruction severely underdetermined.
6. **Subspace-Constrained Cosine Filtering (SCCF) & Temporal Trust Tracking (TTT)**:
   Evaluates cosine trust alignment strictly over each client's active class coordinates, combined with a sentinel NaN/Inf pre-filter, norm-bounding on backbone updates, and multi-round trust scoring to isolate persistent adversaries.

### 3. Master Experimental Suite (src/baselines/)
- **Job 1 (Personalization Benchmark)**:
  CIFAR-100 (ResNet-9) across 5 Dirichlet regimes ($\alpha \in [\infty, 1.0, 0.5, 0.1, 0.05]$) against 6 baselines (**FedAvg, FedProx, Multi-Krum, SCAFFOLD, Ditto, FedRep**) across 3 seeds (`42, 123, 7`).
- **Job 2 (Byzantine Robustness Matrix)**:
  4 Byzantine attack vectors (**label-flipping, sign-flipping, gradient ascent, Gaussian noise**) across corruption rates $q \in [0.0, 0.4]$ on CIFAR-100.
- **Job 3 (50-Client Scalability Benchmark)**:
  Population scaling to $N=50$ clients under partial participation ($C_p=0.20$), evaluating mean accuracy and worst-decile (bottom 10%) tail fairness.
- **Job 4 (Physical Edge Hardware Profiling)**:
  On-device footprint comparison on **MobileNetV3-Small** vs. **ResNet-9** (batch latency, peak VRAM, communication payload per round).
- **Job 5 (Ablation Study & Cluster Valuation)**:
  Component isolation testing **w/o ACLM**, **w/o Parent Head (2-tier bipartite)**, single grand coalition ($K=1$), and the theoretical **$K=3$ Oracle Bound** via ground-truth label distribution clustering.
