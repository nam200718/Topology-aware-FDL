# AAMAS 2027 Submission Metadata & Abstract

**Target Conference**: The 26th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2027)  
**Track**: *Distributed and Collaborative Machine Learning / Robust Distributed Systems*  
**Abstract Submission Deadline**: October 1, 2026 (23:59 AoE)  
**Full Paper Deadline**: October 8, 2026 (23:59 AoE)  
**Submission Portal**: AAMAS 2027 CMT / EasyChair  

---

## Submission Metadata

- **Title**: *HEP-FL: Efficient, Multi-Scale Residual Personalization and Byzantine Resilience in Heterogeneous Federated Learning*
- **Framework Moniker**: **HEP-FL** (*Hierarchical Ensemble Personalization in Federated Learning*)
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
*(Word Count: 228 words — strictly under the 250-word portal ceiling)*

```text
Federated learning (FL) on edge devices faces three practical challenges: (1) statistical heterogeneity causes global consensus models like FedAvg to suffer severe client drift; (2) dual-model personalization approaches like Ditto double on-device memory and latency, while naive split-head baselines suffer representation collapse under uniform data; and (3) standard Byzantine-robust aggregators mistake legitimate statistical skew for adversarial poisoning, penalizing honest specialized clients.

We propose HEP-FL (Hierarchical Ensemble Personalization in Federated Learning), a single-backbone architecture that coordinates three classification tiers: global consensus, cluster-level collaboration, and private on-device specialization. HEP-FL uses dynamic binomial loss weighting and active-class logit masking (ACLM) to prevent representation drag from unobserved classes without dual-model memory bloat. Clients privately organize into peer clusters via low-dimensional parameter sketches without sharing private data. To counter model poisoning without penalizing statistical skew, HEP-FL integrates a skew-calibrated defense that evaluates directional alignment strictly within observed class subspaces alongside temporal trust tracking.

We evaluate HEP-FL against six baselines (FedAvg, FedProx, Multi-Krum, SCAFFOLD, Ditto, and FedRep) on CIFAR-100 across five Dirichlet non-IID regimes. Experiments encompass four Byzantine attack types, a 50-client network under partial participation, component ablations isolating logit masking and the cluster tier against an oracle clustering bound, and physical on-device profiling on MobileNetV3 and ResNet-9. HEP-FL preserves client specialization, maintains Byzantine resilience, improves worst-decile tail fairness, and minimizes edge resource overhead.
```

---

## Formatted Abstract (LaTeX)

```latex
\begin{abstract}
Federated learning (FL) on edge devices faces three practical challenges: (1) statistical heterogeneity causes global consensus models like FedAvg to suffer severe client drift; (2) dual-model personalization approaches like Ditto double on-device memory and latency, while naive split-head baselines suffer representation collapse under uniform data; and (3) standard Byzantine-robust aggregators mistake legitimate statistical skew for adversarial poisoning, penalizing honest specialized clients.

We propose \textbf{HEP-FL} (\textit{Hierarchical Ensemble Personalization in Federated Learning}), a single-backbone architecture that coordinates three classification tiers: global consensus, cluster-level collaboration, and private on-device specialization. HEP-FL uses dynamic binomial loss weighting and active-class logit masking (ACLM) to prevent representation drag from unobserved classes without dual-model memory bloat. Clients privately organize into peer clusters via low-dimensional parameter sketches without sharing private data. To counter model poisoning without penalizing statistical skew, HEP-FL integrates a skew-calibrated defense that evaluates directional alignment strictly within observed class subspaces alongside temporal trust tracking.

We evaluate HEP-FL against six established baselines (FedAvg, FedProx, Multi-Krum, SCAFFOLD, Ditto, and FedRep) on CIFAR-100 across five Dirichlet non-IID regimes ($\alpha \in [\infty, 1.0, 0.5, 0.1, 0.05]$). Experiments encompass four Byzantine attack types (label-flipping, sign-flipping, gradient ascent, and Gaussian noise), a 50-client network under partial participation ($C_p=0.20$), component ablations isolating logit masking and the cluster tier against an oracle clustering bound, and physical on-device profiling on MobileNetV3 and ResNet-9. HEP-FL preserves client specialization, maintains Byzantine resilience, improves worst-decile tail fairness, and minimizes edge resource overhead.
\end{abstract}
```

---

## Methodological Summary

### 1. Three Practical Challenges Addressed
1. **Client Drift**: Extreme statistical non-IID distributions degrade monolithic consensus models.
2. **On-Device Resource Bloat**: Dual-model architectures double parameter footprint and backpropagation VRAM.
3. **Robustness vs. Heterogeneity Dilemma**: Euclidean and global-cosine robust aggregators penalize non-IID specialization as malicious anomalies.

### 2. HEP-FL Core Architectural Mechanics
1. **Single-Backbone 3-Tier Hierarchy**:
   A single shared feature extractor backbone with branched Root (global server), Parent (cluster/peer level), and Local (client on-device) linear classification heads.
2. **Local Label Skew Metric ($R_{skew}$)**:
   Measures empirical local class entropy to smoothly parameterize edge specialization.
3. **Anchored Dynamic Binomial Loss Weighting**:
   Exact continuous partition-of-unity ($\alpha_r + \alpha_p + \alpha_l = 1.0$) with data-support floor $a_i = \max(1/(2K), |\mathcal{Y}_i|/C)$ preventing under-anchoring on sparse edge devices.
4. **Active-Class Logit Masking (ACLM)**:
   Restricts loss computation on Parent and Local heads strictly to observed active classes, eliminating negative gradient drag on the shared backbone.
5. **Information-Reducing Parameter Sketch Clustering ($m=256$)**:
   Projects client head update deltas into 256-dimensional random orthogonal sketches, bounding geometric distance distortion by the Johnson-Lindenstrauss lemma while keeping gradient reconstruction severely underdetermined.
6. **Skew-Calibrated Subspace Defense**:
   Evaluates cosine trust alignment strictly over each client's active class coordinates, combined with a sentinel NaN/Inf pre-filter and multi-round trust tracking.

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
