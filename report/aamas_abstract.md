# AAMAS 2027 Submission Metadata & Abstract

**Target Conference**: The 26th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2027)  
**Track**: *Distributed and Collaborative Machine Learning / Robust Distributed Systems*  
**Abstract Submission Deadline**: October 1, 2026 (23:59 AoE)  
**Full Paper Deadline**: October 8, 2026 (23:59 AoE)  
**Submission Portal**: AAMAS 2027 CMT / EasyChair  

---

## Metadata

- **Title**: *H-ResFL: Efficient, Multi-Scale Residual Personalization and Byzantine Resilience in Heterogeneous Federated Learning*
- **Alternative Acronym**: *HEP-FL (Hierarchical Ensemble Personalization)*
- **Authors**:
  - **Nghiem Duc Khanh Nam**\* (College of Engineering & Computer Science, VinUniversity, Vietnam) — `nam.ndk@vinuni.edu.vn`
  - **Hung Anh Nguyen**\* (College of Engineering & Computer Science, VinUniversity, Vietnam) — `anh.nh@vinuni.edu.vn`
  - **Leandro Soriano Marcolino** (School of Computing & Communications, Lancaster University, UK) — `l.marcolino@lancaster.ac.uk`  
  *\* Equal contribution.*
- **Primary Area**: *Distributed Machine Learning / Cooperative Learning in Multi-Node Systems*
- **Secondary Area**: *Fault Tolerance, Safety, and Trustworthiness in Distributed Systems*
- **Keywords**: Federated Learning, Multi-Scale Personalization, Byzantine Fault Tolerance, Statistical Heterogeneity, Tail Fairness, Edge Computing.

---

## Official Abstract (Pure Federated Learning — Number-Resilient Version)

In distributed edge networks, heterogeneous clients must collaborate to acquire global domain representations while adapting to non-identical (non-IID) local data distributions. However, practical edge federated learning faces a fundamental trilemma: (1) severe statistical heterogeneity causes monolithic consensus models (e.g., FedAvg) to collapse due to client drift; (2) state-of-the-art personalized baselines (e.g., Ditto) duplicate entire neural networks on device, imposing prohibitive memory and compute penalties on resource-constrained hardware; and (3) standard Byzantine-robust aggregators (e.g., Krum, geometric median, uncalibrated cosine filters) confound legitimate statistical specialization with adversarial poison, catastrophically penalizing honest domain specialists.

To resolve these tensions, we introduce **H-ResFL** (*Hierarchical Residual Federated Learning*), a lightweight, multi-scale personalization framework with subspace-calibrated Byzantine defense. H-ResFL factorizes model parameters additively into global foundational, cluster-shared, and private residual components ($W_{\text{eff}, i} = W_0 + \Delta W_{c(i)} + \Delta W_{l, i}$) on a single shared backbone, avoiding the memory duplication of dual-model approaches. To neutralize malicious actors without discarding specialized honest clients, we formulate a **Skew-Calibrated Subspace Defense** that restricts directional alignment evaluation strictly to each client's active decision subspace, coupled with temporal trust tracking.

Extensive empirical evaluations across multiple non-IID Dirichlet skew regimes on high-cardinality CIFAR-100 demonstrate that H-ResFL consistently outperforms canonical global and personalized baselines as data heterogeneity intensifies. Under severe Byzantine attacks, our framework maintains high task accuracy where undefended baselines and conventional distance-based filters suffer catastrophic collapse. Furthermore, H-ResFL significantly elevates tail-client fairness, cuts on-device memory and latency by approximately half, and demonstrates zero-shot architectural transfer to continuous multi-sensor regression.

---

## Structured Scientific Summary for Reviewers

### 1. Problem Formulation: The Federated Trilemma
Real-world federated deployments (e.g., connected autonomous edge devices, distributed sensors, mobile clients) must balance three conflicting requirements:
1. **Representational Expressiveness**: Capturing universal foundational features without succumbing to local client drift under non-IID Dirichlet skew.
2. **On-Device Hardware Feasibility**: Operating within strict on-device RAM/VRAM constraints without doubling parameters like dual-model architectures.
3. **Byzantine Fault Tolerance under Heterogeneity**: Filtering adversarial model poisoning, sign-flipping, and label corruption without falsely rejecting honest specialized nodes.

### 2. Core Methodological Contributions
1. **Multi-Scale Additive Residual Decomposition**:
   Decomposes parameters into $W_{\text{eff}, i} = W_0 + \Delta W_{c(i)} + \Delta W_{l, i}$. The global base $W_0$ preserves common features, cluster residuals $\Delta W_c$ capture domain affinities, and private local residuals $\Delta W_l$ adapt locally on-device.
2. **Skew-Calibrated Subspace Directional Defense**:
   Solves the "heterogeneity vs. robustness" trap by projecting directional similarity into each client's active label support mask ($\mathcal{Y}_i$).
3. **Temporal Trust Tracking with Adaptive Temperature**:
   Maintains dynamic client trust scores $T_i^{(t)} = \beta T_i^{(t-1)} + (1-\beta)\tau_i^{(t)}$, isolating malicious actors whose update trajectories consistently diverge from valid subspaces.
4. **Tail-Client Fairness**:
   Improves worst-performing client deciles (bottom 10%) through sample-adaptive local optimization, ensuring data-sparse nodes are not starved of utility.

### 3. Empirical Highlights
- **High-Class-Cardinality Classification**: Evaluated on CIFAR-100 ($C=100$) across 5 Dirichlet concentration regimes ($\alpha \in [\infty, 1.0, 0.5, 0.1, 0.05]$).
- **Byzantine Attacks**: Robustness verified under Label-Flipping and Sign-Flipping vectors at corruption ratios up to $q = 0.3$.
- **Edge Efficiency**: Profiled on MobileNetV3-Small depthwise-separable convolutional networks, verifying $\approx 50\%$ VRAM and latency savings over Ditto.
- **Cross-Domain Generalization**: Zero-shot transfer to continuous multi-sensor regression ($R^2 = 99.95\%$).
