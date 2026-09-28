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

## CMT / EasyChair Portal-Ready Plain Text Abstract
*(Copy and paste directly into the submission portal text box)*

```text
In distributed edge networks, heterogeneous clients must collaborate to acquire global domain representations while adapting to non-identical (non-IID) local data distributions. However, real-world edge federated learning faces a fundamental trilemma: (1) severe statistical heterogeneity causes monolithic consensus models to suffer severe client drift; (2) state-of-the-art personalized baselines duplicate entire neural networks on device, imposing prohibitive memory and computational overhead on resource-constrained hardware; and (3) standard Byzantine-robust aggregators conflate legitimate statistical specialization with adversarial poison, erroneously penalizing honest domain specialists whose local updates diverge from the global mean.

To resolve these competing tensions, we introduce HEP-FL (Hierarchical Ensemble Personalization in Federated Learning), a lightweight, multi-scale residual framework with subspace-calibrated Byzantine defense. HEP-FL factorizes model parameters additively into global foundational, cluster-shared, and private residual components on a single shared backbone, structurally circumventing the memory duplication inherent to dual-model personalization architectures. To neutralize malicious participants without discarding specialized honest clients, we formulate a Skew-Calibrated Subspace Defense that restricts directional alignment evaluation strictly to each client's active decision subspace, coupled with temporal trust tracking across communication rounds.

We evaluate HEP-FL across a comprehensive experimental suite spanning high-class-cardinality vision tasks across diverse Dirichlet non-IID skew regimes, coordinated Byzantine attack vectors (including gradient sign-flipping and label manipulation), large-scale client populations under partial participation, physical mobile edge profiling, and continuous distributed sensor regression. Our systematic evaluation examines the operational trade-offs between representational consensus, localized personalization, and adversarial resilience, demonstrating that multi-scale residual parameterization provides an efficient, fair, and robust foundation for decentralized edge learning.
```

---

## Official Formatted Abstract (Markdown / LaTeX)

In distributed edge networks, heterogeneous clients must collaborate to acquire global domain representations while adapting to non-identical (non-IID) local data distributions. However, real-world edge federated learning faces a fundamental trilemma: (1) severe statistical heterogeneity causes monolithic consensus models to suffer severe client drift; (2) state-of-the-art personalized baselines duplicate entire neural networks on device, imposing prohibitive memory and computational overhead on resource-constrained hardware; and (3) standard Byzantine-robust aggregators conflate legitimate statistical specialization with adversarial poison, erroneously penalizing honest domain specialists whose local updates diverge from the global mean.

To resolve these competing tensions, we introduce **HEP-FL** (*Hierarchical Ensemble Personalization in Federated Learning*), a lightweight, multi-scale residual framework with subspace-calibrated Byzantine defense. HEP-FL factorizes model parameters additively into global foundational, cluster-shared, and private residual components ($W_{\text{eff}, i} = W_0 + \Delta W_{c(i)} + \Delta W_{l, i}$) on a single shared backbone, structurally circumventing the memory duplication inherent to dual-model personalization architectures. To neutralize malicious participants without discarding specialized honest clients, we formulate a **Skew-Calibrated Subspace Defense** that restricts directional alignment evaluation strictly to each client's active decision subspace, coupled with temporal trust tracking across communication rounds.

We evaluate HEP-FL across a comprehensive experimental suite spanning high-class-cardinality vision tasks across diverse Dirichlet non-IID skew regimes, coordinated Byzantine attack vectors (including gradient sign-flipping and label manipulation), large-scale client populations under partial participation, physical mobile edge profiling, and continuous distributed sensor regression. Our systematic evaluation examines the operational trade-offs between representational consensus, localized personalization, and adversarial resilience, demonstrating that multi-scale residual parameterization provides an efficient, fair, and robust foundation for decentralized edge learning.

---

## Methodological Blueprint for Reviewers

### 1. The Federated Trilemma
Practical edge deployments face three competing system requirements:
1. **Representational Expressiveness**: Capturing shared foundational features without succumbing to local client drift under severe non-IID skew.
2. **On-Device Hardware Feasibility**: Operating within strict on-device RAM/VRAM constraints without duplicating entire model parameter sets.
3. **Byzantine Fault Tolerance under Skew**: Defending against adversarial poison without penalizing honest clients specializing in distinct domain subspaces.

### 2. Methodological Architecture of HEP-FL
1. **Multi-Scale Additive Residual Decomposition**:
   $$W_{\text{eff}, i} = W_0 + \Delta W_{c(i)} + \Delta W_{l, i}$$
   - $W_0$: Global foundational representation trained via consensus.
   - $\Delta W_{c(i)}$: Cluster-shared residual capturing affinity across correlated client subsets.
   - $\Delta W_{l, i}$: Private local residual retained on device for immediate local personalization.
2. **Single Shared Backbone Execution**:
   Unlike dual-model methods (e.g., Ditto) that instantiate two independent networks, HEP-FL routes activations through a single feature extractor with branched residual projections, reducing on-device memory footprint.
3. **Skew-Calibrated Subspace Directional Defense**:
   Solves the heterogeneity-robustness dilemma by projecting directional similarity into each client's active label support mask ($\mathcal{Y}_i$):
   $$\text{sim}_{\text{subspace}}(u_i, \bar{u}) = \frac{\langle u_i \odot m_i, \bar{u} \odot m_i \rangle}{\|u_i \odot m_i\| \|\bar{u} \odot m_i\|}$$
4. **Temporal Trust Tracking with Adaptive Temperature**:
   Maintains dynamic client trust scores $T_i^{(t)} = \beta T_i^{(t-1)} + (1-\beta)\tau_i^{(t)}$, isolating malicious actors whose update trajectories consistently diverge from valid subspaces.

### 3. Redesigned Experimental Suite Scope (outputs/suite_v2/)
- **Track 1 (Statistical Heterogeneity)**: CIFAR-100 ($C=100$) across 5 Dirichlet concentration levels ($\alpha \in [\infty, 1.0, 0.5, 0.1, 0.05]$).
- **Track 2 (Byzantine Multi-Attack Matrix)**: Model poisoning across Sign-Flipping, Label-Flipping, and Gaussian Noise at corruption rates $q \in \{0.0, 0.1, 0.2, 0.3\}$.
- **Track 3 (Scalability & Tail Fairness)**: $N=50$ client swarm with partial participation ($C_p=0.20$), evaluating bottom-10% tail deciles and minimax fairness.
- **Track 4 (Component Ablations)**: Isolated ablation of root, cluster, and local residual tiers, temperature sharpening, and trust momentum.
- **Track 5 (Edge Hardware & Task Transfer)**: Physical MobileNetV3-Small profiling (VRAM, latency, communication bandwidth) and continuous distributed sensor regression.
