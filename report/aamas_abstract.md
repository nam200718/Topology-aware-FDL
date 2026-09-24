# AAMAS 2027 Submission Metadata & Abstract

**Target Conference**: The 26th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2027)  
**Abstract Submission Deadline**: October 1, 2026  
**Full Paper Deadline**: October 8, 2026  
**Submission Portal**: AAMAS 2027 CMT / EasyChair  

---

## Metadata

- **Title**: *Defended H-ResFL: Efficient, Fair, and Byzantine-Resilient Multi-Agent Coordination via Multi-Scale Residual Personalization*
- **Authors**:
  - **Nghiem Duc Khanh Nam**\* (College of Engineering & Computer Science, VinUniversity, Vietnam) — `nam.ndk@vinuni.edu.vn`
  - **Hung Anh Nguyen**\* (College of Engineering & Computer Science, VinUniversity, Vietnam) — `anh.nh@vinuni.edu.vn`
  - **Leandro Soriano Marcolino** (School of Computing & Communications, Lancaster University, UK) — `l.marcolino@lancaster.ac.uk`  
  *\* Equal contribution.*
- **Primary Track**: *Cooperative Multi-Agent Learning / Distributed Multi-Agent Systems*
- **Secondary Track**: *Multi-Agent Safety, Robustness, and Trustworthiness / Economic and Game-Theoretic MAS (Coalition Formation & Fairness)*
- **Keywords**: Multi-Agent Reinforcement and Federated Learning, Byzantine Fault Tolerance, Multi-Tier Coalition Formation, Personalized Federated Learning, Egalitarian Social Welfare, Edge Computing.

---

## Official Abstract (Plain Text for Portal Submission)

In decentralized multi-agent systems (MAS) operating across distributed edge networks, autonomous agents must collaborate to acquire global domain knowledge while adapting to idiosyncratic local environments and private utilities. However, cooperative edge learning faces a fundamental trilemma: (1) severe statistical heterogeneity causes monolithic models to collapse across divergent agent tasks; (2) adversarial Byzantine agents exploit decentralized coordination via model poisoning, sign-flipping, and label corruption; and (3) standard cosine-similarity defenses catastrophically penalize honest, highly specialized agents whose updates naturally diverge from the global centroid.

To resolve these tensions, we introduce **Defended H-ResFL** (*Hierarchical Residual Federated Learning with Multi-Tier Coalition Defense*), a principled multi-agent coordination framework uniting multi-scale residual parameterization with reputation-aware defense. In Defended H-ResFL, agents dynamically organize into cooperative coalitions via 256-dimensional privacy-preserving feature sketches. Agent policies are factorized additively into global foundational, coalition-specialized, and private residual components ($W_{\text{eff}} = W_0 + \Delta W_c + \Delta W_l$). To neutralize malicious actors without sacrificing specialized honest peers, we formulate a **Skew-Calibrated Subspace Cosine Defense** that restricts directional alignment evaluation to each agent's active class subspace, coupled with a variance-gated cumulative reputation model. 

Extensive empirical evaluations across five non-IID skew regimes on high-cardinality CIFAR-100 ($C=100$) demonstrate that Defended H-ResFL outperforms state-of-the-art personalized federated baselines by up to +11.65% mean accuracy under extreme skew (65.06% vs. Ditto's 53.41% and FedAvg's 19.83%), while matching global models on uniform IID data. Under extreme Byzantine attacks ($q = 0.3$), Defended H-ResFL maintains 88.92% test accuracy while undefended baselines collapse to 9.95%, successfully defending specialized agents where prior cosine filters fail (37.58%). Furthermore, our approach improves Rawlsian egalitarian welfare (lifting worst-off tail agents from 0.00% to 61.40%), slashes edge communication latency by 78.3%, and demonstrates zero-shot architectural generalization from deep vision to continuous edge sensor regression ($R^2 = 99.95\%$).

---

## Structured Scientific Summary for Reviewers

### 1. Motivation & Multi-Agent Problem Formulation
In real-world multi-agent deployments (e.g., connected autonomous vehicles, smart medical sensor arrays, robotics swarms), agents are self-interested, resource-constrained, and subject to heterogeneous local objectives. Traditional federated aggregation (FedAvg) enforces consensus on a single global model, creating a zero-sum tension between common utility and local agent performance. Conversely, naive clustering or local personalization fragments the multi-agent collective and leaves communication channels open to adversarial manipulation. Defended H-ResFL treats federated learning as a multi-tier coalition game with autonomous reputation tracking.

### 2. Core Innovations
1. **Multi-Scale Additive Residual Architecture**:
   Factorizes model weights into $W_{\text{eff}, i} = W_0 + \Delta W_{c(i)} + \Delta W_{l, i}$. This provides clean gradient routing: global foundation handles universal representations, coalition heads capture domain affinities, and private residuals overfit safe local nuances without parameter pollution.
2. **Skew-Calibrated Subspace Cosine Defense**:
   Solves the open problem of "heterogeneity vs. defense." Standard robust aggregators (Krum, Bulyan, soft cosine) confound non-IID statistical specialization with malicious poison. By projecting directional similarity into active class support masks ($\mathcal{Y}_i$), our defense detects adversarial attacks even when agents hold disjoint label distributions ($\alpha_{\text{inter}} = 0.1$).
3. **Cumulative MAS Reputation Model with Variance-Gated Temperature**:
   Maintains dynamic agent trust scores $R_i^{(t)} = \beta R_i^{(t-1)} + (1-\beta)\tau_i^{(t)}$. When adversarial variance $\operatorname{Var}(\{\hat{s}_j\})$ exceeds threshold bounds, the temperature $\tau^{(t)}$ automatically sharpens, expediting the permanent exclusion of persistent Byzantine saboteurs.
4. **Fairness via Rawlsian Egalitarian Social Welfare**:
   Specifically optimizes the welfare of the most vulnerable agent ($\max \min_i \mathcal{U}_i$), raising the bottom-10% client performance and eliminating the "starvation" of tail agents common in standard FL.

### 3. Empirical Highlights
- **Statistical Benchmarks**: Evaluated on high-cardinality CIFAR-100 ($C=100$) across 5 distinct partition regimes (Uniform IID, Mild $\alpha=1.0$, Moderate $\alpha=0.5$, Severe $\alpha=0.1$, Extreme $\alpha=0.05$).
- **Baselines**: Compared against canonical paradigms including FedAvg (global consensus), FedRep (split-head), and Ditto (dual-model regularization).
- **Byzantine Robustness**: Evaluated against Label-Flipping ($y \mapsto 99 - y$) and Sign-Flipping attacks ($q \in [0.0, 0.4]$).
- **Physical Edge Efficiency**: 78.3% parameter transfer reduction; tested on MobileNetV3 and ResNet-9 with edge latency/memory profiling.
- **Cross-Domain Generalization**: Zero-shot architectural transfer to continuous multi-sensor time-series regression ($R^2 = 99.95\%$).
