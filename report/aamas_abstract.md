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
*(Word Count: 182 words — strictly under 250 words)*

```text
Federated learning (FL) on edge devices faces three practical challenges: (1) statistical heterogeneity causes global models like FedAvg to suffer client drift; (2) personalized approaches like Ditto duplicate entire networks on-device, doubling memory and compute requirements; and (3) standard Byzantine-robust aggregators mistake legitimate statistical skew for adversarial poisoning, penalizing honest specialized clients.

We propose HEP-FL (Hierarchical Ensemble Personalization in Federated Learning), a multi-scale residual framework with subspace-calibrated Byzantine defense. HEP-FL decomposes model parameters additively into global base, cluster-shared, and private local residuals over a single shared backbone, avoiding the memory overhead of dual-model personalization. To filter malicious updates without discarding specialized honest clients, HEP-FL measures directional alignment strictly within each client's active class subspace and updates client trust scores over communication rounds.

We evaluate HEP-FL across five Dirichlet non-IID regimes on CIFAR-100, coordinated Byzantine attacks (including sign-flipping and label corruption), a 50-client network with partial participation, and physical mobile edge profiling on MobileNetV3 and ResNet-9. HEP-FL preserves client specialization, defends against model poisoning, protects tail-client performance, and maintains low on-device memory and latency, providing an effective framework for heterogeneous federated edge systems.
```

---

## Formatted Abstract (Markdown / LaTeX)
*(Word Count: 191 words)*

Federated learning (FL) on edge devices faces three practical challenges: (1) statistical heterogeneity causes global models like FedAvg to suffer client drift; (2) personalized approaches like Ditto duplicate entire networks on-device, doubling memory and compute requirements; and (3) standard Byzantine-robust aggregators mistake legitimate statistical skew for adversarial poisoning, penalizing honest specialized clients.

We propose **HEP-FL** (*Hierarchical Ensemble Personalization in Federated Learning*), a multi-scale residual framework with subspace-calibrated Byzantine defense. HEP-FL decomposes model parameters additively into global base, cluster-shared, and private local residuals ($W_{\text{eff}, i} = W_0 + \Delta W_{c(i)} + \Delta W_{l, i}$) over a single shared backbone, avoiding the memory overhead of dual-model personalization. To filter malicious updates without discarding specialized honest clients, HEP-FL measures directional alignment strictly within each client's active class subspace and updates client trust scores over communication rounds.

We evaluate HEP-FL across five Dirichlet non-IID regimes on CIFAR-100, coordinated Byzantine attacks (including sign-flipping and label corruption), a 50-client network with partial participation, and physical mobile edge profiling on MobileNetV3 and ResNet-9. HEP-FL preserves client specialization, defends against model poisoning, protects tail-client performance, and maintains low on-device memory and latency, providing an effective framework for heterogeneous federated edge systems.

---

## Methodological Summary

### 1. Three Practical Challenges
1. **Client Drift**: Non-IID distributions degrade monolithic consensus models.
2. **On-Device Memory Bloat**: Dual-model architectures double parameter storage and backpropagation memory.
3. **Robustness vs. Heterogeneity Trap**: Standard robust filters penalize non-IID specialization as anomalies.

### 2. HEP-FL Core Mechanics
1. **Multi-Scale Additive Residuals**:
   $$W_{\text{eff}, i} = W_0 + \Delta W_{c(i)} + \Delta W_{l, i}$$
   Single shared feature extractor backbone with branched global, cluster, and local projection residuals.
2. **Subspace-Calibrated Directional Alignment**:
   Calculates directional similarity strictly over each client's active label mask ($\mathcal{Y}_i$), decoupling domain specialization from malicious gradient deviation.
3. **Temporal Trust Tracking**:
   Tracks moving-average client trust scores $T_i^{(t)} = \beta T_i^{(t-1)} + (1-\beta)\tau_i^{(t)}$ to isolate persistent Byzantine actors.

### 3. Evaluation Suite (outputs/suite_v2/)
- **Track 1**: CIFAR-100 across 5 Dirichlet regimes ($\alpha \in [\infty, 1.0, 0.5, 0.1, 0.05]$).
- **Track 2**: Byzantine matrix (Sign-Flipping, Label-Flipping, Gaussian Noise up to $q=0.3$).
- **Track 3**: 50-client network scaling with partial participation ($C_p=0.20$) and tail decile evaluation.
- **Track 4**: Component ablation study (backbone, cluster head, local head, temperature routing, trust momentum).
- **Track 5**: Physical mobile edge profiling (MobileNetV3-Small vs. ResNet-9 peak VRAM and latency).
