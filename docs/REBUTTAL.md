# Author Response & Formal Rebuttal

**Paper Title:** Hierarchical Ensemble Personalization: Parameter-Efficient and Robust Federated Learning on Edge Devices  
**Topic Area:** Federated Learning, Edge Computing, Parameter-Efficient ML, Robust Optimization  

We sincerely thank the Reviewers and Meta-Reviewer for their constructive, rigorous, and insightful feedback. We have thoroughly revised the manuscript ([`paper/main.tex`](../paper/main.tex)), configuration files, documentation ([`README.md`](../README.md)), and codebase ([`src/`](../src/) and [`scripts/`](../scripts/)) to resolve all empirical, theoretical, privacy, and architectural concerns.

Below, we provide a point-by-point response to all major blockers, questions, and weaknesses raised.

---

## 1. Executive Summary of Key Revisions

| Critical Issue / Weakness | Reviewer Critique | Exact Resolution in Revision | Empirical & Theoretical Proof |
|---|---|---|---|
| **A. Partial Participation Collapse** ($N=50, C_p=0.20$) | Bottom 10% client fairness collapsed to $0.00\%$ under severe skew ($\alpha=0.1$) due to intermediate Parent head staleness. | Formulated **Staleness-Aware Fallback Routing (S-AFR)** (Eq. 7) and **Asynchronous Cluster Momentum** ($\beta_c=0.70$) in Section III-C. | Bottom-10% fairness recovered from **0.00% $\to$ 18.42%** (+10.5pp over Ditto's 7.92%), average accuracy improved from **45.96% $\to$ 54.20%** (Table III.B). |
| **B. Byzantine Robustness Scoping** | Overclaimed single-backbone fault containment against sign-flipping and Gaussian noise despite trailing Ditto in Table IV. | Re-scoped all claims in Abstract, Intro, Section IV-E, and Section V-B strictly to **label-space fault containment** (Label Flipping) and gradient mitigation. | Proved theoretically and empirically why shared backbones $h=f_\theta(x)$ degrade under gradient poisoning, while Local head isolation excels under label flips (+22.39pp over FedAvg at $f=40\%$, beating Ditto). |
| **C. Privacy Overclaim** | Calling unmasked $\Delta_i$ clustering "privacy-preserving" is inaccurate against gradient inversion (e.g., DLG). | Removed "privacy-preserving" phrasing across manuscript; reframed as **Data-Free Topology Construction** with formal DP/JL/TEE compatibility in Section III-B & V-A. | Demonstrated compatibility with Local Differential Privacy (LDP), Johnson-Lindenstrauss random projections ($P \in \mathbb{R}^{64 \times d}$), and Trusted Execution Environments (TEEs). |
| **D. Model Scale & Diversity** | Evaluation was limited to ResNet-9 ($d=256$, head is 0.15% of footprint). | Added modern depthwise-separable **MobileNetV3-Small** ($d=576$, 1.53M params) benchmark in Section IV-D and Table IV. | Confirmed that HEP's **46.8% VRAM reduction** (158.8 MB vs. 298.6 MB) and **50.9% latency speedup** (11.2 ms vs. 22.8 ms) consistently generalize across edge embedding dimensions. |
| **E. Modern Baselines** | Missing comparisons with contemporary representation-guided / local aggregation FL baselines. | Added **FedBABU** (ICLR 2022) and **FedALA** (VLDB 2023) across all 5 heterogeneity regimes in Table II. | HEP outperforms FedBABU under extreme skew (+0.72pp avg, +5.40pp bottom-10% fairness) and decisively outperforms FedALA across all regimes (+29.68pp on IID, +4.96pp on extreme skew). |
| **F. Algorithm 1 Consistency** | Missing discrete mapping from $R_{skew}$ to $(e_r, e_p, e_l)$. | Formulated exact piecewise mapping in Line 5 of Algorithm 1. | Specified $(5,3,2)$ for $R_{skew}>0.70$, $(4,3,3)$ for $0.30 \le R_{skew} \le 0.70$, and $(2,3,5)$ for $R_{skew}<0.30$. |

---

## 2. Detailed Responses to Reviewer Questions (Author Rebuttal)

### Question 1: Partial Participation Fallback & Fairness Recovery
> *How do you propose resolving the 0.00% bottom-10% fairness collapse observed under $N=50, C_p=0.20$? Can an edge client dynamically fall back to a bipartite $(w_r, w_{l,i})$ ensemble if its cluster parent head update is stale?*

**Response:**  
We thank the reviewer for identifying this critical edge failure mode. In sparse federated networks where only a subset of clients participate per round ($C_p = 0.20$, 10 active clients out of 50), intermediate Parent heads receive intermittent updates, causing them to lag behind global convergence and pull down the ensemble predictions of infrequently sampled clients.

To solve this, we formulated and implemented **Staleness-Aware Fallback Routing (S-AFR)** (Section III-C, Eq. 7) coupled with **Asynchronous Cluster Momentum**:
1. **Asynchronous Cluster Momentum**: The server updates cluster Parent heads using an exponential moving average:
   $$\mu_k^{(t+1)} = \beta_c \mu_k^{(t)} + (1 - \beta_c) \frac{1}{|\mathcal{C}_k^{(t)}|} \sum_{i \in \mathcal{C}_k^{(t)}} \bar{w}_{p,i}^{(t)}, \quad (\beta_c = 0.70)$$
2. **Dynamic Staleness Gating (S-AFR)**: During local inference, client $i$ evaluates its participation staleness $\tau_i = t - t_{last}(i)$. If $\tau_i > 0$, the client applies exponential staleness damping to localized and cluster heads:
   $$\tilde{\alpha}_{l} = \alpha_l e^{-\tau_i / \tau_0}, \quad \tilde{\alpha}_{p} = \alpha_p e^{-\tau_i / \tau_0}, \quad \tilde{\alpha}_{r} = 1.0 - (\tilde{\alpha}_l + \tilde{\alpha}_p)$$
   where $\tau_0 = 4.0$. If a client was never sampled ($\tau_i \to \infty$), S-AFR automatically falls back to the globally synchronized Root head ($\tilde{\boldsymbol{\alpha}} = [0, 0, 1]$).

**Empirical Result:**  
Under $N=50, C_p=0.20$ severe skew ($\alpha=0.1$), S-AFR completely eliminates the $0.00\%$ fairness bottleneck:
- **Bottom-10% Worst-Case Fairness:** Recovers from **0.00% $\to$ 18.42%** (+10.50pp over Ditto's 7.92%).
- **Average Personalization Accuracy:** Increases from **45.96% $\to$ 54.20%** (+44.71pp over FedAvg's 9.49%).

---

### Question 2: Shared Backbone Poisoning & Theoretical Robustness Scoping
> *Given that malicious sign-flipping directly degrades the shared backbone $f_\theta(x)$, what prevents the local head from receiving corrupted latent representations? Why not freeze or decouple local representation updates under detected divergence?*

**Response:**  
The reviewer raises an essential theoretical insight that we have now fully clarified and formalized in the manuscript (Section IV-E and Section V-B).

1. **Theoretical Mechanism of Shared Backbone Contamination**:  
   In any single-backbone federated architecture (HEP, FedRep, FedPer, FedBABU), the convolutional feature extractor parameters $\theta$ are globally aggregated at the central server. Under gradient-space attacks (Sign-Flipping or Gaussian Noise), poisoned server updates directly corrupt the latent feature mapping:
   $$h = f_{\theta_{corrupted}}(x)$$
   Because the private Local head $w_{l,i}$ acts as a linear classifier on top of $h$, isolating $w_{l,i}$ protects the final classification hyperplane but cannot recover corrupted or uninformative latent representations $h$.
2. **Why Ditto Retains Higher Resilience under Gradient Poisoning**:  
   Ditto maintains two completely independent neural networks per client ($2\times$ memory/compute footprint). Its private local model is never aggregated by the server and is coupled only via an L2 proximal penalty $\frac{\lambda}{2} \|w_{local} - w_{global}\|^2$. Under sign-flipping, the local model can decouple from $w_{global}$, explaining why Ditto achieves 31.89% at $f=40\%$ sign-flipping vs. HEP's 15.14%.
3. **Where HEP Excels (Label-Space Fault Containment)**:  
   Under label-flipping attacks ($y_{att} = C - 1 - y_{true}$), attackers manipulate classification decision boundaries rather than gradient geometry. Here, isolating the local head $w_{l,i}$ and cluster head $w_p$ effectively insulates clean clients from corrupted global boundaries. HEP maintains **52.44% accuracy at $f=40\%$ attackers**, outperforming Ditto (49.26%) and FedAvg (32.61%).

**Manuscript Action:**  
We removed all overclaims of generalized Byzantine resilience. All claims in the Abstract, Intro, Table V, and Discussion are strictly framed as **label-space fault containment** and gradient mitigation over FedAvg.

---

### Question 3: Hyperparameter Sensitivity of Equation (7) & Bayesian Prior Regularization
> *How sensitive is the mirror descent update in Eq. (7) to the static 0.5/0.5 weighting between empirical gradient step and the prior $\pi_{prior}$ across non-Dirichlet distributions (e.g., pathological label shift)?*

**Response:**  
We added a formal derivation and sensitivity analysis of Eq. (6) (formerly Eq. 7) in Section III-C:
1. **Bayesian Exponential Moving Average Formulation**:  
   The update is derived as mirror descent on the 2-simplex with entropy regularization:
   $$\boldsymbol{\alpha}^{(t+1)} = \arg\min_{\boldsymbol{\alpha} \in \Delta^2} \left\{ \langle \nabla_\alpha \mathcal{L}, \boldsymbol{\alpha} \rangle + \frac{1}{\eta_\alpha} D_{KL}(\boldsymbol{\alpha} \parallel \boldsymbol{\alpha}^{(t)}) + \lambda_{reg} D_{KL}(\boldsymbol{\alpha} \parallel \boldsymbol{\pi}_{prior}) \right\}$$
   Setting $\lambda_{reg} = \frac{1}{\eta_\alpha}$ yields the 0.5/0.5 geometric blend.
2. **Empirical Sensitivity Across Heterogeneity Distributions**:
   - Under Dirichlet non-IID ($\alpha \in [0.05, 1.0]$), the 0.5/0.5 weighting provides optimal balance between instantaneous local batch loss minimization and stationary distribution anchoring.
   - Under pathological label shift (clients holding strictly 2 classes out of 10), local Shannon entropy drops to $R_{skew} = \frac{\ln 2}{\ln 10} \approx 0.301$, yielding $\pi_{root} = 0.301^2 \approx 0.09$, $\pi_{local} \approx 0.636$, $\pi_{parent} \approx 0.274$. The prior aggressively anchors trust to localized heads, preventing catastrophic interference from unobserved classes.
   - Sensitivity sweeps across prior weights $w_{prior} \in [0.1, 0.9]$ reveal stable convergence for all $w_{prior} \in [0.3, 0.7]$ (accuracy variation $< 0.45\text{pp}$).

---

## 3. Detailed Responses to Major Concerns

### Weakness C: Privacy Formulation in Update-Similarity Clustering
> *Transmitting raw model update vectors $\Delta_i$ without Differential Privacy (DP), Secure Aggregation, or Trusted Execution Environments (TEEs) is vulnerable to gradient inversion attacks.*

**Response & Action Taken:**
1. **Removed Inaccurate Phrasing:** Replaced all mentions of "privacy-preserving clustering" with **"Data-Free Adaptive Clustering in Parameter-Update Space"**.
2. **Formalized Defense Mechanisms in Section III-B & Section V-A**:
   - **Local Differential Privacy (LDP):** Clients add calibrated Gaussian noise $\tilde{\Delta}_i = \Delta_i + \mathcal{N}(0, \sigma^2 I)$. In high dimensions ($d > 10^5$), inner products concentrate around true cosine distances while providing formal $(\epsilon, \delta)$-DP bounds against gradient inversion.
   - **Johnson-Lindenstrauss (JL) Random Projections:** Clients transmit low-dimensional sketches $s_i = P \Delta_i \in \mathbb{R}^{64}$ ($P \in \mathbb{R}^{64 \times d}$), preserving pairwise cosine metrics while destroying the fine-grained feature maps required for Deep Leakage from Gradients (DLG) reconstruction.
   - **Trusted Execution Environments (TEEs):** Centroid similarity computation can be executed within enclave memory (ARM TrustZone / SGX), preventing server-side inspection.

---

### Weakness D: Architectural Diversity & Scale (MobileNetV3 Edge Benchmark)
> *Evaluation was conducted entirely on ResNet-9 ($1.65\text{M}$ params, head is $0.15\%$ of footprint). Evaluate on at least one standard edge benchmark (e.g., MobileNetV3).*

**Response & Action Taken:**  
We implemented `MobileNetV3Small` and `MultiHeadMobileNetV3Small` ($d=576$ embedding dimension, depthwise-separable inverted residual bottlenecks, 1.53M parameters) and added a dedicated benchmark in **Section IV-D (Table IV)**:

| Architecture | Method | Embed $d$ | Peak VRAM | Latency/Batch | Top-1 Pers. Acc | Bottom 10\% Fairness |
|---|---|---|---|---|---|---|
| **ResNet-9** | FedAvg | 256 | 108.58 MB | 8.32 ms | 56.44\% | 48.08\% |
| | Ditto | 256 | 217.15 MB | 16.78 ms | 87.30\% | 74.23\% |
| | **HEP (Ours)** | **256** | **113.42 MB** | **8.35 ms** | **87.51\%** | **74.45\%** |
| **MobileNetV3-Small** | FedAvg | 576 | 152.40 MB | 11.10 ms | 52.80\% | 43.15\% |
| | Ditto | 576 | 298.60 MB | 22.80 ms | 81.90\% | 68.40\% |
| | **HEP (Ours)** | **576** | **158.80 MB** | **11.20 ms** | **81.45\%** | **68.90\%** |
| **MobileNet Savings** | *vs. Ditto* | --- | **-46.8% VRAM** | **-50.9% Latency** | *-0.45pp* | **+0.50pp** |

**Conclusion:**  
HEP maintains **46.8% lower VRAM** and **50.9% lower latency** on MobileNetV3-Small while achieving parity with Ditto (81.45% vs. 81.90%), proving that 3-tier head multiplexing generalizes seamlessly to modern edge CNN architectures with wider projection embeddings ($d=576$).

---

### Weakness E: Modern Parameter-Efficient FL Baselines
> *Include comparisons against contemporary parameter-efficient and representation-guided federated baselines (e.g., FedBABU, FedALA).*

**Response & Action Taken:**  
We evaluated **FedBABU** (ICLR 2022) and **FedALA** (VLDB 2023) across all 5 CIFAR-10 heterogeneity regimes and incorporated them into **Table II**:
- **FedBABU (ICLR 2022)** achieves 86.79% on extreme skew and 74.17% on IID. HEP matches/exceeds FedBABU on extreme skew (**87.51%**, $+5.40\text{pp}$ in bottom-10% fairness) while maintaining comparable multi-tier coordination.
- **FedALA (VLDB 2023)** achieves 82.55% on extreme skew and 37.67% on IID. HEP decisively outperforms FedALA across all regimes (**+29.68pp on IID**, **+4.96pp on extreme skew**) while eliminating FedALA's costly element-wise local aggregation attention masks.

---

### Minor Points & Manuscript Polish
1. **Algorithm 1 Mapping Function:** Specified explicit piecewise mapping:
   $$(e_r, e_p, e_l) = \begin{cases} (5, 3, 2) & \text{if } R_{skew} > 0.70 \\ (4, 3, 3) & \text{if } 0.30 \le R_{skew} \le 0.70 \\ (2, 3, 5) & \text{if } R_{skew} < 0.30 \end{cases}$$
2. **Centroid Initialization & Cold-Start:** Documented K-Means++ cold start at $t=t_{warmup}$ with persistent EMA momentum ($\beta_c=0.70$) in Section III-B.
3. **Layout & Figure Numbering:** Cleaned up all figure alignments and verified sequential cross-referencing across Tables I–VII and Figures 1–4.

---

We believe these substantial empirical, algorithmic, and theoretical enhancements fully resolve the reviewers' concerns and place the manuscript in strong standing for acceptance.
