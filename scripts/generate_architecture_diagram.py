"""
Generate publication-quality system architecture diagram for FedHEP.
Covers all three tiers:
1. Edge Client Pipeline: Single Shared Backbone (Phi_theta), 3-Tier Heads (Root, Parent, Local),
   Active-Class Logit Masking (ACLM), Skew-Calibrated Bernstein Loss, 256-Dim Random Projection Sketching,
   and Calibrated Ensemble Inference.
2. Communication Interface: Downlink, Uplink, and Strict Privacy Boundary.
3. Federated Server Orchestrator: Layered Subspace Byzantine Defense (SCCF + TTT),
   Robust Consensus Aggregation, and Privacy-Preserving Collaborative Peer Cluster Discovery.

Exports both high-resolution vector PDF and 300-DPI raster PNG for paper manuscript inclusion.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def build_fedhep_architecture():
    # Set typography and rendering standards
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times", "serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Canvas dimensions: 18.0 x 10.4 inches
    fig, ax = plt.subplots(figsize=(18.0, 10.4), dpi=300)
    ax.set_xlim(0, 18.0)
    ax.set_ylim(0, 10.4)
    ax.axis("off")

    # Color Palette (Bang Wong inspired + modern scientific paper styling)
    c_client_bg = "#F8FAFD"
    c_client_border = "#2563EB"
    c_bus_bg = "#F8FAFC"
    c_bus_border = "#64748B"
    c_server_bg = "#FDFAF8"
    c_server_border = "#DC2626"

    # Component colors
    c_backbone_box = "#EFF6FF"
    c_backbone_border = "#1D4ED8"
    c_backbone_text = "#1E3A8A"

    c_root_box = "#ECFDF5"
    c_root_border = "#059669"
    c_root_text = "#064E3B"

    c_parent_box = "#FFFBEB"
    c_parent_border = "#D97706"
    c_parent_text = "#78350F"

    c_local_box = "#FFF1F2"
    c_local_border = "#E11D48"
    c_local_text = "#881337"

    c_aclm_box = "#F5F3FF"
    c_aclm_border = "#7C3AED"
    c_aclm_text = "#4C1D95"

    c_sketch_box = "#ECFEFF"
    c_sketch_border = "#0891B2"
    c_sketch_text = "#164E63"

    c_byz_box = "#FEF2F2"
    c_byz_border = "#B91C1C"
    c_byz_text = "#7F1D1D"

    c_consensus_box = "#F0FDF4"
    c_consensus_border = "#16A34A"
    c_consensus_text = "#14532D"

    c_cluster_box = "#FFF7ED"
    c_cluster_border = "#EA580C"
    c_cluster_text = "#7C2D12"

    c_ensemble_box = "#F1F5F9"
    c_ensemble_border = "#475569"
    c_ensemble_text = "#0F172A"

    # =========================================================================
    # 0. MAIN HEADER BANNER
    # =========================================================================
    banner_box = FancyBboxPatch((0.4, 9.60), 17.2, 0.65, boxstyle="round,pad=0.04,rounding_size=0.1",
                                facecolor="#0F172A", edgecolor="#0F172A", linewidth=1.0)
    ax.add_patch(banner_box)
    ax.text(9.0, 10.02, r"$\mathbf{FedHEP}$: Federated Hierarchical Ensemble Personalization",
            fontsize=14.0, fontweight="bold", ha="center", va="center", color="#FFFFFF")
    ax.text(9.0, 9.75, "Single-Backbone 3-Tier Representation  •  Active-Class Logit Masking  •  Privacy-Preserving Sketch Clustering  •  Layered Subspace Byzantine Defense (SCCF + TTT)",
            fontsize=9.2, fontstyle="italic", ha="center", va="center", color="#94A3B8")

    # =========================================================================
    # 1. PANEL 1: EDGE CLIENT ON-DEVICE PIPELINE
    # =========================================================================
    panel1 = FancyBboxPatch((0.4, 0.4), 7.4, 9.05, boxstyle="round,pad=0.08,rounding_size=0.15",
                            facecolor=c_client_bg, edgecolor=c_client_border, linewidth=2.0)
    ax.add_patch(panel1)
    ax.text(0.65, 9.20, "EDGE CLIENT ON-DEVICE PIPELINE", fontsize=12.2, fontweight="bold", color="#1D4ED8")
    ax.text(0.65, 8.95, "Single Shared Backbone  •  No Model Duplication  •  158.8 MB Peak VRAM (46.8% vs Ditto)",
            fontsize=8.4, fontstyle="italic", color="#3B82F6")

    # --- Panel 1 Top: Backbone & 3-Tier Heads ---
    # 1.1 Local Batch
    batch_box = FancyBboxPatch((0.6, 6.70), 1.6, 2.05, boxstyle="round,pad=0.05,rounding_size=0.08",
                               facecolor="#FFFFFF", edgecolor="#94A3B8", linewidth=1.4)
    ax.add_patch(batch_box)
    ax.text(1.4, 8.50, "Local Batch", fontsize=10.0, fontweight="bold", ha="center", color="#1E293B")
    ax.text(1.4, 8.05, r"$(X, Y) \sim \mathcal{D}_i$", fontsize=10.0, ha="center", color="#334155")
    ax.text(1.4, 7.55, r"$\mathcal{Y}_i \subset \{1 \dots C\}$", fontsize=9.2, ha="center", color="#0369A1")
    ax.text(1.4, 7.15, "Sparse Support", fontsize=8.0, fontstyle="italic", ha="center", color="#64748B")
    ax.text(1.4, 6.85, r"$|\mathcal{Y}_i| \ll C$ (Non-IID)", fontsize=8.0, ha="center", color="#64748B")

    # Arrow from Batch to Backbone
    ax.annotate("", xy=(2.45, 7.72), xytext=(2.2, 7.72),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#1D4ED8"))

    # 1.2 Shared Backbone
    bb_box = FancyBboxPatch((2.45, 6.55), 1.95, 2.20, boxstyle="round,pad=0.06,rounding_size=0.1",
                            facecolor=c_backbone_box, edgecolor=c_backbone_border, linewidth=1.8)
    ax.add_patch(bb_box)
    ax.text(3.42, 8.50, "Shared Backbone", fontsize=10.2, fontweight="bold", ha="center", color=c_backbone_text)
    ax.text(3.42, 8.12, r"$\Phi_\theta(x)$ (ResNet-9)", fontsize=9.6, fontweight="bold", ha="center", color="#1D4ED8")
    ax.text(3.42, 7.55, "Single forward pass\ncomputed once", fontsize=8.2, ha="center", color="#2563EB")
    ax.text(3.42, 6.85, r"Embedding $\mathbf{h} \in \mathbb{R}^d$", fontsize=9.2, fontweight="bold", ha="center", color="#1E3A8A")

    # Arrows from Backbone to 3 Heads
    ax.annotate("", xy=(4.60, 8.35), xytext=(4.40, 7.90),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=c_root_border))
    ax.annotate("", xy=(4.60, 7.50), xytext=(4.40, 7.50),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=c_parent_border))
    ax.annotate("", xy=(4.60, 6.65), xytext=(4.40, 7.10),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=c_local_border))

    # 1.3 Tripartite Linear Heads (widened to 3.05 for clean margins)
    # (a) Root Head
    head_r = FancyBboxPatch((4.60, 8.00), 3.05, 0.78, boxstyle="round,pad=0.05,rounding_size=0.08",
                            facecolor=c_root_box, edgecolor=c_root_border, linewidth=1.5)
    ax.add_patch(head_r)
    ax.text(4.75, 8.52, r"1. Global Root Head $W_r \in \mathbb{R}^{d \times C}$",
            fontsize=8.8, fontweight="bold", color=c_root_text)
    ax.text(4.75, 8.18, r"Consensus anchor $\to \mathbf{z}_r$ (Unmasked CE)",
            fontsize=7.8, color="#047857")

    # (b) Parent Head
    head_p = FancyBboxPatch((4.60, 7.10), 3.05, 0.78, boxstyle="round,pad=0.05,rounding_size=0.08",
                            facecolor=c_parent_box, edgecolor=c_parent_border, linewidth=1.5)
    ax.add_patch(head_p)
    ax.text(4.75, 7.62, r"2. Cluster Parent Head $W_p \in \mathbb{R}^{d \times C}$",
            fontsize=8.8, fontweight="bold", color=c_parent_text)
    ax.text(4.75, 7.28, r"Collaborative pool $\to \mathbf{z}_p$ (ACLM Masked)",
            fontsize=7.8, color="#B45309")

    # (c) Local Head
    head_l = FancyBboxPatch((4.60, 6.20), 3.05, 0.78, boxstyle="round,pad=0.05,rounding_size=0.08",
                            facecolor=c_local_box, edgecolor=c_local_border, linewidth=1.5)
    ax.add_patch(head_l)
    ax.text(4.75, 6.72, r"3. Private Local Head $W_l \in \mathbb{R}^{d \times C}$",
            fontsize=8.8, fontweight="bold", color=c_local_text)
    ax.text(4.75, 6.38, r"Personalized head $\to \mathbf{z}_l$ (Strictly Private)",
            fontsize=7.8, color="#BE123C")

    # 1.4 Calibrated Ensemble Inference Card
    ens_box = FancyBboxPatch((0.6, 5.05), 7.05, 0.95, boxstyle="round,pad=0.05,rounding_size=0.08",
                             facecolor=c_ensemble_box, edgecolor=c_ensemble_border, linewidth=1.4)
    ax.add_patch(ens_box)
    ax.text(0.75, 5.72, "Temperature-Calibrated Ensemble Inference:", fontsize=8.8, fontweight="bold", color=c_ensemble_text)
    ax.text(0.75, 5.30,
            r"$\mathbf{z}_{\mathrm{ens}} = w_l \frac{\mathbf{z}_{i,l}}{T_l} + w_p \frac{\mathbf{z}_{i,p}}{T_p} + w_r \frac{\mathbf{z}_{i,r}}{T_r}$",
            fontsize=9.0, color="#1E293B")
    ax.text(4.60, 5.30,
            r"$\Longrightarrow \quad \hat{y} = \mathrm{arg\,max}_{c \in \mathcal{Y}} z_{\mathrm{ens}, c}$",
            fontsize=9.0, fontweight="bold", color="#0F172A")
    ax.text(0.75, 5.10, "Preserves prediction sharpness and eliminates probability dilution without dual-model memory overhead",
            fontsize=7.2, fontstyle="italic", color="#475569")

    # --- Panel 1 Bottom: ACLM & Loss Weighting (Left) + RP Sketching (Right) ---
    # 1.5 Active-Class Logit Masking (ACLM) Card
    aclm_box = FancyBboxPatch((0.6, 0.55), 3.90, 4.30, boxstyle="round,pad=0.06,rounding_size=0.1",
                              facecolor=c_aclm_box, edgecolor=c_aclm_border, linewidth=1.6)
    ax.add_patch(aclm_box)
    ax.text(0.75, 4.58, "Active-Class Logit Masking (ACLM)", fontsize=9.6, fontweight="bold", color=c_aclm_text)
    ax.text(0.75, 4.30, "Eliminates Negative Gradient Interference:", fontsize=8.2, fontweight="bold", color="#6D28D9")
    ax.text(0.85, 3.95,
            r"$\tilde{z}_{i,c} = z_{i,c} \;\; (\mathrm{if} \; c \in \mathcal{Y}_i), \quad \tilde{z}_{i,c} = -\infty \;\; (\mathrm{if} \; c \notin \mathcal{Y}_i)$",
            fontsize=8.0, color="#4C1D95")
    ax.text(0.85, 3.65,
            r"$\Longrightarrow \quad \frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial z_{i,c}} = 0 \quad (\forall c \notin \mathcal{Y}_i)$",
            fontsize=8.4, fontweight="bold", color="#3B0764")
    ax.text(0.75, 3.32, r"• Shields shared backbone $\Phi_\theta$ from unobserved penalty", fontsize=7.6, color="#5B21B6")
    ax.text(0.75, 3.08, "• Eliminates gradient corruption under extreme skew", fontsize=7.6, color="#5B21B6")

    ax.text(0.75, 2.75, "Skew-Calibrated Bernstein Loss Weighting:", fontsize=8.2, fontweight="bold", color="#6D28D9")
    ax.text(0.85, 2.42,
            r"$R_{\mathrm{skew}}^{(i)} = 1 - \frac{H(\mathbf{p}_i)}{\ln C} \in [0, 1] \quad (\mathrm{Entropy\;Skew})$",
            fontsize=8.2, color="#4C1D95")
    ax.text(0.85, 2.05,
            r"$\lambda_r = a_i + (1-a_i)(R_{\mathrm{skew}}^{(i)})^2, \; \lambda_p = 2 R_{\mathrm{skew}}^{(i)}(1-R_{\mathrm{skew}}^{(i)})$",
            fontsize=7.8, color="#5B21B6")
    ax.text(0.85, 1.70,
            r"$\lambda_l = (1-R_{\mathrm{skew}}^{(i)})^2, \quad a_i = \max\left(\frac{1}{2K}, \frac{|\mathcal{Y}_i|}{C}\right)$",
            fontsize=7.8, color="#5B21B6")
    ax.text(0.75, 1.30, "Composite Local Training Objective:", fontsize=8.2, fontweight="bold", color="#6D28D9")
    ax.text(0.85, 0.92,
            r"$\mathcal{L}_{\mathrm{total}} = \alpha_r \mathcal{L}_r + \alpha_p \mathcal{L}_p^{\mathrm{mask}} + \alpha_l \mathcal{L}_l^{\mathrm{mask}}$",
            fontsize=8.6, fontweight="bold", color="#3B0764")
    ax.text(0.75, 0.68, "Smooth partition-of-unity dynamic adaptation", fontsize=7.2, fontstyle="italic", color="#7C3AED")

    # 1.6 Random Projection Sketching Card
    sk_box = FancyBboxPatch((4.65, 0.55), 3.00, 4.30, boxstyle="round,pad=0.06,rounding_size=0.1",
                            facecolor=c_sketch_box, edgecolor=c_sketch_border, linewidth=1.6)
    ax.add_patch(sk_box)
    ax.text(4.80, 4.58, "Random Projection (RP)", fontsize=9.6, fontweight="bold", color=c_sketch_text)
    ax.text(4.80, 4.30, "Privacy-Preserving Sketching:", fontsize=8.2, fontweight="bold", color="#0E7490")
    ax.text(4.80, 3.85, r"• Parent Delta: $\Delta W_{p,i} \in \mathbb{R}^{d \cdot C}$", fontsize=8.0, color="#155E75")
    ax.text(4.80, 3.45, r"• Shared Gaussian Matrix:", fontsize=8.0, fontweight="bold", color="#0E7490")
    ax.text(4.95, 3.12, r"$\mathbf{R} \in \mathbb{R}^{256 \times (d \cdot C)}, \quad R_{jk} \sim \mathcal{N}(0, 1/m)$", fontsize=7.8, color="#164E63")
    ax.text(4.80, 2.70, r"• Low-Dim Normalized Sketch:", fontsize=8.0, fontweight="bold", color="#0E7490")
    ax.text(4.95, 2.30, r"$\mathbf{s}_i = \frac{\mathbf{R} \Delta W_{p,i}}{\|\mathbf{R} \Delta W_{p,i}\|_2} \in \mathbb{R}^{256}$", fontsize=8.6, fontweight="bold", color="#083344")
    ax.text(4.80, 1.82, "• Information-Theoretic Privacy:", fontsize=8.0, fontweight="bold", color="#0E7490")
    ax.text(4.95, 1.50, r"Underdetermined system ($m \ll D$)" + "\nprevents raw parameter inversion", fontsize=7.4, color="#155E75")
    ax.text(4.80, 1.05, "• Ultra-Lean Transmission:", fontsize=8.0, fontweight="bold", color="#0E7490")
    ax.text(4.95, 0.75, r"Payload: $\mathbf{1.0\ \mathrm{KB}}$ / round (10$\times$ comp.)", fontsize=7.8, fontweight="bold", color="#0E7490")

    # =========================================================================
    # 2. PANEL 2: COMMUNICATION BUS & PRIVACY BOUNDARY
    # =========================================================================
    panel2 = FancyBboxPatch((7.95, 0.4), 2.75, 9.05, boxstyle="round,pad=0.08,rounding_size=0.15",
                            facecolor=c_bus_bg, edgecolor=c_bus_border, linewidth=1.8, linestyle="--")
    ax.add_patch(panel2)
    ax.text(9.32, 9.20, "COMMUNICATION BUS", fontsize=11.2, fontweight="bold", ha="center", color="#334155")
    ax.text(9.32, 8.95, "Synchronous Federated Rounds", fontsize=8.2, fontstyle="italic", ha="center", color="#64748B")

    # 2.1 Downlink Box
    down_box = FancyBboxPatch((8.10, 6.75), 2.45, 2.00, boxstyle="round,pad=0.05,rounding_size=0.08",
                              facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.4)
    ax.add_patch(down_box)
    ax.text(9.32, 8.48, "Downlink Broadcast", fontsize=9.0, fontweight="bold", ha="center", color="#1D4ED8")
    ax.text(9.32, 8.08, r"Global Backbone: $\theta^{(t)}$", fontsize=8.2, ha="center", color="#1E40AF")
    ax.text(9.32, 7.68, r"Global Root Head: $W_r^{(t)}$", fontsize=8.2, ha="center", color="#065F46")
    ax.text(9.32, 7.28, r"Cluster Parent Head: $W_{p,k(i)}^{(t)}$", fontsize=8.2, ha="center", color="#92400E")
    ax.text(9.32, 6.90, "(Synchronized Peer Model)", fontsize=7.0, fontstyle="italic", ha="center", color="#6B7280")

    # Downlink arrows (Server -> Bus -> Client, pointing LEFT)
    ax.annotate("", xy=(7.65, 7.75), xytext=(8.10, 7.75),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#2563EB"))
    ax.annotate("", xy=(10.55, 7.75), xytext=(10.90, 7.75),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#2563EB"))

    # 2.2 Uplink Box
    up_box = FancyBboxPatch((8.10, 3.85), 2.45, 2.70, boxstyle="round,pad=0.05,rounding_size=0.08",
                            facecolor="#ECFDF5", edgecolor="#10B981", linewidth=1.4)
    ax.add_patch(up_box)
    ax.text(9.32, 6.28, "Uplink Telemetry", fontsize=9.0, fontweight="bold", ha="center", color="#065F46")
    ax.text(9.32, 5.88, r"Backbone Delta: $\Delta \theta_i$", fontsize=8.2, ha="center", color="#047857")
    ax.text(9.32, 5.50, r"Root Delta: $\Delta W_{r,i}$", fontsize=8.2, ha="center", color="#047857")
    ax.text(9.32, 5.12, r"Parent Delta: $\Delta W_{p,i}$", fontsize=8.2, ha="center", color="#B45309")
    ax.text(9.32, 4.70, r"RP Sketch: $\mathbf{s}_i \in \mathbb{R}^{256}$", fontsize=8.5, fontweight="bold", ha="center", color="#0E7490")
    ax.text(9.32, 4.30, r"Active Support: $\mathcal{Y}_i \subset \{1..C\}$", fontsize=8.0, ha="center", color="#374151")
    ax.text(9.32, 4.00, "(Used for SCCF Defense)", fontsize=7.0, fontstyle="italic", ha="center", color="#6B7280")

    # Uplink arrows (Client -> Bus -> Server, pointing RIGHT)
    ax.annotate("", xy=(8.10, 5.20), xytext=(7.65, 5.20),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#059669"))
    ax.annotate("", xy=(10.90, 5.20), xytext=(10.55, 5.20),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#059669"))

    # 2.3 Strict Privacy Seal
    priv_box = FancyBboxPatch((8.10, 0.55), 2.45, 3.10, boxstyle="round,pad=0.05,rounding_size=0.08",
                              facecolor="#FFF1F2", edgecolor="#F43F5E", linewidth=1.6, linestyle=":")
    ax.add_patch(priv_box)
    ax.text(9.32, 3.35, "PRIVACY BOUNDARY", fontsize=8.8, fontweight="bold", ha="center", color="#BE123C")
    ax.text(9.32, 2.95, r"Local Head $W_{l,i}$", fontsize=9.2, fontweight="bold", ha="center", color="#881337")
    ax.text(9.32, 2.50, "NEVER UPLOADED", fontsize=9.2, fontweight="bold", ha="center", color="#E11D48")
    ax.text(9.32, 2.05, "100% On-Device Isolation\nRaw Data $\\mathcal{D}_i$ Protected\nZero Telemetry Leakage",
            fontsize=7.6, ha="center", color="#9F1239")
    ax.text(9.32, 0.95, "Safe against gradient inversion\nand eavesdropping",
            fontsize=7.0, fontstyle="italic", ha="center", color="#BE123C")

    # =========================================================================
    # 3. PANEL 3: FEDERATED SERVER ORCHESTRATOR
    # =========================================================================
    panel3 = FancyBboxPatch((10.90, 0.4), 6.70, 9.05, boxstyle="round,pad=0.08,rounding_size=0.15",
                            facecolor=c_server_bg, edgecolor=c_server_border, linewidth=2.0)
    ax.add_patch(panel3)
    ax.text(11.15, 9.20, "FEDERATED SERVER ORCHESTRATOR", fontsize=12.2, fontweight="bold", color="#B91C1C")
    ax.text(11.15, 8.95, "Layered Subspace Defense (SCCF + TTT)  •  Robust Consensus  •  Peer Clustering",
            fontsize=8.4, fontstyle="italic", color="#EF4444")

    # 3.1 Layered Subspace Byzantine Defense Card (Top)
    byz_box = FancyBboxPatch((11.10, 5.75), 6.30, 3.00, boxstyle="round,pad=0.06,rounding_size=0.1",
                             facecolor=c_byz_box, edgecolor=c_byz_border, linewidth=1.8)
    ax.add_patch(byz_box)
    ax.text(11.25, 8.48, "Layered Subspace Byzantine Defense (SCCF + TTT)", fontsize=10.0, fontweight="bold", color=c_byz_text)
    ax.text(11.25, 8.22, "Attacks Neutralized: Label Flip  •  Sign Flip  •  Gradient Ascent  •  Gaussian Noise",
            fontsize=7.6, fontstyle="italic", color="#991B1B")

    # Defense Step 1: SCCF
    ax.text(11.25, 7.85, "1. Subspace-Constrained Cosine Filtering (SCCF):", fontsize=8.4, fontweight="bold", color="#991B1B")
    ax.text(11.45, 7.42,
            r"$S_{ij}^{\mathrm{SCCF}} = \frac{\langle \Delta \theta_i \odot \mathbf{M}_{ij}, \, \Delta \theta_j \odot \mathbf{M}_{ij} \rangle}{\|\Delta \theta_i \odot \mathbf{M}_{ij}\|_2 \, \|\Delta \theta_j \odot \mathbf{M}_{ij}\|_2}, \quad \mathbf{M}_{ij} = \mathbb{I}(\mathcal{Y}_i \cap \mathcal{Y}_j \neq \emptyset)$",
            fontsize=7.8, color="#7F1D1D")
    ax.text(11.45, 7.02, "• Projects cosine similarity strictly onto observed label overlap subspace", fontsize=7.6, color="#991B1B")
    ax.text(11.45, 6.78, "• Prevents false rejection of honest specialized clients under skew!", fontsize=7.6, fontweight="bold", color="#B91C1C")

    # Defense Step 2: Norm + TTT
    ax.text(11.25, 6.42, "2. Adaptive Norm Bounding & Temporal Trust Tracking (TTT):", fontsize=8.4, fontweight="bold", color="#991B1B")
    ax.text(11.45, 6.05,
            r"$\|\Delta \theta_i\|_2 \leq \tau_{\mathrm{clip}}, \quad \tau_i^{(t)} = \beta_\tau \tau_i^{(t-1)} + (1-\beta_\tau) s_i^{(t)}$",
            fontsize=8.0, color="#7F1D1D")
    ax.text(11.45, 5.82, "Multi-round historical momentum isolates insidious poisoning over time", fontsize=7.4, fontstyle="italic", color="#991B1B")

    # 3.2 Robust Global Consensus Aggregation Card (Middle)
    cons_box = FancyBboxPatch((11.10, 3.85), 6.30, 1.75, boxstyle="round,pad=0.06,rounding_size=0.1",
                              facecolor=c_consensus_box, edgecolor=c_consensus_border, linewidth=1.6)
    ax.add_patch(cons_box)
    ax.text(11.25, 5.38, "Robust Consensus Aggregation Engine", fontsize=9.6, fontweight="bold", color=c_consensus_text)
    ax.text(11.25, 4.95,
            r"$\theta^{(t+1)} \leftarrow \theta^{(t)} + \sum_{i \in \mathcal{H}} \tilde{\tau}_i \Delta \theta_i, \quad W_r^{(t+1)} \leftarrow W_r^{(t)} + \sum_{i \in \mathcal{H}} \tilde{\tau}_i \Delta W_{r,i}$",
            fontsize=8.6, fontweight="bold", color="#14532D")
    ax.text(11.25, 4.52, "• Trust-calibrated weighting $\\tilde{\\tau}_i$ shields global parameters from poisoning", fontsize=7.6, color="#166534")
    ax.text(11.25, 4.25, "• Preserves high consensus representations on homogeneous/IID distributions", fontsize=7.6, color="#166534")
    ax.text(11.25, 3.98, r"• Verified benchmark: $72.73\%$ IID accuracy on CIFAR-10 (40.94% on CIFAR-100)", fontsize=7.2, fontstyle="italic", color="#15803D")

    # 3.3 Privacy-Preserving Collaborative Peer Clustering Card (Bottom)
    clust_box = FancyBboxPatch((11.10, 0.55), 6.30, 3.15, boxstyle="round,pad=0.06,rounding_size=0.1",
                               facecolor=c_cluster_box, edgecolor=c_cluster_border, linewidth=1.6)
    ax.add_patch(clust_box)
    ax.text(11.25, 3.48, "Privacy-Preserving Collaborative Peer Clustering", fontsize=9.6, fontweight="bold", color=c_cluster_text)
    ax.text(11.25, 3.15, "1. Sketch Cosine Affinity Routing:", fontsize=8.4, fontweight="bold", color="#9A3412")
    ax.text(11.45, 2.82,
            r"Assign client $i \to \mathcal{C}_k \;\Longleftrightarrow\; k = \mathrm{arg\,max}_{j} \cos(\mathbf{s}_i, \boldsymbol{\mu}_j)$",
            fontsize=8.2, color="#7C2D12")
    ax.text(11.25, 2.45, "2. Momentum-Stabilized Cluster Parent Head Aggregation:", fontsize=8.4, fontweight="bold", color="#9A3412")
    ax.text(11.45, 2.12,
            r"$W_{p,k}^{(t+1)} \leftarrow \beta_{c} W_{p,k}^{(t)} + (1-\beta_{c}) \bar{W}_{p,k}, \quad \bar{W}_{p,k} = \sum_{i \in \mathcal{C}_k} \frac{n_i}{N_k}(\Delta W_{p,i} + W_{p,k}^{(t)})$",
            fontsize=7.6, color="#7C2D12")

    # Cluster Cards (K=3) (properly positioned and padded)
    cluster_configs = [
        ("Cluster 1", r"Head $W_{p,1}$", "Cohort $\\mathcal{C}_1$", "#FEF3C7", "#D97706"),
        ("Cluster 2", r"Head $W_{p,2}$", "Cohort $\\mathcal{C}_2$", "#FFEDD5", "#EA580C"),
        ("Cluster 3", r"Head $W_{p,3}$", "Cohort $\\mathcal{C}_3$", "#FEE2E2", "#DC2626")
    ]
    for idx, (c_label, c_head, c_sub, bg, fg) in enumerate(cluster_configs):
        c_x = 11.35 + idx * 2.02
        c_card = FancyBboxPatch((c_x, 0.72), 1.85, 1.15, boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor=bg, edgecolor=fg, linewidth=1.4)
        ax.add_patch(c_card)
        ax.text(c_x + 0.925, 1.60, c_label, fontsize=8.6, fontweight="bold", ha="center", color=fg)
        ax.text(c_x + 0.925, 1.25, c_head, fontsize=8.8, fontweight="bold", ha="center", color="#7C2D12")
        ax.text(c_x + 0.925, 0.92, c_sub, fontsize=7.6, fontstyle="italic", ha="center", color="#4B5563")

    # Output paths: both vector PDF and 300 DPI PNG
    out_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures", "architecture.pdf"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures", "architecture.png"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report", "figures", "hep_architecture.png"),
    ]

    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if p.endswith(".pdf"):
            fig.savefig(p, format="pdf", bbox_inches="tight")
        else:
            fig.savefig(p, format="png", dpi=300, bbox_inches="tight")
        print(f"[ok] Generated {p}")

    plt.close(fig)

if __name__ == "__main__":
    build_fedhep_architecture()
