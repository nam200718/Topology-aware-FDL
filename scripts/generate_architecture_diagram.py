"""
Generate publication-quality system architecture diagram for Hierarchical Ensemble Personalization (HEP).
Strictly adheres to the zero-overlap rule with generous margins, clean orthogonal arrows, and distinct visual cards.

Outputs:
- report/figures/hep_architecture.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def create_architecture_diagram():
    # Width 18.0, Height 10.2 provides ample breathing room
    fig, ax = plt.subplots(figsize=(18.0, 10.2), dpi=300)
    ax.set_xlim(0, 18.0)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    # Color Palette
    c_bg_client = "#F4F8FC"
    c_bg_server = "#FAF5F5"
    c_client_border = "#90CAF9"
    c_server_border = "#EF9A9A"
    c_bus_border = "#B0BEC5"
    c_bg_bus = "#F5F7FA"

    c_backbone = "#0D47A1"
    c_root = "#1B5E20"       # Green
    c_parent = "#E65100"     # Orange
    c_local = "#880E4F"      # Crimson / Maroon
    c_analytics = "#4A148C"  # Deep Purple
    c_sketch = "#006064"     # Dark Cyan / Teal

    # =========================================================================
    # 1. TOP-LEVEL CONTAINERS
    # =========================================================================

    # Client Container: x in [0.5, 7.8], y in [0.4, 9.7]
    client_box = FancyBboxPatch((0.5, 0.4), 7.4, 9.3, boxstyle="round,pad=0.08,rounding_size=0.18",
                                facecolor=c_bg_client, edgecolor=c_client_border, linewidth=2.0)
    ax.add_patch(client_box)
    ax.text(0.75, 9.38, "EDGE CLIENT ON-DEVICE PIPELINE",
            fontsize=12.5, fontweight="bold", color="#0D47A1")
    ax.text(0.75, 9.08, "(Single Shared Backbone Envelope: 113.4 MB Peak VRAM)",
            fontsize=8.5, fontstyle="italic", color="#1565C0")

    # Communication Bus Container: x in [8.1, 11.7], y in [0.4, 9.7]
    bus_box = FancyBboxPatch((8.1, 0.4), 3.6, 9.3, boxstyle="round,pad=0.08,rounding_size=0.18",
                             facecolor=c_bg_bus, edgecolor=c_bus_border, linewidth=1.8, linestyle="--")
    ax.add_patch(bus_box)
    ax.text(9.9, 9.38, "COMMUNICATION BUS", fontsize=12.0, fontweight="bold", ha="center", color="#37474F")
    ax.text(9.9, 9.08, r"Synchronous Up/Download $\cdot$ Partial $C_p=0.20$", fontsize=8.2, ha="center", color="#546E7A")

    # Server Container: x in [11.9, 17.5], y in [0.4, 9.7]
    server_box = FancyBboxPatch((11.9, 0.4), 5.6, 9.3, boxstyle="round,pad=0.08,rounding_size=0.18",
                                facecolor=c_bg_server, edgecolor=c_server_border, linewidth=2.0)
    ax.add_patch(server_box)
    ax.text(12.15, 9.38, "FEDERATED SERVER AGGREGATION",
            fontsize=12.5, fontweight="bold", color="#B71C1C")
    ax.text(12.15, 9.08, "(Cluster Routing & Adaptive Aggregation Engine)",
            fontsize=8.5, fontstyle="italic", color="#C62828")

    # =========================================================================
    # 2. CLIENT INTERNALS (Left Column)
    # =========================================================================

    # --- A. Local Batch Input ---
    batch_box = FancyBboxPatch((0.75, 6.75), 1.9, 1.85, boxstyle="round,pad=0.06,rounding_size=0.12",
                               facecolor="#FFFFFF", edgecolor="#78909C", linewidth=1.5)
    ax.add_patch(batch_box)
    ax.text(1.7, 8.25, "Local Batch", fontsize=10.0, fontweight="bold", ha="center", color="#263238")
    ax.text(1.7, 7.80, r"$(X, Y) \sim \mathcal{D}_i$", fontsize=9.5, ha="center", color="#37474F")
    ax.text(1.7, 7.35, r"$\mathcal{Y}_i \subseteq \{1 \dots C\}$", fontsize=8.5, ha="center", color="#546E7A")
    ax.text(1.7, 6.95, "5 Local Epochs", fontsize=7.8, fontstyle="italic", ha="center", color="#78909C")

    # Arrow Batch -> Backbone
    ax.annotate("", xy=(3.05, 7.67), xytext=(2.65, 7.67),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#37474F"))

    # --- B. Shared Backbone ---
    bb_box = FancyBboxPatch((3.05, 6.4), 1.9, 2.55, boxstyle="round,pad=0.08,rounding_size=0.14",
                            facecolor="#E3F2FD", edgecolor=c_backbone, linewidth=2.0)
    ax.add_patch(bb_box)
    ax.text(4.0, 8.60, "Shared Backbone", fontsize=10.5, fontweight="bold", ha="center", color=c_backbone)
    ax.text(4.0, 8.25, r"$f_\theta(x)$ (ResNet-9)", fontsize=9.5, fontweight="bold", ha="center", color="#0D47A1")
    ax.text(4.0, 7.65, "Single Pass Forward\n" + r"$\nabla_\theta$ Backprop", fontsize=8.0, ha="center", color="#1565C0")
    ax.text(4.0, 6.75, r"Embedding $h \in \mathbb{R}^d$", fontsize=9.0, fontweight="bold", ha="center", color="#0D47A1")

    # Arrows from Backbone to 3 Heads
    ax.annotate("", xy=(5.35, 8.45), xytext=(4.95, 7.90), arrowprops=dict(arrowstyle="->", lw=1.8, color=c_root))
    ax.annotate("", xy=(5.35, 7.40), xytext=(4.95, 7.40), arrowprops=dict(arrowstyle="->", lw=1.8, color=c_parent))
    ax.annotate("", xy=(5.35, 6.35), xytext=(4.95, 6.90), arrowprops=dict(arrowstyle="->", lw=1.8, color=c_local))

    # --- C. 3-Tier Classification Heads ---
    # 1. Global Root Head
    head_r = FancyBboxPatch((5.35, 8.00), 2.35, 1.05, boxstyle="round,pad=0.06,rounding_size=0.1",
                            facecolor="#E8F5E9", edgecolor=c_root, linewidth=1.6)
    ax.add_patch(head_r)
    ax.text(5.50, 8.75, "1. Global Root Head", fontsize=9.0, fontweight="bold", color=c_root)
    ax.text(5.50, 8.45, r"$w_r \in \mathbb{R}^{d \times C} \to z_r \in \mathbb{R}^C$", fontsize=8.0, color="#1B5E20")
    ax.text(5.50, 8.15, "Unmasked global anchor", fontsize=7.2, color="#2E7D32")

    # 2. Cluster Parent Head
    head_p = FancyBboxPatch((5.35, 6.85), 2.35, 1.05, boxstyle="round,pad=0.06,rounding_size=0.1",
                            facecolor="#FFF3E0", edgecolor=c_parent, linewidth=1.6)
    ax.add_patch(head_p)
    ax.text(5.50, 7.60, "2. Cluster Parent Head", fontsize=9.0, fontweight="bold", color=c_parent)
    ax.text(5.50, 7.30, r"$w_{p(k)} \in \mathbb{R}^{d \times C} \to z_p \in \mathbb{R}^C$", fontsize=8.0, color="#BF360C")
    ax.text(5.50, 7.00, r"ACLM Masked ($c \notin \mathcal{Y}_i \to -\infty$)", fontsize=7.2, color="#E65100")

    # 3. Private Local Head
    head_l = FancyBboxPatch((5.35, 5.70), 2.35, 1.05, boxstyle="round,pad=0.06,rounding_size=0.1",
                            facecolor="#FCE4EC", edgecolor=c_local, linewidth=1.6)
    ax.add_patch(head_l)
    ax.text(5.50, 6.45, "3. Private Local Head", fontsize=9.0, fontweight="bold", color=c_local)
    ax.text(5.50, 6.15, r"$w_{l,i} \in \mathbb{R}^{d \times C} \to z_l \in \mathbb{R}^C$", fontsize=8.0, color="#880E4F")
    ax.text(5.50, 5.85, r"ACLM Masked $\cdot$ Strict Local", fontsize=7.2, color="#AD1457")
    
    # "PRIVATE" badge
    p_badge = FancyBboxPatch((7.00, 6.42), 0.62, 0.25, boxstyle="round,pad=0.02,rounding_size=0.04",
                             facecolor="#C2185B", edgecolor="none")
    ax.add_patch(p_badge)
    ax.text(7.31, 6.54, "PRIVATE", fontsize=5.8, fontweight="bold", color="#FFFFFF", ha="center", va="center")

    # --- D. Analytics & Loss Weighting Card (Bottom-Left) ---
    ana_box = FancyBboxPatch((0.75, 0.65), 3.9, 4.75, boxstyle="round,pad=0.08,rounding_size=0.14",
                             facecolor="#F3E5F5", edgecolor=c_analytics, linewidth=1.6)
    ax.add_patch(ana_box)
    ax.text(0.95, 5.08, "Dynamic Loss & Calibration", fontsize=10.0, fontweight="bold", color=c_analytics)

    ax.text(0.95, 4.68, r"$\bullet$ Local Shannon Label Skew ($R_{skew,i}$):", fontsize=8.5, fontweight="bold", color="#4A148C")
    ax.text(1.15, 4.35, r"$R_{skew,i} = \frac{\exp\left(-\sum_{c=1}^C p_c \ln p_c\right) - 1}{C - 1} \in [0, 1]$", fontsize=8.2, color="#4A148C")

    ax.text(0.95, 3.85, r"$\bullet$ Anchored Binomial Loss Weights ($\sum q = 1$):", fontsize=8.5, fontweight="bold", color="#4A148C")
    ax.text(1.15, 3.52, r"$q_{r,i} = a_i + (1-a_i) R_{skew,i}^2 \quad (\text{Root Anchor})$", fontsize=7.8, color="#311B92")
    ax.text(1.15, 3.20, r"$q_{p,i} = 2 R_{skew,i} (1 - R_{skew,i}) \quad (\text{Cluster Parent})$", fontsize=7.8, color="#311B92")
    ax.text(1.15, 2.88, r"$q_{l,i} = (1 - R_{skew,i})^2 \quad (\text{Private Local})$", fontsize=7.8, color="#311B92")

    ax.text(0.95, 2.38, r"$\bullet$ Composite On-Device Training Objective:", fontsize=8.5, fontweight="bold", color="#4A148C")
    ax.text(1.15, 2.05, r"$\mathcal{L}_{batch} = \lambda_r \mathcal{L}_{CE}(z_r, Y) + \lambda_p \mathcal{L}_{CE}^{mask}(z_p, Y) + \lambda_l \mathcal{L}_{CE}^{mask}(z_l, Y)$", fontsize=7.2, color="#1A237E", fontweight="bold")

    ax.text(0.95, 1.55, r"$\bullet$ Inference Blending with Active-Class Mask:", fontsize=8.5, fontweight="bold", color="#4A148C")
    ax.text(1.15, 1.22, r"$z_{pred} = \alpha_r z_r + \gamma_i (\alpha_p z_p + \alpha_l z_l) + \mathbf{m}_i$", fontsize=7.8, color="#1A237E")
    ax.text(1.15, 0.90, r"$\mathbf{m}_i[c] = -\infty \text{ if } c \notin \mathcal{Y}_i \text{ else } 0$", fontsize=7.5, color="#546E7A")

    # --- E. 256-Dim Sketch Compression Card (Bottom-Right of Client) ---
    sk_box = FancyBboxPatch((4.9, 0.65), 2.8, 4.75, boxstyle="round,pad=0.08,rounding_size=0.14",
                            facecolor="#E0F7FA", edgecolor=c_sketch, linewidth=1.6)
    ax.add_patch(sk_box)
    ax.text(5.05, 5.08, "256-Dim Sketching", fontsize=10.0, fontweight="bold", color=c_sketch)
    
    ax.text(5.05, 4.60, r"$\bullet$ Root Head Delta:", fontsize=8.2, fontweight="bold", color="#006064")
    ax.text(5.20, 4.25, r"$\Delta w_{r,i} = w_{r,i}^{(t+1)} - w_r^{(t)}$", fontsize=8.0, color="#006064")
    ax.text(5.20, 3.95, r"$\in \mathbb{R}^{d_{head}} \; (d_{head}=2570)$", fontsize=7.8, color="#00838F")

    ax.text(5.05, 3.45, r"$\bullet$ Random Projection:", fontsize=8.2, fontweight="bold", color="#006064")
    ax.text(5.20, 3.10, r"$s_i = P \cdot \Delta w_{r,i} \in \mathbb{R}^{256}$", fontsize=8.5, fontweight="bold", color="#004D40")
    ax.text(5.20, 2.75, r"$P \sim \mathcal{N}(0, 1/m)$ (Seed 42)", fontsize=7.8, color="#00838F")

    ax.text(5.05, 2.25, r"$\bullet$ Information Reduction:", fontsize=8.2, fontweight="bold", color="#006064")
    ax.text(5.20, 1.90, r"$10\times$ dimensional reduction", fontsize=7.8, color="#006064")
    ax.text(5.20, 1.60, r"Underdetermined inversion", fontsize=7.8, color="#006064")
    ax.text(5.20, 1.30, r"$m=256 \ll d_{head}=2570$", fontsize=7.8, color="#006064")
    ax.text(5.20, 0.90, r"Payload: $1.0\text{ KB}$ / round", fontsize=7.8, fontweight="bold", color="#004D40")

    # =========================================================================
    # 3. COMMUNICATION BUS (Middle Column: x in [8.1, 11.7])
    # =========================================================================
    # 5 Dedicated Non-Overlapping Communication Lanes + 1 Privacy Seal

    # Lane 1 (y=8.45): Download Global Model
    ax.annotate("", xy=(7.75, 8.45), xytext=(11.9, 8.45),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#1565C0", linestyle="-"))
    ax.text(9.9, 8.70, r"Download: Global $(\theta^{(t)}, w_r^{(t)})$", fontsize=8.5, fontweight="bold", color="#1565C0", ha="center")

    # Lane 2 (y=7.20): Upload Backbone & Root Updates
    ax.annotate("", xy=(11.9, 7.20), xytext=(7.75, 7.20),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#2E7D32", linestyle="-"))
    ax.text(9.9, 7.45, r"Upload: $(\Delta \theta_i, \Delta w_{r,i})$", fontsize=8.5, fontweight="bold", color="#2E7D32", ha="center")

    # Lane 3 (y=5.95): Download Cluster Parent Head
    ax.annotate("", xy=(7.75, 5.95), xytext=(11.9, 5.95),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#BF360C", linestyle="-"))
    ax.text(9.9, 6.20, r"Download: Cluster Head $w_{p,k}^{(t)}$", fontsize=8.5, fontweight="bold", color="#BF360C", ha="center")

    # Lane 4 (y=4.70): Upload Cluster Parent Update
    ax.annotate("", xy=(11.9, 4.70), xytext=(7.75, 4.70),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#E65100", linestyle="-"))
    ax.text(9.9, 4.95, r"Upload: Parent Update $\Delta w_{p,i}$", fontsize=8.5, fontweight="bold", color="#E65100", ha="center")

    # Lane 5 (y=3.45): Upload Sketch Vector
    ax.annotate("", xy=(11.9, 3.45), xytext=(7.75, 3.45),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#00838F", linestyle="-"))
    ax.text(9.9, 3.70, r"Upload: Sketch $s_i \in \mathbb{R}^{256}$ ($1.0\text{ KB}$)", fontsize=8.5, fontweight="bold", color="#00838F", ha="center")

    # Lane 6 (y in [0.75, 2.45]): STRICT PRIVACY BARRIER
    priv_box = FancyBboxPatch((8.35, 0.75), 3.1, 1.85, boxstyle="round,pad=0.06,rounding_size=0.1",
                              facecolor="#FFEBEE", edgecolor="#C2185B", linewidth=1.5, linestyle=":")
    ax.add_patch(priv_box)
    ax.text(9.9, 2.25, "ON-DEVICE PRIVACY BOUNDARY", fontsize=8.0, fontweight="bold", ha="center", color="#C2185B")
    ax.text(9.9, 1.85, r"Local Head $w_{l,i}$ and Updates $\Delta w_{l,i}$", fontsize=8.2, fontweight="bold", ha="center", color="#880E4F")
    ax.text(9.9, 1.45, r"$\times$ NEVER TRANSMITTED $\times$", fontsize=8.0, fontweight="bold", ha="center", color="#D32F2F")
    ax.text(9.9, 1.05, "Strict Zero-Exposure Local State", fontsize=7.2, fontstyle="italic", ha="center", color="#AD1457")

    # =========================================================================
    # 4. SERVER INTERNALS (Right Column: x in [11.9, 17.5])
    # =========================================================================

    # --- A. Global Consensus Engine (Top Right) ---
    srv_glob = FancyBboxPatch((12.15, 5.85), 5.1, 2.85, boxstyle="round,pad=0.08,rounding_size=0.14",
                              facecolor="#FFFFFF", edgecolor="#2E7D32", linewidth=1.8)
    ax.add_patch(srv_glob)
    ax.text(12.35, 8.35, "Global Aggregation Engine", fontsize=10.5, fontweight="bold", color="#1B5E20")

    ax.text(12.35, 7.95, r"$\bullet$ Backbone FedAvg Aggregation:", fontsize=8.5, fontweight="bold", color="#2E7D32")
    ax.text(12.55, 7.55, r"$\theta^{(t+1)} = \sum_{i=1}^N \frac{|\mathcal{D}_i|}{|\mathcal{D}|} \left( \theta^{(t)} + \Delta \theta_i \right)$", fontsize=8.2, color="#1B5E20")

    ax.text(12.35, 6.95, r"$\bullet$ Global Root Head Consensus:", fontsize=8.5, fontweight="bold", color="#2E7D32")
    ax.text(12.55, 6.55, r"$w_r^{(t+1)} = \sum_{i=1}^N \frac{|\mathcal{D}_i|}{|\mathcal{D}|} \left( w_r^{(t)} + \Delta w_{r,i} \right)$", fontsize=8.2, color="#1B5E20")

    ax.text(12.35, 6.10, r"Preserves full global IID capability ($72.73\%$ consensus)", fontsize=7.5, fontstyle="italic", color="#388E3C")

    # --- B. Cluster Routing & Momentum Updates (Bottom Right) ---
    srv_clust = FancyBboxPatch((12.15, 0.65), 5.1, 4.95, boxstyle="round,pad=0.08,rounding_size=0.14",
                               facecolor="#FFFFFF", edgecolor="#E65100", linewidth=1.8)
    ax.add_patch(srv_clust)
    ax.text(12.35, 5.25, "Cluster Routing & Adaptive Aggregation", fontsize=10.5, fontweight="bold", color="#BF360C")

    ax.text(12.35, 4.80, r"1. Sketch Cosine Similarity Routing:", fontsize=8.8, fontweight="bold", color="#D84315")
    ax.text(12.55, 4.45, r"$\mathcal{C}_k = \{ i : \arg\max_j \cos(s_i, \mu_j) = k \}$", fontsize=8.5, color="#BF360C")
    ax.text(12.55, 4.12, r"Directional stability trigger: $\bar{S}_{thresh} = 0.30$", fontsize=7.8, color="#546E7A")

    ax.text(12.35, 3.62, r"2. Momentum Cluster Head Update:", fontsize=8.8, fontweight="bold", color="#D84315")
    ax.text(12.55, 3.25, r"$w_{p,k}^{(t+1)} = \beta_{c,k} w_{p,k}^{(t)} + (1 - \beta_{c,k}) \sum_{i \in \mathcal{C}_k} \frac{|\mathcal{D}_i|}{|\mathcal{D}_{\mathcal{C}_k}|} (w_{p,k}^{(t)} + \Delta w_{p,i})$", fontsize=7.3, color="#BF360C")
    ax.text(12.55, 2.85, r"Adaptive momentum: $\beta_{c,k} = V_k / (V_k + \|\Delta_k\|^2)$", fontsize=7.8, color="#E65100")

    ax.text(12.35, 2.38, r"3. Collaborative Cluster Head Bank ($K=3$):", fontsize=8.8, fontweight="bold", color="#D84315")

    # Visual Cluster Cards (K=3)
    c_pal = [("#FFF8E1", "#FFA000"), ("#FFF3E0", "#FB8C00"), ("#FBE9E7", "#F4511E")]
    for idx, ((bg, fg), c_name) in enumerate(zip(c_pal, ["Cluster 1", "Cluster 2", "Cluster 3"])):
        c_x = 12.35 + idx * 1.58
        c_box = FancyBboxPatch((c_x, 1.0), 1.45, 1.05, boxstyle="round,pad=0.04,rounding_size=0.08",
                               facecolor=bg, edgecolor=fg, linewidth=1.4)
        ax.add_patch(c_box)
        ax.text(c_x + 0.725, 1.75, c_name, fontsize=8.2, fontweight="bold", ha="center", color=fg)
        ax.text(c_x + 0.725, 1.42, f"Head $w_{{p,{idx+1}}}$", fontsize=8.0, fontweight="bold", ha="center", color="#BF360C")
        ax.text(c_x + 0.725, 1.15, "ACLM Head", fontsize=7.0, ha="center", color="#78909C")

    # =========================================================================
    # 5. SAVE HIGH-RES OUTPUT
    # =========================================================================
    plt.tight_layout()
    out_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report", "figures", "hep_architecture.png"),
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"[ok] Generated zero-overlap architecture diagram at: {p}")
    plt.close(fig)

if __name__ == "__main__":
    create_architecture_diagram()
