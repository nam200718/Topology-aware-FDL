"""
Generate clean, simplified, publication-quality architecture diagram for HEP.
Strictly adheres to zero overlap with uncluttered cards and high readability.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16.5, 9.2), dpi=300)
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    # Colors
    c_bg_client = "#F4F8FC"
    c_bg_server = "#FAF5F5"
    c_client_border = "#90CAF9"
    c_server_border = "#EF9A9A"
    c_bus_border = "#B0BEC5"
    c_bg_bus = "#F8FAFC"

    c_backbone = "#0D47A1"
    c_root = "#1B5E20"       # Green
    c_parent = "#E65100"     # Orange
    c_local = "#880E4F"      # Maroon
    c_analytics = "#4A148C"  # Purple
    c_sketch = "#006064"     # Teal

    # Container 1: Edge Client Pipeline
    client_box = FancyBboxPatch((0.4, 0.4), 6.9, 8.4, boxstyle="round,pad=0.08,rounding_size=0.18",
                                facecolor=c_bg_client, edgecolor=c_client_border, linewidth=2.0)
    ax.add_patch(client_box)
    ax.text(0.65, 8.45, "EDGE CLIENT ON DEVICE PIPELINE", fontsize=12.0, fontweight="bold", color="#0D47A1")
    ax.text(0.65, 8.18, "(Single Shared Backbone: 113.4 MB Peak VRAM)", fontsize=8.5, fontstyle="italic", color="#1565C0")

    # Container 2: Communication Bus
    bus_box = FancyBboxPatch((7.6, 0.4), 3.2, 8.4, boxstyle="round,pad=0.08,rounding_size=0.18",
                             facecolor=c_bg_bus, edgecolor=c_bus_border, linewidth=1.8, linestyle="--")
    ax.add_patch(bus_box)
    ax.text(9.2, 8.45, "COMMUNICATION BUS", fontsize=11.5, fontweight="bold", ha="center", color="#37474F")
    ax.text(9.2, 8.18, "Synchronous Upload and Download", fontsize=8.0, ha="center", color="#546E7A")

    # Container 3: Federated Server
    server_box = FancyBboxPatch((11.1, 0.4), 5.0, 8.4, boxstyle="round,pad=0.08,rounding_size=0.18",
                                facecolor=c_bg_server, edgecolor=c_server_border, linewidth=2.0)
    ax.add_patch(server_box)
    ax.text(11.35, 8.45, "FEDERATED SERVER", fontsize=12.0, fontweight="bold", color="#B71C1C")
    ax.text(11.35, 8.18, "(Consensus Aggregation and Cluster Routing)", fontsize=8.5, fontstyle="italic", color="#C62828")

    # Client Section: Input Batch
    batch_box = FancyBboxPatch((0.65, 5.8), 1.6, 1.9, boxstyle="round,pad=0.06,rounding_size=0.1",
                               facecolor="#FFFFFF", edgecolor="#78909C", linewidth=1.4)
    ax.add_patch(batch_box)
    ax.text(1.45, 7.25, "Local Batch", fontsize=9.5, fontweight="bold", ha="center", color="#263238")
    ax.text(1.45, 6.75, r"$(X, Y) \sim \mathcal{D}_i$", fontsize=9.0, ha="center", color="#37474F")
    ax.text(1.45, 6.25, r"$\mathcal{Y}_i \subseteq \{1 \dots C\}$", fontsize=8.0, ha="center", color="#546E7A")

    # Arrow Input to Backbone
    ax.annotate("", xy=(2.6, 6.75), xytext=(2.25, 6.75),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#37474F"))

    # Client Section: Shared Backbone
    bb_box = FancyBboxPatch((2.6, 5.5), 1.8, 2.5, boxstyle="round,pad=0.06,rounding_size=0.12",
                            facecolor="#E3F2FD", edgecolor=c_backbone, linewidth=1.8)
    ax.add_patch(bb_box)
    ax.text(3.5, 7.60, "Shared Backbone", fontsize=9.5, fontweight="bold", ha="center", color=c_backbone)
    ax.text(3.5, 7.22, r"$f_\theta(x)$ (ResNet-9)", fontsize=8.8, fontweight="bold", ha="center", color="#0D47A1")
    ax.text(3.5, 6.55, "Single forward pass\ncomputed once", fontsize=7.8, ha="center", color="#1565C0")
    ax.text(3.5, 5.85, r"Embedding $h \in \mathbb{R}^d$", fontsize=8.2, fontweight="bold", ha="center", color="#0D47A1")

    # Arrows from Backbone to 3 Heads
    ax.annotate("", xy=(4.75, 7.45), xytext=(4.4, 7.0), arrowprops=dict(arrowstyle="->", lw=1.6, color=c_root))
    ax.annotate("", xy=(4.75, 6.45), xytext=(4.4, 6.45), arrowprops=dict(arrowstyle="->", lw=1.6, color=c_parent))
    ax.annotate("", xy=(4.75, 5.45), xytext=(4.4, 5.9), arrowprops=dict(arrowstyle="->", lw=1.6, color=c_local))

    # Client Section: 3 Linear Classification Heads
    # 1. Root Head
    head_r = FancyBboxPatch((4.75, 7.0), 2.3, 0.9, boxstyle="round,pad=0.05,rounding_size=0.08",
                            facecolor="#E8F5E9", edgecolor=c_root, linewidth=1.5)
    ax.add_patch(head_r)
    ax.text(4.9, 7.60, "1. Global Root Head", fontsize=8.5, fontweight="bold", color=c_root)
    ax.text(4.9, 7.25, r"$w_r \to z_r$ (Unmasked Anchor)", fontsize=7.8, color="#1B5E20")

    # 2. Parent Head
    head_p = FancyBboxPatch((4.75, 6.0), 2.3, 0.9, boxstyle="round,pad=0.05,rounding_size=0.08",
                            facecolor="#FFF3E0", edgecolor=c_parent, linewidth=1.5)
    ax.add_patch(head_p)
    ax.text(4.9, 6.60, "2. Cluster Parent Head", fontsize=8.5, fontweight="bold", color=c_parent)
    ax.text(4.9, 6.25, r"$w_{p(k)} \to z_p$ (ACLM Masked)", fontsize=7.8, color="#BF360C")

    # 3. Local Head
    head_l = FancyBboxPatch((4.75, 5.0), 2.3, 0.9, boxstyle="round,pad=0.05,rounding_size=0.08",
                            facecolor="#FCE4EC", edgecolor=c_local, linewidth=1.5)
    ax.add_patch(head_l)
    ax.text(4.9, 5.60, "3. Private Local Head", fontsize=8.5, fontweight="bold", color=c_local)
    ax.text(4.9, 5.25, r"$w_{l,i} \to z_l$ (On Device Only)", fontsize=7.8, color="#880E4F")

    # Client Section: Dynamic Loss Weighting Card (Bottom Left)
    loss_box = FancyBboxPatch((0.65, 0.65), 3.5, 4.0, boxstyle="round,pad=0.08,rounding_size=0.12",
                              facecolor="#F3E5F5", edgecolor=c_analytics, linewidth=1.5)
    ax.add_patch(loss_box)
    ax.text(0.85, 4.30, "Dynamic Loss & Weighting", fontsize=9.5, fontweight="bold", color=c_analytics)
    ax.text(0.85, 3.85, r"• Label Skew: $R_{skew,i} \in [0, 1]$", fontsize=8.2, fontweight="bold", color="#4A148C")
    ax.text(1.05, 3.45, "Computed from Shannon entropy", fontsize=7.5, color="#6A1B9A")
    ax.text(0.85, 2.95, r"• Binomial Weights: $\sum \lambda = 1$", fontsize=8.2, fontweight="bold", color="#4A148C")
    ax.text(1.05, 2.55, r"$\lambda_r$ (Root), $\lambda_p$ (Parent), $\lambda_l$ (Local)", fontsize=7.5, color="#6A1B9A")
    ax.text(0.85, 2.05, r"• Training Objective:", fontsize=8.2, fontweight="bold", color="#4A148C")
    ax.text(1.05, 1.65, r"$\mathcal{L} = \lambda_r \mathcal{L}_{CE} + \lambda_p \mathcal{L}_{mask} + \lambda_l \mathcal{L}_{mask}$", fontsize=7.5, fontweight="bold", color="#311B92")
    ax.text(0.85, 1.15, r"• Blended Inference: $z_{pred} + \mathbf{m}_i$", fontsize=8.0, color="#4A148C")

    # Client Section: 256-Dim Sketching Card (Bottom Right)
    sk_box = FancyBboxPatch((4.45, 0.65), 2.6, 4.0, boxstyle="round,pad=0.08,rounding_size=0.12",
                            facecolor="#E0F7FA", edgecolor=c_sketch, linewidth=1.5)
    ax.add_patch(sk_box)
    ax.text(4.65, 4.30, "256-Dim Sketching", fontsize=9.5, fontweight="bold", color=c_sketch)
    ax.text(4.65, 3.75, r"• Delta: $\Delta w_{r,i} \in \mathbb{R}^{2570}$", fontsize=8.0, color="#006064")
    ax.text(4.65, 3.15, r"• Projection: $s_i = P \Delta w_{r,i}$", fontsize=8.2, fontweight="bold", color="#004D40")
    ax.text(4.65, 2.75, r"  $s_i \in \mathbb{R}^{256}$ ($10\times$ compression)", fontsize=7.5, color="#006064")
    ax.text(4.65, 2.15, "• Privacy Aware Routing", fontsize=8.0, color="#006064")
    ax.text(4.65, 1.75, "  Underdetermined inversion", fontsize=7.5, color="#00838F")
    ax.text(4.65, 1.15, r"• Payload: $1.0\text{ KB}$ / round", fontsize=8.0, fontweight="bold", color="#004D40")

    # Communication Bus Transactions
    # 1. Download
    down_box = FancyBboxPatch((7.8, 6.2), 2.8, 1.4, boxstyle="round,pad=0.05,rounding_size=0.08",
                              facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.3)
    ax.add_patch(down_box)
    ax.text(9.2, 7.20, "Download Parameters", fontsize=8.2, fontweight="bold", ha="center", color="#0D47A1")
    ax.text(9.2, 6.80, r"Global $\theta^{(t)}, w_r^{(t)}$", fontsize=7.8, ha="center", color="#1565C0")
    ax.text(9.2, 6.45, r"Cluster Head $w_{p,k}^{(t)}$", fontsize=7.8, ha="center", color="#1565C0")
    ax.annotate("", xy=(7.3, 6.9), xytext=(7.8, 6.9), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1565C0"))
    ax.annotate("", xy=(10.6, 6.9), xytext=(11.1, 6.9), arrowprops=dict(arrowstyle="<-", lw=1.8, color="#1565C0"))

    # 2. Upload
    up_box = FancyBboxPatch((7.8, 3.8), 2.8, 1.9, boxstyle="round,pad=0.05,rounding_size=0.08",
                            facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.3)
    ax.add_patch(up_box)
    ax.text(9.2, 5.30, "Upload Updates", fontsize=8.2, fontweight="bold", ha="center", color="#1B5E20")
    ax.text(9.2, 4.85, r"Backbone $\Delta \theta_i$", fontsize=7.8, ha="center", color="#2E7D32")
    ax.text(9.2, 4.45, r"Root $\Delta w_{r,i}$ & Parent $\Delta w_{p,i}$", fontsize=7.8, ha="center", color="#2E7D32")
    ax.text(9.2, 4.05, r"Sketch $s_i \in \mathbb{R}^{256}$", fontsize=7.8, fontweight="bold", ha="center", color="#00695C")
    ax.annotate("", xy=(7.8, 4.75), xytext=(7.3, 4.75), arrowprops=dict(arrowstyle="<-", lw=1.8, color="#2E7D32"))
    ax.annotate("", xy=(11.1, 4.75), xytext=(10.6, 4.75), arrowprops=dict(arrowstyle="->", lw=1.8, color="#2E7D32"))

    # 3. Privacy Seal
    priv_box = FancyBboxPatch((7.8, 1.0), 2.8, 2.2, boxstyle="round,pad=0.05,rounding_size=0.08",
                              facecolor="#FFEBEE", edgecolor="#C2185B", linewidth=1.4, linestyle=":")
    ax.add_patch(priv_box)
    ax.text(9.2, 2.80, "PRIVACY BOUNDARY", fontsize=8.0, fontweight="bold", ha="center", color="#C2185B")
    ax.text(9.2, 2.30, r"Local Head $w_{l,i}$", fontsize=8.2, fontweight="bold", ha="center", color="#880E4F")
    ax.text(9.2, 1.75, "NEVER UPLOADED", fontsize=8.2, fontweight="bold", ha="center", color="#D32F2F")
    ax.text(9.2, 1.25, "100% On-Device Isolation", fontsize=7.2, fontstyle="italic", ha="center", color="#AD1457")

    # Server Section: Global Aggregation Box
    srv_glob = FancyBboxPatch((11.35, 4.9), 4.5, 2.9, boxstyle="round,pad=0.08,rounding_size=0.12",
                              facecolor="#FFFFFF", edgecolor="#2E7D32", linewidth=1.6)
    ax.add_patch(srv_glob)
    ax.text(11.55, 7.45, "Global Aggregation Engine", fontsize=9.5, fontweight="bold", color="#1B5E20")
    ax.text(11.55, 6.95, r"• Backbone FedAvg Aggregation:", fontsize=8.0, fontweight="bold", color="#2E7D32")
    ax.text(11.75, 6.55, r"$\theta^{(t+1)} \leftarrow \sum_{i=1}^N \frac{|\mathcal{D}_i|}{|\mathcal{D}|} (\theta^{(t)} + \Delta \theta_i)$", fontsize=8.0, color="#1B5E20")
    ax.text(11.55, 5.95, r"• Global Root Head Consensus:", fontsize=8.0, fontweight="bold", color="#2E7D32")
    ax.text(11.75, 5.55, r"$w_r^{(t+1)} \leftarrow \sum_{i=1}^N \frac{|\mathcal{D}_i|}{|\mathcal{D}|} (w_r^{(t)} + \Delta w_{r,i})$", fontsize=8.0, color="#1B5E20")
    ax.text(11.55, 5.15, "Full global consensus preserved (72.73% IID)", fontsize=7.2, fontstyle="italic", color="#388E3C")

    # Server Section: Cluster Routing and Momentum Updates
    srv_clust = FancyBboxPatch((11.35, 0.65), 4.5, 3.9, boxstyle="round,pad=0.08,rounding_size=0.12",
                               facecolor="#FFFFFF", edgecolor="#E65100", linewidth=1.6)
    ax.add_patch(srv_clust)
    ax.text(11.55, 4.20, "Cluster Routing & Aggregation", fontsize=9.5, fontweight="bold", color="#BF360C")
    ax.text(11.55, 3.70, "1. Sketch Cosine Routing:", fontsize=8.0, fontweight="bold", color="#D84315")
    ax.text(11.75, 3.35, r"Assign client $i \to \mathcal{C}_k$ via $\cos(s_i, \mu_k)$", fontsize=7.8, color="#BF360C")
    ax.text(11.55, 2.75, "2. Momentum Cluster Head Updates:", fontsize=8.0, fontweight="bold", color="#D84315")
    ax.text(11.75, 2.35, r"$w_{p,k}^{(t+1)} \leftarrow \beta_{c,k} w_{p,k}^{(t)} + (1-\beta_{c,k}) \bar{w}_{p,k}$", fontsize=7.8, color="#BF360C")

    # Cluster Cards (K=3)
    c_pal = [("#FFF8E1", "#FFA000"), ("#FFF3E0", "#FB8C00"), ("#FBE9E7", "#F4511E")]
    for idx, ((bg, fg), c_name) in enumerate(zip(c_pal, ["Cluster 1", "Cluster 2", "Cluster 3"])):
        c_x = 11.55 + idx * 1.38
        c_box = FancyBboxPatch((c_x, 0.95), 1.25, 1.0, boxstyle="round,pad=0.03,rounding_size=0.06",
                               facecolor=bg, edgecolor=fg, linewidth=1.2)
        ax.add_patch(c_box)
        ax.text(c_x + 0.625, 1.62, c_name, fontsize=7.8, fontweight="bold", ha="center", color=fg)
        ax.text(c_x + 0.625, 1.30, f"Head $w_{{p,{idx+1}}}$", fontsize=7.5, fontweight="bold", ha="center", color="#BF360C")
        ax.text(c_x + 0.625, 1.08, "ACLM Masked", fontsize=6.5, ha="center", color="#78909C")

    plt.tight_layout()
    out_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report", "figures", "hep_architecture.png"),
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"[ok] Saved simplified architecture figure to {p}")
    plt.close(fig)

if __name__ == "__main__":
    create_architecture_diagram()
