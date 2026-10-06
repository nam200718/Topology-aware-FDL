"""
Generate clean, simplistic, publication-quality system architecture diagram for FedHEP.
Adheres strictly to the user's preference:
- Clean, iconic, high-level hierarchical topology (Server -> Peer Clusters -> Edge Clients)
- No mathematical equations (zero LaTeX formulas)
- Distinct non-plagiarized design specifically tailored to FedHEP's 3-tier architecture:
  1. Top: Federated Server (Global Consensus Aggregator) + Byzantine Defense (Subspace SCCF & Trust Tracking)
  2. Middle: Collaborative Peer Clusters (Parent Heads via RP Sketches)
  3. Bottom: Heterogeneous Edge Clients (Single Shared Backbone, ACLM, Private Local Head)
- Curved dual-directional communication arrows (Download dashed, Upload solid)
- Clear legend and pristine white background
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    FancyBboxPatch, Ellipse, Rectangle, Polygon, FancyArrowPatch, Circle, PathPatch
)
from matplotlib.path import Path

def create_cloud_patch(cx, cy, w, h, edge_color="#1E293B", face_color="#FFFFFF", lw=2.2, zorder=2):
    """Generate a clean, single-contour iconic cloud outline with zero interior lines."""
    N = 400
    x = np.linspace(-1, 1, N)
    y = np.linspace(-0.8, 0.8, N)
    X, Y = np.meshgrid(x, y)

    # Base rectangle and 5 perimeter puffs
    mask = (np.abs(X) <= 0.38) & (Y >= -0.35) & (Y <= 0.05)
    puffs = [
        (-0.35, -0.10, 0.26, 0.26),  # left-low
        (-0.25, 0.18, 0.28, 0.28),   # left-top
        (0.04, 0.28, 0.36, 0.36),    # apex
        (0.32, 0.12, 0.26, 0.26),    # right-top
        (0.40, -0.12, 0.24, 0.24),   # right-low
    ]
    for px, py, rx, ry in puffs:
        mask |= ((X - px)**2 / rx**2 + (Y - py)**2 / ry**2 <= 1.0)

    fig_d, ax_d = plt.subplots()
    cs = ax_d.contour(X, Y, mask.astype(float), levels=[0.5])
    raw_path = cs.get_paths()[0]
    plt.close(fig_d)

    # Normalize bounds and transform to target (cx, cy, w, h)
    vx = raw_path.vertices[:, 0]
    vy = raw_path.vertices[:, 1]
    norm_x = (vx - 0.0) / (0.65 - (-0.62))
    norm_y = (vy - 0.14) / (0.64 - (-0.35))
    
    scaled_verts = np.column_stack([cx + norm_x * w, cy + norm_y * h])
    new_path = Path(scaled_verts, raw_path.codes)
    return PathPatch(new_path, facecolor=face_color, edgecolor=edge_color, lw=lw, zorder=zorder)

def draw_server_rack(ax, cx, cy, w, h, zorder=5):
    """Draw a modern, crisp server rack icon."""
    # Outer rack chassis
    chassis = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             facecolor="#1E293B", edgecolor="#0F172A", linewidth=1.8, zorder=zorder)
    ax.add_patch(chassis)
    
    # 3 server bays
    bay_h = h * 0.22
    bay_w = w * 0.84
    gap = h * 0.08
    start_y = cy + h/2 - bay_h - gap*0.8
    for i in range(3):
        by = start_y - i * (bay_h + gap)
        bay = FancyBboxPatch((cx - bay_w/2, by), bay_w, bay_h,
                             boxstyle="round,pad=0.01,rounding_size=0.02",
                             facecolor="#334155", edgecolor="#475569", linewidth=1.0, zorder=zorder+1)
        ax.add_patch(bay)
        # LED indicators
        led1 = Circle((cx - bay_w/2 + 0.15*w, by + bay_h/2), bay_h*0.22, facecolor="#10B981", edgecolor="none", zorder=zorder+2)
        led2 = Circle((cx - bay_w/2 + 0.28*w, by + bay_h/2), bay_h*0.22, facecolor="#38BDF8", edgecolor="none", zorder=zorder+2)
        ax.add_patch(led1)
        ax.add_patch(led2)
        # Drive vent slots
        for vx in [0.48*w, 0.62*w, 0.76*w]:
            vent = Rectangle((cx - bay_w/2 + vx, by + bay_h*0.35), bay_w*0.12, bay_h*0.3,
                             facecolor="#1E293B", edgecolor="none", zorder=zorder+2)
            ax.add_patch(vent)

def draw_shield(ax, cx, cy, w, h, zorder=5):
    """Draw a clean defense shield icon."""
    pts = [
        [cx - w/2, cy + h/2],
        [cx + w/2, cy + h/2],
        [cx + w/2, cy],
        [cx, cy - h/2],
        [cx - w/2, cy],
    ]
    shield = Polygon(pts, closed=True, facecolor="#F8FAFC", edgecolor="#1E293B", linewidth=2.0, zorder=zorder)
    ax.add_patch(shield)
    
    # Inner red cross accent
    cross_w = w * 0.18
    cross_h = h * 0.45
    vbar = Rectangle((cx - cross_w/2, cy - cross_h/2 + 0.05*h), cross_w, cross_h,
                     facecolor="#DC2626", edgecolor="none", zorder=zorder+1)
    hbar = Rectangle((cx - cross_h/2 + 0.05*h, cy - cross_w/2 + 0.05*h), cross_h, cross_w,
                     facecolor="#DC2626", edgecolor="none", zorder=zorder+1)
    ax.add_patch(vbar)
    ax.add_patch(hbar)

def draw_neural_net(ax, cx, cy, w, h, color="#059669"):
    """Draw a minimalist neural network diagram representing a cluster parent model."""
    layer_xs = [cx - w*0.38, cx, cx + w*0.38]
    in_ys = [cy - h*0.32, cy, cy + h*0.32]
    mid_ys = [cy - h*0.22, cy + h*0.22]
    out_ys = [cy - h*0.32, cy, cy + h*0.32]
    
    # Synapse connections
    for iy in in_ys:
        for my in mid_ys:
            ax.plot([layer_xs[0], layer_xs[1]], [iy, my], color=color, alpha=0.35, lw=1.2, zorder=3)
    for my in mid_ys:
        for oy in out_ys:
            ax.plot([layer_xs[1], layer_xs[2]], [my, oy], color=color, alpha=0.35, lw=1.2, zorder=3)
            
    # Input nodes
    for y in in_ys:
        c = Circle((layer_xs[0], y), h*0.13, facecolor="#FFFFFF", edgecolor=color, lw=2.0, zorder=4)
        ax.add_patch(c)
    # Subtle vertical dots between input nodes
    ax.text(layer_xs[0], (in_ys[0] + in_ys[1])/2, ":", fontsize=7.0, ha="center", va="center", color=color, zorder=4)
    ax.text(layer_xs[0], (in_ys[1] + in_ys[2])/2, ":", fontsize=7.0, ha="center", va="center", color=color, zorder=4)
        
    # Hidden blocks (feature representation)
    block_w = w * 0.22
    block_h = h * 0.26
    for y in mid_ys:
        b = FancyBboxPatch((layer_xs[1] - block_w/2, y - block_h/2), block_w, block_h,
                           boxstyle="round,pad=0.01,rounding_size=0.03",
                           facecolor="#FFFFFF", edgecolor=color, lw=2.0, zorder=4)
        ax.add_patch(b)
    ax.text(layer_xs[1], cy, ":", fontsize=7.0, ha="center", va="center", color=color, zorder=4)
        
    # Output nodes
    for y in out_ys:
        c = Circle((layer_xs[2], y), h*0.13, facecolor="#FFFFFF", edgecolor=color, lw=2.0, zorder=4)
        ax.add_patch(c)
    ax.text(layer_xs[2], (out_ys[0] + out_ys[1])/2, ":", fontsize=7.0, ha="center", va="center", color=color, zorder=4)
    ax.text(layer_xs[2], (out_ys[1] + out_ys[2])/2, ":", fontsize=7.0, ha="center", va="center", color=color, zorder=4)

def draw_database(ax, cx, cy, w, h, color="#059669"):
    """Draw a 3-layer cylinder representing an on-device private dataset."""
    disk_h = h * 0.26
    spacing = h * 0.30
    for i in range(3):
        dy = cy - h/2 + i * spacing
        body = Rectangle((cx - w/2, dy), w, disk_h, facecolor=color, edgecolor=color, lw=1.0, zorder=3)
        ax.add_patch(body)
        top = Ellipse((cx, dy + disk_h), w, disk_h * 0.75, facecolor=color, edgecolor="#FFFFFF", lw=1.2, zorder=4)
        ax.add_patch(top)
        if i == 0:
            bot = Ellipse((cx, dy), w, disk_h * 0.75, facecolor=color, edgecolor="none", lw=1.0, zorder=2)
            ax.add_patch(bot)

def draw_curved_arrow(ax, p_start, p_end, is_dashed=False, color="#2563EB", rad=0.15, lw=1.6):
    """Draw clean curved arrows between tiers."""
    ls = "--" if is_dashed else "-"
    arrow = FancyArrowPatch(p_start, p_end,
                            connectionstyle=f"arc3,rad={rad}",
                            arrowstyle="->",
                            mutation_scale=13,
                            linestyle=ls,
                            color=color,
                            linewidth=lw,
                            zorder=6)
    ax.add_patch(arrow)

def build_simplistic_fedhep_diagram():
    # Typography: clean serif matching academic publications
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times", "serif"]
    plt.rcParams["pdf.fonttype"] = 42

    # Canvas dimensions: 15.5 x 8.8 inches
    fig, ax = plt.subplots(figsize=(15.5, 8.8), dpi=300)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    # Clean publication white background
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # =========================================================================
    # 1. TOP TIER: SERVER & DEFENSE & LEGEND
    # =========================================================================
    # 1.1 Byzantine Defense Module (Top Left)
    draw_server_rack(ax, 2.2, 7.30, 0.75, 0.85, zorder=5)
    draw_shield(ax, 3.1, 7.30, 0.70, 0.85, zorder=6)
    ax.text(2.65, 8.28, "Byzantine Defense", fontsize=11.5, fontweight="bold", ha="center", color="#0F172A")
    ax.text(2.65, 8.00, "(Subspace SCCF + Trust Tracking)", fontsize=8.8, ha="center", color="#475569")

    # 1.2 Federated Server (Top Center)
    cloud_patch = create_cloud_patch(7.75, 7.20, 3.4, 1.70, edge_color="#1E293B", face_color="#FFFFFF", lw=2.2, zorder=3)
    ax.add_patch(cloud_patch)
    draw_server_rack(ax, 7.75, 7.10, 1.15, 0.80, zorder=5)
    ax.text(7.75, 8.48, "Federated Server", fontsize=13.5, fontweight="bold", ha="center", color="#0F172A")
    ax.text(7.75, 8.22, "(Global Consensus Aggregator)", fontsize=9.5, fontstyle="italic", ha="center", color="#475569")

    # 1.3 Communication Legend (Top Right)
    legend_box = FancyBboxPatch((11.9, 7.05), 3.2, 1.25,
                                boxstyle="round,pad=0.04,rounding_size=0.06",
                                facecolor="#FFFFFF", edgecolor="#334155", linewidth=1.4, zorder=3)
    ax.add_patch(legend_box)
    
    # Legend items
    ax.annotate("", xy=(12.85, 7.85), xytext=(12.15, 7.85),
                arrowprops=dict(arrowstyle="<-", linestyle="--", lw=1.8, color="#2563EB"))
    ax.text(13.05, 7.85, "Download", fontsize=9.8, fontweight="bold", va="center", color="#1E293B")
    
    ax.annotate("", xy=(12.85, 7.40), xytext=(12.15, 7.40),
                arrowprops=dict(arrowstyle="->", linestyle="-", lw=1.8, color="#2563EB"))
    ax.text(13.05, 7.40, "Upload", fontsize=9.8, fontweight="bold", va="center", color="#1E293B")

    # =========================================================================
    # 2. MIDDLE TIER: COLLABORATIVE PEER CLUSTERS (PARENT HEADS)
    # =========================================================================
    cluster_centers = [3.2, 7.75, 12.3]
    cluster_colors = ["#059669", "#7C3AED", "#0284C7"]  # Green, Purple, Cyan/Blue
    cluster_labels = ["Peer Cluster 1", "Peer Cluster k", "Peer Cluster K"]
    cluster_sublabels = ["(Parent Head Model)", "(Parent Head Model)", "(Parent Head Model)"]

    for cx, col, clbl, csub in zip(cluster_centers, cluster_colors, cluster_labels, cluster_sublabels):
        # Dashed cluster container box with ample height
        c_box = FancyBboxPatch((cx - 1.5, 3.40), 3.0, 1.90,
                               boxstyle="round,pad=0.06,rounding_size=0.12",
                               facecolor="#FFFFFF", edgecolor=col, linewidth=1.8, linestyle="--", zorder=2)
        ax.add_patch(c_box)
        # Neural network icon
        draw_neural_net(ax, cx, 4.55, 2.2, 1.15, color=col)
        # Clean labels positioned comfortably inside box
        ax.text(cx, 3.82, clbl, fontsize=10.0, fontweight="bold", ha="center", color="#0F172A")
        ax.text(cx, 3.58, csub, fontsize=8.2, fontstyle="italic", ha="center", color="#475569")

    # Ellipsis dots between middle clusters
    ax.text(5.47, 4.40, "•  •  •", fontsize=15.0, ha="center", va="center", color="#64748B")
    ax.text(10.02, 4.40, "•  •  •", fontsize=15.0, ha="center", va="center", color="#64748B")

    # =========================================================================
    # 3. BOTTOM TIER: HETEROGENEOUS EDGE CLIENTS
    # =========================================================================
    client_groups = [
        # Cluster 1 clients
        (3.2, [2.15, 3.20, 4.25], cluster_colors[0], ["Client", "Client", "Client"]),
        # Cluster k clients
        (7.75, [6.70, 7.75, 8.80], cluster_colors[1], ["Client", "Client", "Client"]),
        # Cluster K clients
        (12.3, [11.25, 12.30, 13.35], cluster_colors[2], ["Client", "Client", "Client"])
    ]

    for group_cx, client_xs, col, tags in client_groups:
        # Client container dashed box
        grp_w = 2.95
        grp_box = FancyBboxPatch((group_cx - grp_w/2, 1.05), grp_w, 1.50,
                                 boxstyle="round,pad=0.04,rounding_size=0.1",
                                 facecolor="#FFFFFF", edgecolor="#94A3B8", linewidth=1.4, linestyle="--", zorder=2)
        ax.add_patch(grp_box)

        for cx, tag in zip(client_xs, tags):
            # Client sub-box
            c_box = FancyBboxPatch((cx - 0.42, 1.16), 0.84, 1.28,
                                   boxstyle="round,pad=0.02,rounding_size=0.06",
                                   facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0, zorder=3)
            ax.add_patch(c_box)
            # Database cylinder
            draw_database(ax, cx, 1.90, 0.48, 0.48, color=col)
            # Label
            ax.text(cx, 1.33, tag, fontsize=8.2, fontweight="bold", ha="center", color="#1E293B")

        # Ellipsis between 2nd and 3rd client
        ax.text((client_xs[1] + client_xs[2])/2, 1.83, "···", fontsize=9.0, ha="center", va="center", color="#94A3B8")

    # Ellipsis dots between bottom groups
    ax.text(5.47, 1.75, "•  •  •", fontsize=15.0, ha="center", va="center", color="#64748B")
    ax.text(10.02, 1.75, "•  •  •", fontsize=15.0, ha="center", va="center", color="#64748B")

    # =========================================================================
    # 4. CURVED COMMUNICATION ARROWS (TIER-TO-TIER DUAL FLOWS)
    # =========================================================================
    # 4.1 Server <-> Middle Cluster 1
    draw_curved_arrow(ax, (6.55, 6.30), (3.30, 5.35), is_dashed=True, rad=-0.14)
    draw_curved_arrow(ax, (3.70, 5.35), (6.75, 6.30), is_dashed=False, rad=0.06)

    # 4.2 Server <-> Middle Cluster k (Center)
    draw_curved_arrow(ax, (7.55, 6.30), (7.55, 5.35), is_dashed=True, rad=0.16)
    draw_curved_arrow(ax, (7.95, 5.35), (7.95, 6.30), is_dashed=False, rad=0.16)

    # 4.3 Server <-> Middle Cluster K (Right)
    draw_curved_arrow(ax, (8.75, 6.30), (12.00, 5.35), is_dashed=True, rad=0.14)
    draw_curved_arrow(ax, (11.60, 5.35), (8.55, 6.30), is_dashed=False, rad=-0.06)

    # 4.4 Defense link to Server
    ax.annotate("", xy=(6.05, 7.30), xytext=(3.65, 7.30),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#DC2626"))
    ax.text(4.85, 7.50, "Filtered Updates", fontsize=8.5, fontweight="bold", ha="center", color="#DC2626")

    # 4.5 Middle Tier <-> Bottom Tier (Clusters to Edge Clients)
    # Cluster 1 <-> Clients
    draw_curved_arrow(ax, (2.70, 3.35), (2.25, 2.60), is_dashed=True, rad=0.15)
    draw_curved_arrow(ax, (2.45, 2.60), (2.90, 3.35), is_dashed=False, rad=-0.15)

    draw_curved_arrow(ax, (3.50, 3.35), (4.05, 2.60), is_dashed=True, rad=-0.15)
    draw_curved_arrow(ax, (4.25, 2.60), (3.70, 3.35), is_dashed=False, rad=0.15)

    # Cluster k <-> Clients
    draw_curved_arrow(ax, (7.25, 3.35), (6.80, 2.60), is_dashed=True, rad=0.15)
    draw_curved_arrow(ax, (7.00, 2.60), (7.45, 3.35), is_dashed=False, rad=-0.15)

    draw_curved_arrow(ax, (8.05, 3.35), (8.60, 2.60), is_dashed=True, rad=-0.15)
    draw_curved_arrow(ax, (8.80, 2.60), (8.25, 3.35), is_dashed=False, rad=0.15)

    # Cluster K <-> Clients
    draw_curved_arrow(ax, (11.80, 3.35), (11.35, 2.60), is_dashed=True, rad=0.15)
    draw_curved_arrow(ax, (11.55, 2.60), (12.00, 3.35), is_dashed=False, rad=-0.15)

    draw_curved_arrow(ax, (12.60, 3.35), (13.15, 2.60), is_dashed=True, rad=-0.15)
    draw_curved_arrow(ax, (13.35, 2.60), (12.80, 3.35), is_dashed=False, rad=0.15)

    # =========================================================================
    # 5. BOTTOM CLIENT SPECIFICATION CALLOUT (FEDHEP EDGE MECHANISMS)
    # =========================================================================
    spec_box = FancyBboxPatch((1.6, 0.35), 12.3, 0.45,
                              boxstyle="round,pad=0.03,rounding_size=0.06",
                              facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2, zorder=2)
    ax.add_patch(spec_box)
    ax.text(7.75, 0.57, "On-Device Client: Single Shared Backbone  •  Active-Class Logit Masking (ACLM)  •  Private Local Head (100% On-Device)",
            fontsize=9.2, fontweight="bold", ha="center", va="center", color="#334155")

    # Output file paths
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
        print(f"[ok] Saved simplistic architecture diagram to {p}")

    plt.close(fig)

if __name__ == "__main__":
    build_simplistic_fedhep_diagram()
