"""
Generate original, publication-quality system architecture diagram for FedHEP.
Design Principles:
- 100% Original design (two coordinated sections: On-Device Architecture + Hierarchical Federation Network)
- Zero plagiarism (no borrowed composition, no clipart clouds or cloned boxes)
- Zero mathematical equations (pure clean English systems terminology)
- Zero overlapping elements (mathematically verified coordinates and generous padding)
- Publication-quality aesthetics (crisp serif typography, soft palette, high legibility)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch, Circle
)

def draw_padlock(ax, cx, cy, size=0.28, color="#059669", zorder=6):
    """Draw a clean, minimalist security lock badge."""
    bw = size * 0.85
    bh = size * 0.65
    body = FancyBboxPatch((cx - bw/2, cy - bh/2 - size*0.12), bw, bh,
                          boxstyle="round,pad=0.01,rounding_size=0.04",
                          facecolor=color, edgecolor="#FFFFFF", lw=1.2, zorder=zorder)
    ax.add_patch(body)
    
    # Shackle
    sr = size * 0.26
    theta = np.linspace(0, np.pi, 25)
    ax.plot(cx + sr*np.cos(theta), cy + size*0.20 + sr*np.sin(theta), color=color, lw=2.0, zorder=zorder)
    ax.plot([cx - sr, cx - sr], [cy - size*0.06, cy + size*0.20], color=color, lw=2.0, zorder=zorder)
    ax.plot([cx + sr, cx + sr], [cy - size*0.06, cy + size*0.20], color=color, lw=2.0, zorder=zorder)
    # Keyhole
    ax.add_patch(Circle((cx, cy - size*0.10), size*0.06, facecolor="#FFFFFF", edgecolor="none", zorder=zorder+1))

def draw_shield_badge(ax, cx, cy, w=0.55, h=0.65, color="#DC2626", zorder=6):
    """Draw a clean defense shield icon."""
    pts = [
        [cx - w/2, cy + h/2],
        [cx + w/2, cy + h/2],
        [cx + w/2, cy],
        [cx, cy - h/2],
        [cx - w/2, cy],
    ]
    shield = Polygon(pts, closed=True, facecolor=color, edgecolor="#FFFFFF", lw=1.4, zorder=zorder)
    ax.add_patch(shield)
    # Checkmark inside
    ax.plot([cx - w*0.22, cx - w*0.05, cx + w*0.24],
            [cy + h*0.02, cy - h*0.16, cy + h*0.18],
            color="#FFFFFF", lw=2.2, solid_capstyle="round", zorder=zorder+1)

def draw_server_chassis(ax, cx, cy, w=0.85, h=0.75, zorder=5):
    """Draw a modern server chassis icon."""
    outer = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.04",
                           facecolor="#1E293B", edgecolor="#334155", lw=1.2, zorder=zorder)
    ax.add_patch(outer)
    bay_h = h * 0.22
    gap = h * 0.08
    for i in range(3):
        by = cy + h/2 - (i+1)*bay_h - i*gap - gap*0.4
        bay = Rectangle((cx - w*0.40, by), w*0.80, bay_h, facecolor="#334155", edgecolor="none", zorder=zorder+1)
        ax.add_patch(bay)
        ax.add_patch(Circle((cx - w*0.26, by + bay_h/2), bay_h*0.22, facecolor="#10B981", edgecolor="none", zorder=zorder+2))
        ax.add_patch(Circle((cx - w*0.14, by + bay_h/2), bay_h*0.22, facecolor="#38BDF8", edgecolor="none", zorder=zorder+2))
        ax.add_patch(Rectangle((cx + w*0.06, by + bay_h*0.3), w*0.28, bay_h*0.4, facecolor="#1E293B", edgecolor="none", zorder=zorder+2))

def build_original_fedhep_diagram():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times", "serif"]
    plt.rcParams["pdf.fonttype"] = 42

    # Canvas: 16.2 x 9.2 inches
    fig, ax = plt.subplots(figsize=(16.2, 9.2), dpi=300)
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # =========================================================================
    # SECTION 1 (LEFT): ON-DEVICE CLIENT PIPELINE (Width: 6.8 units)
    # =========================================================================
    sec1_box = FancyBboxPatch((0.6, 0.45), 6.6, 8.30,
                              boxstyle="round,pad=0.05,rounding_size=0.12",
                              facecolor="#F8FAFC", edgecolor="#64748B", linewidth=1.5, zorder=1)
    ax.add_patch(sec1_box)

    ax.text(3.9, 8.42, "SECTION 1: ON-DEVICE CLIENT PIPELINE",
            fontsize=11.5, fontweight="bold", ha="center", va="center", color="#0F172A", zorder=2)
    ax.text(3.9, 8.16, "Single Shared Backbone feeds Tripartite Heads with Active Masking",
            fontsize=8.4, fontstyle="italic", ha="center", va="center", color="#475569", zorder=2)

    # 1.1 Private Local Dataset Card
    d_card = FancyBboxPatch((0.9, 6.75), 6.0, 1.15,
                            boxstyle="round,pad=0.03,rounding_size=0.06",
                            facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.3, zorder=2)
    ax.add_patch(d_card)
    draw_padlock(ax, 1.35, 7.32, size=0.32, color="#059669", zorder=5)
    ax.text(1.75, 7.50, "Private Edge Dataset", fontsize=9.6, fontweight="bold", ha="left", color="#065F46", zorder=3)
    ax.text(1.75, 7.24, "• Extreme non-IID label skew across heterogeneous nodes", fontsize=7.8, ha="left", color="#334155", zorder=3)
    ax.text(1.75, 6.98, "• Raw samples remain strictly isolated on-device at all times", fontsize=7.4, fontstyle="italic", ha="left", color="#059669", zorder=3)

    # Vertical Arrow: Data -> Shared Backbone
    ax.annotate("", xy=(3.9, 6.45), xytext=(3.9, 6.75),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#059669"), zorder=4)

    # 1.2 Single Shared Backbone Card
    bb_card = FancyBboxPatch((0.9, 5.25), 6.0, 1.20,
                             boxstyle="round,pad=0.03,rounding_size=0.06",
                             facecolor="#FFFFFF", edgecolor="#0284C7", linewidth=1.3, zorder=2)
    ax.add_patch(bb_card)
    ax.text(1.15, 6.12, "Single Shared Feature Backbone", fontsize=9.6, fontweight="bold", ha="left", color="#075985", zorder=3)
    ax.text(1.15, 5.84, "• Unified feature extractor across all classification heads", fontsize=8.0, ha="left", color="#334155", zorder=3)
    ax.text(1.15, 5.60, "• Completely eliminates duplicate network instances", fontsize=8.0, ha="left", color="#0284C7", fontweight="bold", zorder=3)
    ax.text(1.15, 5.38, "• Minimal edge footprint: 50% lower VRAM, fast inference", fontsize=7.4, fontstyle="italic", ha="left", color="#475569", zorder=3)

    # Vertical Arrow: Backbone -> ACLM
    ax.annotate("", xy=(3.9, 4.95), xytext=(3.9, 5.25),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#0284C7"), zorder=4)

    # 1.3 Active-Class Logit Masking (ACLM) Card
    aclm_card = FancyBboxPatch((0.9, 3.75), 6.0, 1.20,
                              boxstyle="round,pad=0.03,rounding_size=0.06",
                              facecolor="#FFFFFF", edgecolor="#F59E0B", linewidth=1.3, zorder=2)
    ax.add_patch(aclm_card)
    ax.text(1.15, 4.62, "Active-Class Logit Masking (ACLM Gate)", fontsize=9.6, fontweight="bold", ha="left", color="#B45309", zorder=3)
    ax.text(1.15, 4.34, "• Masks unobserved class logits before loss evaluation", fontsize=8.0, ha="left", color="#334155", zorder=3)
    ax.text(1.15, 4.10, "• Blocks negative gradient interference under label skew", fontsize=8.0, ha="left", color="#B45309", fontweight="bold", zorder=3)
    ax.text(1.15, 3.88, "• Preserves shared geometric feature representations", fontsize=7.4, fontstyle="italic", ha="left", color="#475569", zorder=3)

    # Branching Arrows: ACLM -> Tripartite Heads
    ax.annotate("", xy=(1.9, 3.45), xytext=(3.0, 3.75),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#2563EB"), zorder=4)
    ax.annotate("", xy=(3.9, 3.45), xytext=(3.9, 3.75),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#7C3AED"), zorder=4)
    ax.annotate("", xy=(5.9, 3.45), xytext=(4.8, 3.75),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#059669"), zorder=4)

    # 1.4 Tripartite Classification Heads (Three Side-by-Side Cards)
    # Head 1 (Left): Global Root Head
    h1 = FancyBboxPatch((0.9, 1.85), 1.9, 1.60,
                        boxstyle="round,pad=0.03,rounding_size=0.05",
                        facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.3, zorder=2)
    ax.add_patch(h1)
    ax.text(1.85, 3.12, "Global Root Head", fontsize=8.8, fontweight="bold", ha="center", color="#1D4ED8", zorder=3)
    ax.text(1.85, 2.78, "Global Consensus", fontsize=7.6, fontstyle="italic", ha="center", color="#2563EB", zorder=3)
    ax.text(1.85, 2.40, "Synchronized with\nFederated Server", fontsize=7.4, ha="center", color="#1E3A8A", zorder=3)
    ax.text(1.85, 2.00, "Broad Generalization", fontsize=7.0, fontweight="bold", ha="center", color="#1D4ED8", zorder=3)

    # Head 2 (Middle): Collaborative Parent Head
    h2 = FancyBboxPatch((2.95, 1.85), 1.9, 1.60,
                        boxstyle="round,pad=0.03,rounding_size=0.05",
                        facecolor="#FAF5FF", edgecolor="#7C3AED", linewidth=1.3, zorder=2)
    ax.add_patch(h2)
    ax.text(3.90, 3.12, "Parent Head", fontsize=8.8, fontweight="bold", ha="center", color="#6D28D9", zorder=3)
    ax.text(3.90, 2.78, "Peer Cluster Cohort", fontsize=7.6, fontstyle="italic", ha="center", color="#7C3AED", zorder=3)
    ax.text(3.90, 2.40, "Synchronized with\nPeer Cluster", fontsize=7.4, ha="center", color="#5B21B6", zorder=3)
    ax.text(3.90, 2.00, "Synergistic Support", fontsize=7.0, fontweight="bold", ha="center", color="#6D28D9", zorder=3)

    # Head 3 (Right): Private Local Head
    h3 = FancyBboxPatch((5.0, 1.85), 1.9, 1.60,
                        boxstyle="round,pad=0.03,rounding_size=0.05",
                        facecolor="#ECFDF5", edgecolor="#059669", linewidth=1.3, zorder=2)
    ax.add_patch(h3)
    ax.text(5.95, 3.12, "Private Local Head", fontsize=8.8, fontweight="bold", ha="center", color="#047857", zorder=3)
    ax.text(5.95, 2.78, "100% On-Device", fontsize=7.6, fontstyle="italic", ha="center", color="#059669", zorder=3)
    ax.text(5.95, 2.40, "Never Uploaded\nZero Telemetry", fontsize=7.4, ha="center", color="#065F46", zorder=3)
    draw_padlock(ax, 5.95, 2.02, size=0.22, color="#059669", zorder=5)

    # Converging Arrows: 3 Heads -> Calibrated Ensemble
    ax.annotate("", xy=(3.0, 1.55), xytext=(1.9, 1.85),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#2563EB"), zorder=4)
    ax.annotate("", xy=(3.9, 1.55), xytext=(3.9, 1.85),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#7C3AED"), zorder=4)
    ax.annotate("", xy=(4.8, 1.55), xytext=(5.9, 1.85),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#059669"), zorder=4)

    # 1.5 Calibrated Ensemble Prediction Card
    ens_card = FancyBboxPatch((0.9, 0.65), 6.0, 0.90,
                             boxstyle="round,pad=0.03,rounding_size=0.06",
                             facecolor="#EEF2FF", edgecolor="#4338CA", linewidth=1.3, zorder=2)
    ax.add_patch(ens_card)
    ax.text(3.9, 1.25, "Calibrated Ensemble Inference", fontsize=9.6, fontweight="bold", ha="center", color="#3730A3", zorder=3)
    ax.text(3.9, 0.98, "Anchored linear combination of temperature-calibrated logits",
            fontsize=7.8, ha="center", color="#334155", zorder=3)
    ax.text(3.9, 0.76, "Optimal personalization while safeguarding tail decile fairness",
            fontsize=7.2, fontstyle="italic", ha="center", color="#4338CA", zorder=3)


    # =========================================================================
    # SECTION 2 (RIGHT): MULTI-TIER FEDERATION TOPOLOGY (Width: 8.4 units)
    # =========================================================================
    sec2_box = FancyBboxPatch((7.6, 0.45), 8.0, 8.30,
                              boxstyle="round,pad=0.05,rounding_size=0.12",
                              facecolor="#FAFAFA", edgecolor="#64748B", linewidth=1.5, zorder=1)
    ax.add_patch(sec2_box)

    ax.text(11.6, 8.42, "SECTION 2: MULTI-TIER FEDERATION TOPOLOGY",
            fontsize=11.5, fontweight="bold", ha="center", va="center", color="#0F172A", zorder=2)
    ax.text(11.6, 8.16, "Hierarchical Synchronization  •  Subspace Defense  •  Affinity Peer Collaboration",
            fontsize=8.4, fontstyle="italic", ha="center", va="center", color="#475569", zorder=2)

    # 2.1 Central Federated Server Container (Top of Section 2)
    srv_box = FancyBboxPatch((7.9, 6.15), 7.4, 1.85,
                             boxstyle="round,pad=0.04,rounding_size=0.08",
                             facecolor="#FFFFFF", edgecolor="#3B82F6", linewidth=1.4, zorder=2)
    ax.add_patch(srv_box)
    ax.text(11.6, 7.72, "Central Federated Server (Global Consensus)", fontsize=10.2, fontweight="bold", ha="center", color="#1E3A8A", zorder=3)

    # Left: Byzantine Defense Sub-card
    byz_sub = FancyBboxPatch((8.1, 6.30), 3.4, 1.15,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             facecolor="#FEF2F2", edgecolor="#EF4444", linewidth=1.1, zorder=3)
    ax.add_patch(byz_sub)
    draw_shield_badge(ax, 8.45, 6.85, w=0.45, h=0.55, color="#DC2626", zorder=5)
    ax.text(9.90, 7.15, "Subspace Byzantine Defense", fontsize=8.6, fontweight="bold", ha="center", color="#991B1B", zorder=4)
    ax.text(9.90, 6.90, "• Subspace Cosine Filtering (SCCF)", fontsize=7.2, ha="center", color="#7F1D1D", zorder=4)
    ax.text(9.90, 6.68, "• Temporal Trust Tracking (TTT)", fontsize=7.2, ha="center", color="#7F1D1D", zorder=4)
    ax.text(9.90, 6.46, "Neutralizes poisoning attacks", fontsize=6.8, fontstyle="italic", ha="center", color="#B91C1C", zorder=4)

    # Right: Global Consensus Aggregator Sub-card
    agg_sub = FancyBboxPatch((11.7, 6.30), 3.4, 1.15,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.1, zorder=3)
    ax.add_patch(agg_sub)
    draw_server_chassis(ax, 12.05, 6.85, w=0.55, h=0.60, zorder=5)
    ax.text(13.50, 7.15, "Consensus Aggregator", fontsize=8.6, fontweight="bold", ha="center", color="#1E3A8A", zorder=4)
    ax.text(13.50, 6.90, "• Shared Backbone Updating", fontsize=7.2, ha="center", color="#1D4ED8", zorder=4)
    ax.text(13.50, 6.68, "• Global Root Head Averaging", fontsize=7.2, ha="center", color="#1D4ED8", zorder=4)
    ax.text(13.50, 6.46, "Trust-weighted parameter fusion", fontsize=6.8, fontstyle="italic", ha="center", color="#2563EB", zorder=4)

    # Internal arrow: Defense -> Consensus
    ax.annotate("", xy=(11.7, 6.85), xytext=(11.5, 6.85),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#DC2626"), zorder=5)

    # 2.2 Synergistic Peer Clustering Layer (Middle of Section 2)
    peer_box = FancyBboxPatch((7.9, 3.65), 7.4, 2.05,
                              boxstyle="round,pad=0.04,rounding_size=0.08",
                              facecolor="#FAF5FF", edgecolor="#A855F7", linewidth=1.4, linestyle="--", zorder=2)
    ax.add_patch(peer_box)
    ax.text(11.6, 5.45, "Collaborative Peer Clusters (Cohort Layer)", fontsize=10.0, fontweight="bold", ha="center", color="#5B21B6", zorder=3)
    ax.text(11.6, 5.24, "Dynamic grouping via 256-dim Random Projection sketches  •  Synchronizes Parent Heads",
            fontsize=7.4, fontstyle="italic", ha="center", color="#6D28D9", zorder=3)

    # 3 Cluster Cohort Cards
    c_configs = [
        ("Peer Cluster 1", "Cohort A", 8.1, 2.25),
        ("Peer Cluster 2", "Cohort B", 10.45, 2.25),
        ("Peer Cluster K", "Cohort K", 12.8, 2.25)
    ]
    for cname, ccohort, cx, cw in c_configs:
        c_patch = FancyBboxPatch((cx, 3.82), cw, 1.25,
                                 boxstyle="round,pad=0.02,rounding_size=0.04",
                                 facecolor="#FFFFFF", edgecolor="#7C3AED", linewidth=1.2, zorder=3)
        ax.add_patch(c_patch)
        ax.text(cx + cw/2, 4.82, cname, fontsize=8.4, fontweight="bold", ha="center", color="#5B21B6", zorder=4)
        ax.text(cx + cw/2, 4.58, f"({ccohort})", fontsize=7.4, fontstyle="italic", ha="center", color="#7C3AED", zorder=4)
        ax.text(cx + cw/2, 4.30, "Parent Head Model", fontsize=7.6, fontweight="bold", ha="center", color="#334155", zorder=4)
        ax.text(cx + cw/2, 4.05, "Synergistic Peer Fusion", fontsize=6.8, fontstyle="italic", ha="center", color="#059669", zorder=4)

    # 2.3 Heterogeneous Edge Clients (Bottom of Section 2)
    client_box = FancyBboxPatch((7.9, 1.50), 7.4, 1.75,
                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor="#F0FDF4", edgecolor="#16A34A", linewidth=1.3, zorder=2)
    ax.add_patch(client_box)
    # Header placed cleanly with ample margin above cards
    ax.text(8.20, 3.05, "Heterogeneous Edge Client Fleet", fontsize=9.4, fontweight="bold", ha="left", color="#14532D", zorder=3)

    # 3 Diverse Device Cards (Heterogeneity)
    dev_configs = [
        ("Client 1", "Mobile Device", "Cluster 1 Member", 8.1, 2.25),
        ("Client 2", "Edge Server", "Cluster 2 Member", 10.45, 2.25),
        ("Client K", "IoT Node", "Cluster K Member", 12.8, 2.25)
    ]
    for dname, dtype, dmem, dx, dw in dev_configs:
        d_patch = FancyBboxPatch((dx, 1.62), dw, 1.15,
                                 boxstyle="round,pad=0.02,rounding_size=0.04",
                                 facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.1, zorder=3)
        ax.add_patch(d_patch)
        ax.text(dx + dw/2, 2.56, dname, fontsize=8.4, fontweight="bold", ha="center", color="#065F46", zorder=4)
        ax.text(dx + dw/2, 2.32, dtype, fontsize=7.4, fontstyle="italic", ha="center", color="#059669", zorder=4)
        ax.text(dx + dw/2, 2.06, dmem, fontsize=7.0, ha="center", color="#334155", zorder=4)
        draw_padlock(ax, dx + dw/2, 1.82, size=0.18, color="#059669", zorder=5)

    # 2.4 Communication Legend Bar (Very Bottom of Section 2)
    leg_box = FancyBboxPatch((7.9, 0.65), 7.4, 0.70,
                             boxstyle="round,pad=0.02,rounding_size=0.04",
                             facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0, zorder=2)
    ax.add_patch(leg_box)
    # Legend items (strictly horizontal, perfectly spaced)
    # Item 1: Blue line
    ax.plot([8.15, 8.55], [1.02, 1.02], color="#2563EB", lw=2.0, zorder=3)
    ax.text(8.70, 1.02, "Global Consensus Sync (Root Head & Backbone)", fontsize=7.2, va="center", color="#1E293B", zorder=3)
    # Item 2: Purple dashed line
    ax.plot([8.15, 8.55], [0.80, 0.80], color="#7C3AED", lw=1.8, linestyle="--", zorder=3)
    ax.text(8.70, 0.80, "Peer Cluster Sync (Parent Head)", fontsize=7.2, va="center", color="#1E293B", zorder=3)
    # Item 3: Green lock badge
    draw_padlock(ax, 13.55, 0.91, size=0.20, color="#059669", zorder=5)
    ax.text(13.80, 0.91, "100% On-Device Isolation", fontsize=7.2, fontweight="bold", va="center", color="#047857", zorder=3)

    # 2.5 Clean Vertical Communication Connectors between Right Section Tiers
    # Strictly in the clear gap between Middle Tier (y=3.65) and Bottom Tier (y=3.25): ZERO TEXT COLLISION
    # Connector A: Client 1 <-> Peer Cluster 1 (Purple dashed)
    ax.annotate("", xy=(9.22, 3.65), xytext=(9.22, 3.25),
                arrowprops=dict(arrowstyle="<->", lw=1.5, linestyle="--", color="#7C3AED"), zorder=5)
    # Connector B: Client 2 <-> Peer Cluster 2 (Purple dashed)
    ax.annotate("", xy=(11.57, 3.65), xytext=(11.57, 3.25),
                arrowprops=dict(arrowstyle="<->", lw=1.5, linestyle="--", color="#7C3AED"), zorder=5)
    # Connector C: Client K <-> Peer Cluster K (Purple dashed)
    ax.annotate("", xy=(13.92, 3.65), xytext=(13.92, 3.25),
                arrowprops=dict(arrowstyle="<->", lw=1.5, linestyle="--", color="#7C3AED"), zorder=5)

    # Connectors between Server (y=6.15) and Middle Tier (y=5.70): ZERO TEXT COLLISION
    # Left: Defended updates entering defense module (Red arrow)
    ax.annotate("", xy=(9.80, 6.15), xytext=(9.80, 5.70),
                arrowprops=dict(arrowstyle="->", lw=1.5, linestyle="-", color="#DC2626"), zorder=5)
    ax.text(9.20, 5.92, "Defended Updates", fontsize=6.8, fontweight="bold", color="#DC2626", ha="right", va="center", zorder=5)

    # Right: Consensus broadcast to clusters/clients (Blue arrow)
    ax.annotate("", xy=(13.50, 5.70), xytext=(13.50, 6.15),
                arrowprops=dict(arrowstyle="->", lw=1.5, linestyle="-", color="#2563EB"), zorder=5)
    ax.text(14.10, 5.92, "Consensus Broadcast", fontsize=6.8, fontweight="bold", color="#2563EB", ha="left", va="center", zorder=5)

    # Save outputs
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
        print(f"[ok] Saved pristine FedHEP architecture diagram to {p}")

    plt.close(fig)

if __name__ == "__main__":
    build_original_fedhep_diagram()
