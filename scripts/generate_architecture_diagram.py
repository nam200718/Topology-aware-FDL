"""
Generate comprehensive yet simple, highly symbolic, publication-quality system architecture diagram for FedHEP.

Includes EVERY core component of FedHEP:
1. Private Local Dataset & Label Skew Profiling (R_skew)
2. Single Shared Feature Backbone (Phi_theta)
3. Active-Class Logit Masking (ACLM)
4. Tripartite Classification Heads (Global Root W_r, Cluster Parent W_p, Private Local W_l)
5. Calibrated Ensemble Inference (Temperature Scaling, Adaptive Weights, Prediction)
6. Heterogeneous Edge Fleet (Mobile, Edge Server, IoT)
7. Privacy-Preserving Random Projection Sketching (JL Lemma dimension reduction s_i)
8. Collaborative Peer Clustering (Cohort Parent Head sharing)
9. Layered Subspace Byzantine Defense (Subspace SCCF, Adaptive Norm-Bounding, Temporal Trust TTT)
10. Global Consensus Aggregation (Trust-weighted Backbone & Root updates)

Design Principles:
- Symbolic, iconic visual representations
- Ultra-low text density (punchy titles, concise tags, zero paragraphs)
- Zero mathematical equations
- Mathematically verified geometry: zero overlapping text or icons
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch, Circle, Ellipse
)

# -----------------------------------------------------------------------------
# PROCEDURAL SYMBOL DRAWING PRIMITIVES
# -----------------------------------------------------------------------------

def draw_padlock(ax, cx, cy, size=0.22, color="#059669", zorder=6):
    """Draw a clean, minimalist security padlock."""
    bw = size * 0.82
    bh = size * 0.62
    body = FancyBboxPatch((cx - bw/2, cy - bh/2 - size*0.12), bw, bh,
                          boxstyle="round,pad=0.01,rounding_size=0.04",
                          facecolor=color, edgecolor="#FFFFFF", lw=1.2, zorder=zorder)
    ax.add_patch(body)
    
    sr = size * 0.25
    theta = np.linspace(0, np.pi, 25)
    ax.plot(cx + sr*np.cos(theta), cy + size*0.19 + sr*np.sin(theta), color=color, lw=1.8, zorder=zorder)
    ax.plot([cx - sr, cx - sr], [cy - size*0.06, cy + size*0.19], color=color, lw=1.8, zorder=zorder)
    ax.plot([cx + sr, cx + sr], [cy - size*0.06, cy + size*0.19], color=color, lw=1.8, zorder=zorder)
    ax.add_patch(Circle((cx, cy - size*0.10), size*0.055, facecolor="#FFFFFF", edgecolor="none", zorder=zorder+1))

def draw_shield_badge(ax, cx, cy, w=0.52, h=0.60, color="#DC2626", zorder=6):
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
    ax.plot([cx - w*0.22, cx - w*0.05, cx + w*0.24],
            [cy + h*0.02, cy - h*0.16, cy + h*0.18],
            color="#FFFFFF", lw=2.2, solid_capstyle="round", zorder=zorder+1)

def draw_server_chassis(ax, cx, cy, w=0.66, h=0.58, zorder=5):
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
        ax.add_patch(Circle((cx - w*0.25, by + bay_h/2), bay_h*0.22, facecolor="#10B981", edgecolor="none", zorder=zorder+2))
        ax.add_patch(Circle((cx - w*0.12, by + bay_h/2), bay_h*0.22, facecolor="#38BDF8", edgecolor="none", zorder=zorder+2))
        ax.add_patch(Rectangle((cx + w*0.06, by + bay_h*0.3), w*0.28, bay_h*0.4, facecolor="#1E293B", edgecolor="none", zorder=zorder+2))

def draw_database_symbol(ax, cx, cy, w=0.75, h=0.55, color="#059669", zorder=4):
    """Draw a 3-layer database platter stack."""
    layer_h = h * 0.28
    gap = h * 0.08
    for i in range(3):
        dy = cy - h/2 + i * (layer_h + gap)
        body = Rectangle((cx - w/2, dy), w, layer_h, facecolor=color, edgecolor=color, lw=1.0, zorder=zorder)
        ax.add_patch(body)
        top = Ellipse((cx, dy + layer_h), w, layer_h * 0.70, facecolor="#34D399" if i==2 else color, edgecolor="#FFFFFF", lw=1.0, zorder=zorder+1)
        ax.add_patch(top)
        if i == 0:
            bot = Ellipse((cx, dy), w, layer_h * 0.70, facecolor=color, edgecolor="none", zorder=zorder)
            ax.add_patch(bot)

def draw_skew_gauge(ax, cx, cy, size=0.36, color="#059669", zorder=5):
    """Draw a clean label entropy skew dial/gauge icon."""
    # Semi-circle gauge
    theta = np.linspace(0, np.pi, 30)
    r = size
    ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), color=color, lw=1.6, zorder=zorder)
    # Dial needle pointing towards high-skew
    ax.plot([cx, cx + r*0.65*np.cos(np.pi*0.72)], [cy, cy + r*0.65*np.sin(np.pi*0.72)],
            color="#D97706", lw=1.8, solid_capstyle="round", zorder=zorder+1)
    ax.add_patch(Circle((cx, cy), size*0.14, facecolor=color, edgecolor="none", zorder=zorder+2))

def draw_cnn_pyramid(ax, cx, cy, w=2.1, h=0.75, color="#0284C7", zorder=4):
    """Draw a multi-stage feature extractor symbol (input patch -> conv maps -> vector)."""
    inp_x = cx - w*0.42
    inp = FancyBboxPatch((inp_x - 0.16, cy - 0.20), 0.32, 0.40,
                         boxstyle="round,pad=0.01,rounding_size=0.02",
                         facecolor="#F1F5F9", edgecolor="#64748B", lw=1.0, zorder=zorder)
    ax.add_patch(inp)
    ax.text(inp_x, cy, "Data", fontsize=6.8, fontweight="bold", ha="center", va="center", color="#475569", zorder=zorder+1)
    
    block_sizes = [(0.44, 0.58), (0.34, 0.46)]
    xs = [cx - w*0.14, cx + w*0.14]
    
    ax.plot([inp_x + 0.16, xs[0] - block_sizes[0][0]/2], [cy, cy], color=color, lw=1.4, zorder=zorder+1)

    for (bw, bh), bx in zip(block_sizes, xs):
        patch = FancyBboxPatch((bx - bw/2, cy - bh/2), bw, bh,
                               boxstyle="round,pad=0.01,rounding_size=0.03",
                               facecolor="#E0F2FE", edgecolor=color, lw=1.4, zorder=zorder)
        ax.add_patch(patch)
        ax.plot([bx - bw*0.2, bx - bw*0.2], [cy - bh*0.3, cy + bh*0.3], color=color, lw=0.9, alpha=0.6, zorder=zorder+1)
    
    ax.plot([xs[0] + block_sizes[0][0]/2, xs[1] - block_sizes[1][0]/2], [cy, cy], color=color, lw=1.4, zorder=zorder+1)
    
    emb_x = cx + w*0.42
    emb = FancyBboxPatch((emb_x - 0.08, cy - 0.28), 0.16, 0.56,
                         boxstyle="round,pad=0.01,rounding_size=0.02",
                         facecolor=color, edgecolor="#0369A1", lw=1.1, zorder=zorder)
    ax.add_patch(emb)
    ax.plot([xs[1] + block_sizes[1][0]/2, emb_x - 0.08], [cy, cy], color=color, lw=1.4, zorder=zorder+1)

def draw_aclm_gate_symbol(ax, cx, cy, w=2.0, h=0.65, zorder=4):
    """Draw an Active-Class Logit Masking matrix/gate symbol."""
    slot_w = w * 0.16
    slot_h = h * 0.72
    start_x = cx - w*0.42
    gap = w * 0.04
    active_mask = [True, True, False, False, True]
    for i, is_act in enumerate(active_mask):
        sx = start_x + i * (slot_w + gap)
        bg = "#FEF3C7" if is_act else "#F1F5F9"
        border = "#F59E0B" if is_act else "#94A3B8"
        slot = FancyBboxPatch((sx, cy - slot_h/2), slot_w, slot_h,
                              boxstyle="round,pad=0.01,rounding_size=0.03",
                              facecolor=bg, edgecolor=border, lw=1.2, zorder=zorder)
        ax.add_patch(slot)
        if is_act:
            ax.plot([sx + slot_w*0.25, sx + slot_w*0.45, sx + slot_w*0.75],
                    [cy - slot_h*0.05, cy - slot_h*0.22, cy + slot_h*0.22],
                    color="#D97706", lw=2.0, solid_capstyle="round", zorder=zorder+1)
        else:
            ax.plot([sx + slot_w*0.25, sx + slot_w*0.75], [cy - slot_h*0.22, cy + slot_h*0.22],
                    color="#94A3B8", lw=1.8, solid_capstyle="round", zorder=zorder+1)
            ax.plot([sx + slot_w*0.25, sx + slot_w*0.75], [cy + slot_h*0.22, cy - slot_h*0.22],
                    color="#94A3B8", lw=1.8, solid_capstyle="round", zorder=zorder+1)

def draw_globe_symbol(ax, cx, cy, r=0.25, color="#2563EB", zorder=4):
    """Draw a clean global consensus sphere symbol."""
    ax.add_patch(Circle((cx, cy), r, facecolor="#DBEAFE", edgecolor=color, lw=1.3, zorder=zorder))
    ax.add_patch(Ellipse((cx, cy), r*1.9, r*0.9, facecolor="none", edgecolor=color, lw=0.9, linestyle="--", zorder=zorder+1))
    ax.plot([cx, cx], [cy - r, cy + r], color=color, lw=1.0, zorder=zorder+1)
    ax.plot([cx - r, cx + r], [cy, cy], color=color, lw=1.0, zorder=zorder+1)

def draw_cohort_symbol(ax, cx, cy, r=0.25, color="#7C3AED", zorder=4):
    """Draw a peer cohort cluster graph symbol."""
    ax.add_patch(Circle((cx, cy), r, facecolor="#EDE9FE", edgecolor=color, lw=1.3, zorder=zorder))
    nr = 0.055
    p1 = (cx, cy + r*0.48)
    p2 = (cx - r*0.50, cy - r*0.38)
    p3 = (cx + r*0.50, cy - r*0.38)
    ax.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], color=color, lw=1.0, zorder=zorder+1)
    ax.add_patch(Circle(p1, nr, facecolor=color, edgecolor="none", zorder=zorder+2))
    ax.add_patch(Circle(p2, nr, facecolor=color, edgecolor="none", zorder=zorder+2))
    ax.add_patch(Circle(p3, nr, facecolor=color, edgecolor="none", zorder=zorder+2))

def draw_sketch_matrix(ax, cx, cy, w=0.68, h=0.48, zorder=4):
    """Draw a random projection matrix / sketch compression icon."""
    # Projection matrix grid
    mat = FancyBboxPatch((cx - w/2, cy - h/2), w*0.48, h,
                         boxstyle="round,pad=0.01,rounding_size=0.02",
                         facecolor="#FAF5FF", edgecolor="#7C3AED", lw=1.1, zorder=zorder)
    ax.add_patch(mat)
    # Grid lines inside matrix
    ax.plot([cx - w*0.24, cx], [cy, cy], color="#A855F7", lw=0.8, zorder=zorder+1)
    ax.plot([cx - w*0.12, cx - w*0.12], [cy - h*0.4, cy + h*0.4], color="#A855F7", lw=0.8, zorder=zorder+1)
    
    # Arrow R -> sketch
    ax.annotate("", xy=(cx + w*0.16, cy), xytext=(cx + w*0.02, cy),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#7C3AED"), zorder=zorder+1)
    
    # Compressed sketch vector s_i
    vec = FancyBboxPatch((cx + w*0.22, cy - h*0.42), w*0.20, h*0.84,
                         boxstyle="round,pad=0.01,rounding_size=0.02",
                         facecolor="#7C3AED", edgecolor="#5B21B6", lw=1.0, zorder=zorder)
    ax.add_patch(vec)

def draw_device_icon(ax, cx, cy, dev_type="phone", color="#059669", zorder=4):
    """Draw iconic edge device silhouettes."""
    if dev_type == "phone":
        pw, ph = 0.34, 0.52
        chassis = FancyBboxPatch((cx - pw/2, cy - ph/2), pw, ph,
                                 boxstyle="round,pad=0.01,rounding_size=0.06",
                                 facecolor="#1E293B", edgecolor=color, lw=1.4, zorder=zorder)
        ax.add_patch(chassis)
        screen = FancyBboxPatch((cx - pw*0.38, cy - ph*0.36), pw*0.76, ph*0.72,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor="#F8FAFC", edgecolor="none", zorder=zorder+1)
        ax.add_patch(screen)
        ax.plot([cx - pw*0.16, cx + pw*0.16], [cy - ph*0.42, cy - ph*0.42], color="#94A3B8", lw=1.3, zorder=zorder+2)
    elif dev_type == "server":
        sw, sh = 0.54, 0.48
        draw_server_chassis(ax, cx, cy, w=sw, h=sh, zorder=zorder)
    elif dev_type == "iot":
        iw, ih = 0.42, 0.42
        chip = FancyBboxPatch((cx - iw/2, cy - ih/2), iw, ih,
                              boxstyle="round,pad=0.01,rounding_size=0.04",
                              facecolor="#1E293B", edgecolor=color, lw=1.3, zorder=zorder)
        ax.add_patch(chip)
        core = Rectangle((cx - iw*0.26, cy - ih*0.26), iw*0.52, ih*0.52, facecolor=color, edgecolor="none", zorder=zorder+1)
        ax.add_patch(core)
        ax.plot([cx + iw*0.55, cx + iw*0.72, cx + iw*0.55], [cy + 0.12, cy, cy - 0.12], color=color, lw=1.4, zorder=zorder+2)
        ax.plot([cx + iw*0.76, cx + iw*0.96, cx + iw*0.76], [cy + 0.18, cy, cy - 0.18], color=color, lw=1.4, zorder=zorder+2)


# -----------------------------------------------------------------------------
# MAIN FIGURE GENERATOR
# -----------------------------------------------------------------------------

def build_symbolic_fedhep_diagram():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times", "serif"]
    plt.rcParams["pdf.fonttype"] = 42

    # Canvas dimensions: 16.2 x 9.5 inches
    fig, ax = plt.subplots(figsize=(16.2, 9.5), dpi=300)
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 9.5)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # =========================================================================
    # SECTION 1 (LEFT): ON-DEVICE CLIENT PIPELINE (Width: 6.8 units)
    # =========================================================================
    sec1_box = FancyBboxPatch((0.6, 0.45), 6.6, 8.60,
                              boxstyle="round,pad=0.04,rounding_size=0.12",
                              facecolor="#F8FAFC", edgecolor="#64748B", linewidth=1.5, zorder=1)
    ax.add_patch(sec1_box)

    ax.text(3.9, 8.78, "SECTION 1: ON-DEVICE CLIENT PIPELINE",
            fontsize=11.5, fontweight="bold", ha="center", va="center", color="#0F172A", zorder=2)
    ax.text(3.9, 8.52, "Shared Backbone • Skew-Aware Tripartite Heads • Active Masking",
            fontsize=8.5, fontstyle="italic", ha="center", va="center", color="#475569", zorder=2)

    # 1.1 Private Local Dataset & Skew Profiler Card
    d_card = FancyBboxPatch((0.9, 7.10), 6.0, 1.20,
                            boxstyle="round,pad=0.03,rounding_size=0.06",
                            facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.3, zorder=2)
    ax.add_patch(d_card)
    draw_database_symbol(ax, 1.60, 7.70, w=0.80, h=0.52, color="#059669", zorder=4)
    draw_padlock(ax, 2.22, 7.70, size=0.25, color="#059669", zorder=5)
    
    # Skew gauge on right of icon
    draw_skew_gauge(ax, 2.75, 7.62, size=0.28, color="#059669", zorder=4)
    
    ax.text(3.20, 7.82, "Private Dataset & Skew Profiler", fontsize=10.2, fontweight="bold", ha="left", color="#065F46", zorder=3)
    ax.text(3.20, 7.56, "Strict Isolation • Label Entropy Profiling", fontsize=8.4, fontstyle="italic", ha="left", color="#047857", zorder=3)

    # Vertical Arrow: Data -> Shared Backbone
    ax.annotate("", xy=(3.9, 6.78), xytext=(3.9, 7.10),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#059669"), zorder=4)

    # 1.2 Single Shared Backbone Card
    bb_card = FancyBboxPatch((0.9, 5.50), 6.0, 1.25,
                             boxstyle="round,pad=0.03,rounding_size=0.06",
                             facecolor="#FFFFFF", edgecolor="#0284C7", linewidth=1.3, zorder=2)
    ax.add_patch(bb_card)
    draw_cnn_pyramid(ax, 2.10, 6.12, w=2.0, h=0.72, color="#0284C7", zorder=4)
    ax.text(3.35, 6.27, "Shared Feature Backbone", fontsize=10.5, fontweight="bold", ha="left", color="#075985", zorder=3)
    ax.text(3.35, 5.99, "Unified Conv / Vision Extractor", fontsize=8.6, fontstyle="italic", ha="left", color="#0284C7", zorder=3)

    # Vertical Arrow: Backbone -> ACLM Gate
    ax.annotate("", xy=(3.9, 5.18), xytext=(3.9, 5.50),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#0284C7"), zorder=4)

    # 1.3 Active-Class Logit Masking (ACLM) Card
    aclm_card = FancyBboxPatch((0.9, 3.95), 6.0, 1.20,
                              boxstyle="round,pad=0.03,rounding_size=0.06",
                              facecolor="#FFFFFF", edgecolor="#F59E0B", linewidth=1.3, zorder=2)
    ax.add_patch(aclm_card)
    draw_aclm_gate_symbol(ax, 2.10, 4.55, w=2.0, h=0.68, zorder=4)
    ax.text(3.35, 4.70, "ACLM Masking Gate", fontsize=10.5, fontweight="bold", ha="left", color="#B45309", zorder=3)
    ax.text(3.35, 4.42, "Blocks Gradient Pollution on Unseen Classes", fontsize=8.6, fontstyle="italic", ha="left", color="#D97706", zorder=3)

    # Branching Arrows: ACLM -> Tripartite Heads
    ax.annotate("", xy=(1.9, 3.55), xytext=(3.0, 3.95),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#2563EB"), zorder=4)
    ax.annotate("", xy=(3.9, 3.55), xytext=(3.9, 3.95),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#7C3AED"), zorder=4)
    ax.annotate("", xy=(5.9, 3.55), xytext=(4.8, 3.95),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#059669"), zorder=4)

    # 1.4 Tripartite Classification Heads (Three Visual Symbolic Columns)
    # Head 1 (Left): Global Root Head
    h1 = FancyBboxPatch((0.9, 1.95), 1.9, 1.55,
                        boxstyle="round,pad=0.03,rounding_size=0.05",
                        facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.3, zorder=2)
    ax.add_patch(h1)
    draw_globe_symbol(ax, 1.85, 2.95, r=0.25, color="#2563EB", zorder=4)
    ax.text(1.85, 2.50, "Global Root Head", fontsize=8.8, fontweight="bold", ha="center", color="#1D4ED8", zorder=3)
    ax.text(1.85, 2.30, "Global Consensus", fontsize=7.6, fontstyle="italic", ha="center", color="#2563EB", zorder=3)
    ax.text(1.85, 2.10, "Skew-Weighted Loss", fontsize=7.2, ha="center", color="#1E40AF", zorder=3)

    # Head 2 (Middle): Collaborative Parent Head
    h2 = FancyBboxPatch((2.95, 1.95), 1.9, 1.55,
                        boxstyle="round,pad=0.03,rounding_size=0.05",
                        facecolor="#FAF5FF", edgecolor="#7C3AED", linewidth=1.3, zorder=2)
    ax.add_patch(h2)
    draw_cohort_symbol(ax, 3.90, 2.95, r=0.25, color="#7C3AED", zorder=4)
    ax.text(3.90, 2.50, "Parent Head", fontsize=8.8, fontweight="bold", ha="center", color="#6D28D9", zorder=3)
    ax.text(3.90, 2.30, "Cluster Cohort", fontsize=7.6, fontstyle="italic", ha="center", color="#7C3AED", zorder=3)
    ax.text(3.90, 2.10, "Skew-Weighted Loss", fontsize=7.2, ha="center", color="#5B21B6", zorder=3)

    # Head 3 (Right): Private Local Head
    h3 = FancyBboxPatch((5.0, 1.95), 1.9, 1.55,
                        boxstyle="round,pad=0.03,rounding_size=0.05",
                        facecolor="#ECFDF5", edgecolor="#059669", linewidth=1.3, zorder=2)
    ax.add_patch(h3)
    draw_padlock(ax, 5.95, 2.95, size=0.28, color="#059669", zorder=5)
    ax.text(5.95, 2.50, "Local Head", fontsize=8.8, fontweight="bold", ha="center", color="#047857", zorder=3)
    ax.text(5.95, 2.30, "Strictly On-Device", fontsize=7.6, fontstyle="italic", ha="center", color="#059669", zorder=3)
    ax.text(5.95, 2.10, "Zero Telemetry Sync", fontsize=7.2, ha="center", color="#065F46", zorder=3)

    # Converging Arrows: 3 Heads -> Ensemble Card
    ax.annotate("", xy=(3.0, 1.60), xytext=(1.9, 1.95),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#2563EB"), zorder=4)
    ax.annotate("", xy=(3.9, 1.60), xytext=(3.9, 1.95),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#7C3AED"), zorder=4)
    ax.annotate("", xy=(4.8, 1.60), xytext=(5.9, 1.95),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#059669"), zorder=4)

    # 1.5 Calibrated Ensemble Card (Symbolic Fusion Junction)
    ens_card = FancyBboxPatch((0.9, 0.65), 6.0, 0.92,
                             boxstyle="round,pad=0.03,rounding_size=0.06",
                             facecolor="#EEF2FF", edgecolor="#4338CA", linewidth=1.3, zorder=2)
    ax.add_patch(ens_card)
    # Fusion junction circle badge
    ax.add_patch(Circle((1.55, 1.11), 0.28, facecolor="#4338CA", edgecolor="#FFFFFF", lw=1.2, zorder=3))
    ax.text(1.55, 1.11, "Σ", fontsize=12.0, fontweight="bold", ha="center", va="center", color="#FFFFFF", zorder=4)
    ax.text(2.05, 1.24, "Calibrated Ensemble Inference", fontsize=9.8, fontweight="bold", ha="left", color="#3730A3", zorder=3)
    ax.text(2.05, 0.98, "Temperature Scaling • Anchored Logit Fusion", fontsize=8.0, fontstyle="italic", ha="left", color="#4338CA", zorder=3)
    # Output arrow & badge
    ax.annotate("", xy=(5.35, 1.11), xytext=(4.75, 1.11),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#4338CA"), zorder=3)
    pred_badge = FancyBboxPatch((5.45, 0.86), 1.25, 0.50,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor="#4338CA", edgecolor="none", zorder=3)
    ax.add_patch(pred_badge)
    ax.text(6.075, 1.11, "Prediction", fontsize=8.2, fontweight="bold", ha="center", va="center", color="#FFFFFF", zorder=4)


    # =========================================================================
    # SECTION 2 (RIGHT): MULTI-TIER FEDERATION TOPOLOGY (Width: 8.0 units)
    # =========================================================================
    sec2_box = FancyBboxPatch((7.6, 0.45), 8.0, 8.60,
                              boxstyle="round,pad=0.04,rounding_size=0.12",
                              facecolor="#FAFAFA", edgecolor="#64748B", linewidth=1.5, zorder=1)
    ax.add_patch(sec2_box)

    ax.text(11.6, 8.78, "SECTION 2: MULTI-TIER FEDERATION TOPOLOGY",
            fontsize=11.5, fontweight="bold", ha="center", va="center", color="#0F172A", zorder=2)
    ax.text(11.6, 8.52, "Hierarchical Consensus • Layered Defense • Privacy-Preserving Clustering",
            fontsize=8.5, fontstyle="italic", ha="center", va="center", color="#475569", zorder=2)

    # 2.1 Central Federated Server Container
    srv_box = FancyBboxPatch((7.9, 6.35), 7.4, 1.95,
                             boxstyle="round,pad=0.03,rounding_size=0.08",
                             facecolor="#FFFFFF", edgecolor="#3B82F6", linewidth=1.4, zorder=2)
    ax.add_patch(srv_box)
    ax.text(11.6, 8.05, "Central Federated Server", fontsize=10.5, fontweight="bold", ha="center", color="#1E3A8A", zorder=3)

    # Sub-card 1: Layered Byzantine Defense (Subspace SCCF + Norm Bounding + TTT)
    byz_sub = FancyBboxPatch((8.12, 6.52), 3.42, 1.25,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             facecolor="#FEF2F2", edgecolor="#EF4444", linewidth=1.1, zorder=3)
    ax.add_patch(byz_sub)
    draw_shield_badge(ax, 8.52, 7.15, w=0.48, h=0.56, color="#DC2626", zorder=5)
    ax.text(8.92, 7.46, "Layered Byzantine Defense", fontsize=8.6, fontweight="bold", ha="left", color="#991B1B", zorder=4)
    ax.text(8.92, 7.24, "• Subspace Filtering (SCCF)", fontsize=7.4, ha="left", color="#B91C1C", zorder=4)
    ax.text(8.92, 7.04, "• Adaptive Norm-Bounding", fontsize=7.4, ha="left", color="#B91C1C", zorder=4)
    ax.text(8.92, 6.84, "• Temporal Trust Tracking", fontsize=7.4, ha="left", color="#B91C1C", zorder=4)

    # Arrow between Defense and Aggregator
    ax.annotate("", xy=(11.75, 7.15), xytext=(11.54, 7.15),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#DC2626"), zorder=5)

    # Sub-card 2: Consensus Aggregator
    agg_sub = FancyBboxPatch((11.75, 6.52), 3.35, 1.25,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.1, zorder=3)
    ax.add_patch(agg_sub)
    draw_server_chassis(ax, 12.18, 7.15, w=0.58, h=0.54, zorder=5)
    ax.text(12.60, 7.46, "Consensus Aggregator", fontsize=8.6, fontweight="bold", ha="left", color="#1E3A8A", zorder=4)
    ax.text(12.60, 7.24, "• Trust-Weighted Averaging", fontsize=7.4, ha="left", color="#2563EB", zorder=4)
    ax.text(12.60, 7.04, "• Global Backbone Sync", fontsize=7.4, ha="left", color="#2563EB", zorder=4)
    ax.text(12.60, 6.84, "• Global Root Head Sync", fontsize=7.4, ha="left", color="#2563EB", zorder=4)

    # 2.2 Synergistic Peer Clustering Layer (Middle of Section 2)
    peer_box = FancyBboxPatch((7.9, 3.75), 7.4, 2.15,
                              boxstyle="round,pad=0.03,rounding_size=0.08",
                              facecolor="#FAF5FF", edgecolor="#A855F7", linewidth=1.4, linestyle="--", zorder=2)
    ax.add_patch(peer_box)
    ax.text(11.6, 5.66, "Collaborative Peer Clusters", fontsize=10.2, fontweight="bold", ha="center", color="#5B21B6", zorder=3)
    ax.text(11.6, 5.42, "Privacy Sketches (Random Projections) • Synchronizes Parent Heads",
            fontsize=7.8, fontstyle="italic", ha="center", color="#6D28D9", zorder=3)

    # 3 Cluster Cohort Cards
    c_configs = [
        ("Peer Cluster 1", 8.1, 2.25),
        ("Peer Cluster 2", 10.45, 2.25),
        ("Peer Cluster K", 12.8, 2.25)
    ]
    for cname, cx, cw in c_configs:
        c_patch = FancyBboxPatch((cx, 3.92), cw, 1.30,
                                 boxstyle="round,pad=0.02,rounding_size=0.04",
                                 facecolor="#FFFFFF", edgecolor="#7C3AED", linewidth=1.2, zorder=3)
        ax.add_patch(c_patch)
        
        # Triangle peer cluster graph (3 interconnected nodes)
        mid_x = cx + cw/2
        r_clust = 0.30
        p_top = (mid_x, 4.88)
        p_bl = (mid_x - r_clust, 4.48)
        p_br = (mid_x + r_clust, 4.48)
        ax.plot([p_top[0], p_bl[0], p_br[0], p_top[0]], [p_top[1], p_bl[1], p_br[1], p_top[1]], color="#C4B5FD", lw=1.3, zorder=3)
        ax.add_patch(Circle(p_top, 0.085, facecolor="#7C3AED", edgecolor="#FFFFFF", lw=1.0, zorder=4))
        ax.add_patch(Circle(p_bl, 0.085, facecolor="#8B5CF6", edgecolor="#FFFFFF", lw=1.0, zorder=4))
        ax.add_patch(Circle(p_br, 0.085, facecolor="#8B5CF6", edgecolor="#FFFFFF", lw=1.0, zorder=4))
        
        ax.text(mid_x, 4.22, cname, fontsize=8.6, fontweight="bold", ha="center", color="#5B21B6", zorder=4)
        ax.text(mid_x, 4.04, "Parent Head Sync", fontsize=7.4, fontstyle="italic", ha="center", color="#7C3AED", zorder=4)

    # 2.3 Heterogeneous Edge Clients (Bottom of Section 2)
    client_box = FancyBboxPatch((7.9, 1.65), 7.4, 1.70,
                                boxstyle="round,pad=0.03,rounding_size=0.08",
                                facecolor="#F0FDF4", edgecolor="#16A34A", linewidth=1.3, zorder=2)
    ax.add_patch(client_box)
    ax.text(8.15, 3.12, "Heterogeneous Edge Client Fleet", fontsize=9.6, fontweight="bold", ha="left", color="#14532D", zorder=3)

    # 3 Diverse Device Cards with Projection Sketch Indicator
    dev_configs = [
        ("Mobile Client", "phone", 8.1, 2.25),
        ("Edge Server", "server", 10.45, 2.25),
        ("IoT Device", "iot", 12.8, 2.25)
    ]
    for dtitle, dev_kind, dx, dw in dev_configs:
        d_patch = FancyBboxPatch((dx, 1.78), dw, 1.18,
                                 boxstyle="round,pad=0.02,rounding_size=0.04",
                                 facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.1, zorder=3)
        ax.add_patch(d_patch)
        
        # 1. Top Title (cleanly above icon)
        ax.text(dx + dw/2, 2.74, dtitle, fontsize=8.4, fontweight="bold", ha="center", color="#065F46", zorder=4)
        
        # 2. Device silhouette + Padlock centered in the lower area (cy = 2.20)
        draw_device_icon(ax, dx + dw*0.32, 2.20, dev_type=dev_kind, color="#059669", zorder=4)
        draw_padlock(ax, dx + dw*0.74, 2.20, size=0.22, color="#059669", zorder=5)

    # 2.4 Communication Legend Bar (Very Bottom of Section 2)
    leg_box = FancyBboxPatch((7.9, 0.65), 7.4, 0.78,
                             boxstyle="round,pad=0.02,rounding_size=0.04",
                             facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0, zorder=2)
    ax.add_patch(leg_box)
    # Legend items
    # Item 1: Blue line (Global Sync)
    ax.plot([8.15, 8.55], [1.16, 1.16], color="#2563EB", lw=2.0, zorder=3)
    ax.text(8.65, 1.16, "Global Consensus Sync (Root & Backbone)", fontsize=7.4, va="center", color="#1E293B", zorder=3)
    # Item 2: Purple dashed line (Cluster Sync)
    ax.plot([8.15, 8.55], [0.88, 0.88], color="#7C3AED", lw=1.8, linestyle="--", zorder=3)
    ax.text(8.65, 0.88, "Cluster Peer Sync (Parent Head & Sketches)", fontsize=7.4, va="center", color="#1E293B", zorder=3)
    # Item 3: Green lock badge (Private Isolation)
    draw_padlock(ax, 13.0, 1.02, size=0.20, color="#059669", zorder=5)
    ax.text(13.25, 1.02, "Strict On-Device Isolation", fontsize=7.4, fontweight="bold", va="center", color="#047857", zorder=3)

    # 2.5 Clean Vertical Communication Connectors between Right Section Tiers (Zero Collisions)
    # Client <-> Cluster connectors (Purple dashed)
    ax.annotate("", xy=(9.22, 3.75), xytext=(9.22, 3.35),
                arrowprops=dict(arrowstyle="<->", lw=1.5, linestyle="--", color="#7C3AED"), zorder=5)
    ax.annotate("", xy=(11.57, 3.75), xytext=(11.57, 3.35),
                arrowprops=dict(arrowstyle="<->", lw=1.5, linestyle="--", color="#7C3AED"), zorder=5)
    ax.annotate("", xy=(13.92, 3.75), xytext=(13.92, 3.35),
                arrowprops=dict(arrowstyle="<->", lw=1.5, linestyle="--", color="#7C3AED"), zorder=5)

    # Cluster <-> Server connectors
    # Left: Defended updates entering defense module (Red arrow)
    ax.annotate("", xy=(9.80, 6.35), xytext=(9.80, 5.90),
                arrowprops=dict(arrowstyle="->", lw=1.5, linestyle="-", color="#DC2626"), zorder=5)
    ax.text(9.20, 6.12, "Defended Updates", fontsize=7.2, fontweight="bold", color="#DC2626", ha="right", va="center", zorder=5)

    # Right: Consensus broadcast to clusters (Blue arrow)
    ax.annotate("", xy=(13.50, 5.90), xytext=(13.50, 6.35),
                arrowprops=dict(arrowstyle="->", lw=1.5, linestyle="-", color="#2563EB"), zorder=5)
    ax.text(14.05, 6.12, "Consensus Broadcast", fontsize=7.2, fontweight="bold", color="#2563EB", ha="left", va="center", zorder=5)

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
        print(f"[ok] Saved symbolic FedHEP architecture diagram to {p}")

    plt.close(fig)

if __name__ == "__main__":
    build_symbolic_fedhep_diagram()
