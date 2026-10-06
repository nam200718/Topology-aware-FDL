#!/usr/bin/env python3
"""
Scientific Publication Plotter for FedHEP Manuscript Figures.
Adheres strictly to the Scientific Publication Plotting Standard:
- Python -> plotnine only.
- Single top-of-file parameter block defining all tunables, dimensions, and color mappings.
- Bang Wong's colorblind-safe discrete palette in exact ordered sequence.
- Mandatory viridis continuous color settings block.
- Standardized typography: Times serif family, ONE font size (text_size_pt) for all elements.
- No plot title (ggtitle/labs(title=...) blank).
- Clean white panel, light major grid, transparent figure canvas.
- Vector PDF output (85 mm single-column and 180 mm double-column) + high-res raster preview.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Times', 'serif']
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Ensure plotnine is available
try:
    from plotnine import (
        ggplot, aes, geom_line, geom_point, geom_errorbar,
        facet_wrap, scale_color_manual, scale_shape_manual,
        labs, theme_bw, theme, element_text, element_blank,
        element_line, element_rect, ggsave, scale_x_discrete,
        guide_legend, guides
    )
except ImportError as e:
    print(f"Error: plotnine is required: {e}")
    sys.exit(1)

# ==============================================================================
# PARAMETER BLOCK (TUNABLE SETTINGS)
# ==============================================================================
# Typography: One serif family and one font size for ALL text elements
font_family = "serif"      # Times / serif family via matplotlib font configuration
text_size_pt = 8.0         # Exactly one font size for ALL text (axes, ticks, legend, strips)

# Stroke and marker sizing
line_width_data = 0.75     # Stroke width for series lines (pt)
point_size_data = 2.2      # Marker size for data points (pt)
errorbar_width = 0.25      # Width of errorbar end caps
axis_line_width = 0.4      # Axis line and tick stroke width

# Figure dimensions (mm)
width_double_mm = 180.0    # Double-column width (180 mm)
width_single_mm = 85.0     # Single-column width (85 mm)
height_double_mm = 72.0    # Flat aspect ratio for multi-panel figure
height_single_mm = 65.0    # Flat aspect ratio for single-column figure

# Bang Wong's colorblind-safe discrete palette (exact ordered sequence)
# 1: orange, 2: sky blue, 3: blue green, 4: pale violet, 5: vermillion, 6: yellow, 7: blue, 8: black
wong_palette = [
    "#E69F00",  # 1. Orange
    "#56B4E9",  # 2. Sky Blue
    "#009E73",  # 3. Blue Green
    "#CC79A7",  # 4. Pale Violet
    "#D55E00",  # 5. Vermillion
    "#F0E442",  # 6. Yellow
    "#0072B2",  # 7. Blue
    "#000000",  # 8. Black
]

# Continuous color settings (mandatory in parameter block per standard)
continuous_colormap = "viridis"
# Rule: continuous color maps follow equal-count quantile breaks derived from data
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def build_cifar100_manuscript_figure():
    """Generates the flagship 180 mm double-column CIFAR-100 benchmark figure."""
    data_path = os.path.join(OUTPUTS_DIR, "section_5_2_cifar100_personalization", "data_cifar100.json")
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Regimes in mathematical ordering from IID to acute non-IID skew
    regime_order = ["iid", "mild", "moderate", "severe", "extreme"]
    regime_labels = {
        "iid": "IID (α=∞)",
        "mild": "Mild (α=1.0)",
        "moderate": "Mod (α=0.5)",
        "severe": "Sev (α=0.1)",
        "extreme": "Ext (α=0.05)",
    }

    # Stable method ordering (7 series <= 8 Wong colors)
    method_order = [
        "FedAvg",
        "FedProx",
        "Multi-Krum",
        "SCAFFOLD",
        "Ditto",
        "FedRep",
        "FedHEP",
    ]
    method_labels = {
        "FedAvg": "FedAvg",
        "FedProx": "FedProx",
        "Multi-Krum": "Multi-Krum",
        "SCAFFOLD": "SCAFFOLD",
        "Ditto": "Ditto",
        "FedRep": "FedRep",
        "FedHEP": "FedHEP (Ours)",
    }

    # Distinct shapes for series accessibility
    shape_list = ["o", "s", "^", "D", "v", "P", "X"]

    rows = []
    for r in records:
        reg = r.get("regime")
        m = r.get("method")
        if reg in regime_order and m in method_order:
            # Metric 1: Mean Personalized Accuracy
            rows.append({
                "Regime": regime_labels[reg],
                "Regime_idx": regime_order.index(reg),
                "Method": method_labels[m],
                "Metric": "Mean Personalized Accuracy (%)",
                "Value": float(r.get("mean_acc", 0.0)),
                "Std": float(r.get("std_acc", 0.0)),
            })
            # Metric 2: Worst-Decile Tail Fairness (Bottom 10%)
            rows.append({
                "Regime": regime_labels[reg],
                "Regime_idx": regime_order.index(reg),
                "Method": method_labels[m],
                "Metric": "Worst-Decile Tail Fairness (Bottom 10%, %)",
                "Value": float(r.get("mean_b10", 0.0)),
                "Std": float(r.get("std_b10", 0.0)),
            })

    df = pd.DataFrame(rows)
    df["Regime"] = pd.Categorical(df["Regime"], categories=[regime_labels[k] for k in regime_order], ordered=True)
    df["Method"] = pd.Categorical(df["Method"], categories=[method_labels[k] for k in method_order], ordered=True)
    df["Metric"] = pd.Categorical(
        df["Metric"],
        categories=["Mean Personalized Accuracy (%)", "Worst-Decile Tail Fairness (Bottom 10%, %)"],
        ordered=True
    )

    palette_cifar = wong_palette[:len(method_order)]

    # Standardized theme wiring
    p = (
        ggplot(df, aes(x="Regime", y="Value", color="Method", group="Method", shape="Method"))
        + geom_errorbar(
            aes(ymin="Value - Std", ymax="Value + Std"),
            width=errorbar_width,
            size=axis_line_width,
            alpha=0.6,
            show_legend=False
        )
        + geom_line(size=line_width_data, alpha=0.9)
        + geom_point(size=point_size_data, alpha=0.95)
        + facet_wrap("~Metric", ncol=2, scales="free_y")
        + scale_color_manual(values=palette_cifar)
        + scale_shape_manual(values=shape_list[:len(method_order)])
        + labs(x="Statistical Heterogeneity Regime", y="Accuracy / Fairness (%)")
        + theme_bw()
        + theme(
            text=element_text(family=font_family, size=text_size_pt),
            axis_title=element_text(family=font_family, size=text_size_pt),
            axis_text=element_text(family=font_family, size=text_size_pt),
            axis_text_x=element_text(family=font_family, size=text_size_pt, angle=15, ha="right"),
            strip_text=element_text(family=font_family, size=text_size_pt),
            strip_background=element_rect(fill="#F2F2F2", color="#CCCCCC", size=axis_line_width),
            legend_title=element_text(family=font_family, size=text_size_pt),
            legend_text=element_text(family=font_family, size=text_size_pt),
            legend_position="bottom",
            legend_box="horizontal",
            legend_background=element_blank(),
            panel_grid_major=element_line(color="#EBEBEB", size=0.3),
            panel_grid_minor=element_blank(),
            panel_border=element_rect(color="#555555", size=axis_line_width, fill=None),
            plot_background=element_blank(),
            plot_title=element_blank(),
            plot_subtitle=element_blank(),
        )
        + guides(
            color=guide_legend(nrow=1, title="Method:"),
            shape=guide_legend(nrow=1, title="Method:")
        )
    )

    # Export vector PDF and raster preview
    pdf_path = os.path.join(FIGURES_DIR, "fig_cifar100_manuscript.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig_cifar100_manuscript.png")
    
    # Save dimensions in inches
    w_in = width_double_mm / 25.4
    h_in = height_double_mm / 25.4

    p.save(pdf_path, width=w_in, height=h_in, units="in", dpi=300)
    p.save(png_path, width=w_in, height=h_in, units="in", dpi=300)
    print(f"Generated Vector PDF: {pdf_path}")
    print(f"Generated Raster PNG: {png_path}")


def build_ablation_manuscript_figure():
    """Generates the targeted 85 mm single-column Ablation Study & Component Valuation figure."""
    data_path = os.path.join(OUTPUTS_DIR, "baselines", "ablation", "results_ablation.json")
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        ablation_records = json.load(f)

    # Master full FedHEP baseline
    cifar_path = os.path.join(OUTPUTS_DIR, "section_5_2_cifar100_personalization", "data_cifar100.json")
    fedhep_records = []
    if os.path.exists(cifar_path):
        with open(cifar_path, "r", encoding="utf-8") as f:
            cifar_data = json.load(f)
            fedhep_records = [x for x in cifar_data if x.get("method") == "FedHEP"]

    regime_order = ["iid", "mild", "moderate", "severe", "extreme"]
    regime_labels = {
        "iid": "IID",
        "mild": "α=1.0",
        "moderate": "α=0.5",
        "severe": "α=0.1",
        "extreme": "α=0.05",
    }

    height_single_mm = 78.0    # Single-column height with bottom legend

    config_order = [
        "Full Defended FedHEP",
        "w/o ACLM",
        "w/o Parent Head",
        "K=1 Grand Coalition",
        "K=3 Oracle Bound",
    ]
    config_labels = {
        "Full Defended FedHEP": "FedHEP (Full)",
        "w/o ACLM": "w/o ACLM",
        "w/o Parent Head": "w/o Parent Head",
        "K=1 Grand Coalition": "K=1 Coalition",
        "K=3 Oracle Bound": "K=3 Oracle Bound",
    }

    rows = []
    for r in fedhep_records:
        reg = r.get("regime")
        if reg in regime_order:
            rows.append({
                "Regime": regime_labels[reg],
                "Config": config_labels["Full Defended FedHEP"],
                "Accuracy": float(r.get("mean_acc", 0.0)),
                "Std": float(r.get("std_acc", 0.0)),
            })

    method_to_config = {
        "topo_no_aclm": "w/o ACLM",
        "topo_no_parent": "w/o Parent Head",
        "topo_k1": "K=1 Grand Coalition",
        "topo_oracle_k3": "K=3 Oracle Bound",
    }

    for r in ablation_records:
        m = r.get("method")
        reg = r.get("regime")
        if m in method_to_config and reg in regime_order:
            cfg = method_to_config[m]
            rows.append({
                "Regime": regime_labels[reg],
                "Config": config_labels[cfg],
                "Accuracy": float(r.get("mean_acc", 0.0)),
                "Std": float(r.get("std_acc", 0.0)),
            })

    df = pd.DataFrame(rows)
    df["Regime"] = pd.Categorical(df["Regime"], categories=[regime_labels[k] for k in regime_order], ordered=True)
    df["Config"] = pd.Categorical(df["Config"], categories=[config_labels[k] for k in config_order], ordered=True)

    palette_ablation = wong_palette[:len(config_order)]
    shapes_ablation = ["o", "s", "^", "v", "D"]

    p = (
        ggplot(df, aes(x="Regime", y="Accuracy", color="Config", group="Config", shape="Config"))
        + geom_errorbar(
            aes(ymin="Accuracy - Std", ymax="Accuracy + Std"),
            width=errorbar_width,
            size=axis_line_width,
            alpha=0.6,
            show_legend=False
        )
        + geom_line(size=line_width_data, alpha=0.9)
        + geom_point(size=point_size_data, alpha=0.95)
        + scale_color_manual(values=palette_ablation)
        + scale_shape_manual(values=shapes_ablation)
        + labs(x="Dirichlet Skew Regime (α)", y="Personalized Accuracy (%)")
        + theme_bw()
        + theme(
            text=element_text(family=font_family, size=text_size_pt),
            axis_title=element_text(family=font_family, size=text_size_pt),
            axis_text=element_text(family=font_family, size=text_size_pt),
            axis_text_x=element_text(family=font_family, size=text_size_pt),
            legend_title=element_blank(),
            legend_text=element_text(family=font_family, size=text_size_pt),
            legend_position="bottom",
            legend_box="horizontal",
            legend_background=element_blank(),
            panel_grid_major=element_line(color="#EBEBEB", size=0.3),
            panel_grid_minor=element_blank(),
            panel_border=element_rect(color="#555555", size=axis_line_width, fill=None),
            plot_background=element_blank(),
            plot_title=element_blank(),
            plot_subtitle=element_blank(),
        )
        + guides(
            color=guide_legend(ncol=2, title=""),
            shape=guide_legend(ncol=2, title="")
        )
    )

    pdf_path = os.path.join(FIGURES_DIR, "fig_ablation_manuscript.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig_ablation_manuscript.png")
    
    w_in = width_single_mm / 25.4
    h_in = height_single_mm / 25.4

    p.save(pdf_path, width=w_in, height=h_in, units="in", dpi=300)
    p.save(png_path, width=w_in, height=h_in, units="in", dpi=300)
    print(f"Generated Vector PDF: {pdf_path}")
    print(f"Generated Raster PNG: {png_path}")


if __name__ == "__main__":
    print("Generating manuscript-quality figures conforming to Scientific Publication Plotting Standard...")
    build_cifar100_manuscript_figure()
    build_ablation_manuscript_figure()
    print("All manuscript figures successfully exported!")
