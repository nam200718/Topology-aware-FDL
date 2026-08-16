import os
import json
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Professional executive color palette
COLOR_MAP = {
    "Star (FedAvg)": "#6b7280",                         # Slate Gray
    "Star (FedAvg) (Pers.)": "#6b7280",
    "Star (Ditto)": "#d97706",                          # Warm Amber
    "Star (Ditto) (Pers.)": "#d97706",
    "Star (APFL - Shared Backbone)": "#0d9488",          # Teal
    "Star (APFL - Shared Backbone) (Pers.)": "#0d9488",
    "Hierarchical Ensemble (Adaptive Update-Sim)": "#4f46e5",  # Royal Indigo (Proposed)
    "Hierarchical Ensemble (HEP)": "#4f46e5",
    "Hierarchical Ensemble Personalization": "#4f46e5",
    "HEP": "#4f46e5",
    "Hierarchical Ensemble (T-HEP)": "#4f46e5",
    "Hierarchical Ensemble": "#4f46e5",
    "T-HEP": "#4f46e5",
    "Ensemble": "#4f46e5",
}

DEFAULT_PALETTE = sns.color_palette("tab10")


def _get_color(label: str, idx: int = 0):
    for key, color in COLOR_MAP.items():
        if key in label:
            return color
    return DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]


def plot_experiment_results(metrics_path: str, output_dir: Optional[str] = None):
    """Plots clean accuracy and loss convergence from a single experiment run."""
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    if metrics_path.endswith('.json'):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(metrics_path)

    if output_dir is None:
        output_dir = os.path.dirname(metrics_path)

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # 1. Accuracy Plot
    plt.figure(figsize=(9, 5), dpi=300)
    rounds = np.asarray(df['round'])
    plt.plot(rounds, np.asarray(df['test_accuracy']), label='Global Model', color='#3b82f6', linewidth=2.2, marker='o', markevery=5)

    if 'ensemble_test_accuracy' in df.columns:
        plt.plot(rounds, np.asarray(df['ensemble_test_accuracy']), label='Personalized Ensemble', color='#10b981', linestyle='--', linewidth=2.2, marker='s', markevery=5)

    plt.title('Model Accuracy Convergence', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Round', fontsize=11, labelpad=8)
    plt.ylabel('Accuracy (%)', fontsize=11, labelpad=8)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_convergence.png'), bbox_inches='tight')
    plt.close()

    # 2. Loss Plot
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(rounds, np.asarray(df['test_loss']), label='Global Model Loss', color='#ef4444', linewidth=2.2, marker='o', markevery=5)

    if 'ensemble_test_loss' in df.columns:
        plt.plot(rounds, np.asarray(df['ensemble_test_loss']), label='Personalized Ensemble Loss', color='#f59e0b', linestyle='--', linewidth=2.2, marker='s', markevery=5)

    plt.title('Model Loss Convergence', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Round', fontsize=11, labelpad=8)
    plt.ylabel('Loss', fontsize=11, labelpad=8)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_convergence.png'), bbox_inches='tight')
    plt.close()


def plot_comparison_convergence(experiment_dirs, labels, output_path: str):
    """
    Plots a clean 2-panel comparative convergence analysis without marker clutter.
    Left Panel: Global Model Consensus Accuracy
    Right Panel: Personalized Edge Model Accuracy
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, dpi=300)

    has_personalized = False

    for idx, (exp_dir, label) in enumerate(zip(experiment_dirs, labels)):
        json_path = os.path.join(exp_dir, 'metrics.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)

            rounds = np.asarray(df['round'])
            acc_global = np.asarray(df['test_accuracy'])
            color = _get_color(label, idx)

            is_proposed = "Adaptive" in label or "Hierarchical" in label
            ls = '-' if is_proposed else '-'
            lw = 2.5 if is_proposed else 1.8
            alpha = 1.0 if is_proposed else 0.8

            ax1.plot(rounds, acc_global, label=label, color=color, linestyle=ls, linewidth=lw, alpha=alpha, marker='o', markevery=5, markersize=4)

            if 'ensemble_test_accuracy' in df.columns:
                has_personalized = True
                acc_pers = np.asarray(df['ensemble_test_accuracy'])
                ax2.plot(rounds, acc_pers, label=f'{label}', color=color, linestyle=ls, linewidth=lw, alpha=alpha, marker='s', markevery=5, markersize=4)

    ax1.set_title("Global Model Accuracy", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Training Round", fontsize=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='lower right', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.95)

    ax2.set_title("Personalized Model Accuracy", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Training Round", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.4)

    if has_personalized:
        ax2.legend(loc='lower right', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.95)
    else:
        ax2.text(0.5, 0.5, "No Personalized Models Logged", ha='center', va='center', transform=ax2.transAxes, fontsize=10, color='gray')

    plt.suptitle("Comparative Convergence Analysis across Topologies & Algorithms", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_robustness_summary(df_summary: pd.DataFrame, output_path: str, title: str = "Robustness Comparison", y_col: Optional[str] = None):
    """
    Plots a clean, uncluttered 2-panel bar comparison chart.
    Left Panel: Global Accuracy across Heterogeneity Scenarios
    Right Panel: Personalized Accuracy across Heterogeneity Scenarios
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    if y_col is None:
        y_col = "Last5 Avg Accuracy" if "Last5 Avg Accuracy" in df_summary.columns else "Final Accuracy"

    df_copy = df_summary.copy()
    
    # Split into Global vs Personalized metrics
    df_global = df_copy[df_copy["Metric"] == "Global"]
    df_pers = df_copy[df_copy["Metric"] == "Personalized"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True, dpi=300)

    # Clean Topology palette
    topos = df_copy["Topology"].unique()
    palette = {t: _get_color(t, i) for i, t in enumerate(topos)}

    # Panel 1: Global
    sns.barplot(data=df_global, x="Scenario", y=y_col, hue="Topology", ax=ax1, palette=palette, edgecolor="none", alpha=0.9)
    ax1.set_title("Global Model Accuracy (Consensus)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Heterogeneity Scenario", fontsize=10)
    ax1.set_ylabel(f"{y_col} (%)", fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax1.tick_params(axis='x', rotation=25, labelsize=8.5)
    ax1.legend(loc='upper right', fontsize=8, frameon=True, facecolor='white', framealpha=0.95)

    # Annotate top bar in extreme non-iid
    for p in ax1.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 70:
            ax1.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=7.5, fontweight='bold', xytext=(0, 2), textcoords='offset points')

    # Panel 2: Personalized
    if not df_pers.empty:
        sns.barplot(data=df_pers, x="Scenario", y=y_col, hue="Topology", ax=ax2, palette=palette, edgecolor="none", alpha=0.9)
        ax2.set_title("Personalized Edge Model Accuracy", fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel("Heterogeneity Scenario", fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
        ax2.tick_params(axis='x', rotation=25, labelsize=8.5)
        ax2.legend(loc='upper right', fontsize=8, frameon=True, facecolor='white', framealpha=0.95)

        for p in ax2.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 70:
                ax2.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=7.5, fontweight='bold', xytext=(0, 2), textcoords='offset points')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_robustness_heatmap(df_summary: pd.DataFrame, output_path: str, title: str = "Accuracy Heatmap across Heterogeneity Scenarios"):
    """Plots an executive, clean 2D heatmap matrix of Last-5 Avg Accuracy."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="white", font_scale=1.0)

    y_col = "Last5 Avg Accuracy" if "Last5 Avg Accuracy" in df_summary.columns else "Final Accuracy"
    pivot_df = df_summary.pivot(index="Topology", columns="Scenario", values=y_col)

    plt.figure(figsize=(9.5, 5), dpi=300)
    ax = sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': f'{y_col} (%)'},
                     linewidths=1.5, linecolor='white', annot_kws={"size": 10, "weight": "bold"})

    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Topology / Personalization Method", fontsize=10, labelpad=8)
    plt.xlabel("Heterogeneity Scenario", fontsize=10, labelpad=8)
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_byzantine_matrix(df_matrix: pd.DataFrame, output_path: str, title: str = "Byzantine Robustness Matrix"):
    """Plots clean final/last-5 accuracy vs Byzantine rate line plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    plt.figure(figsize=(9, 5.5), dpi=300)
    topos = df_matrix["Topology"].unique()
    palette = {t: _get_color(t, i) for i, t in enumerate(topos)}

    y_col = "Last5 Avg Accuracy" if "Last5 Avg Accuracy" in df_matrix.columns else "Final Accuracy"
    
    sns.lineplot(
        data=df_matrix, 
        x="Byzantine Rate", 
        y=y_col, 
        hue="Topology", 
        palette=palette, 
        linewidth=2.5, 
        marker="o", 
        markersize=7
    )
    
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.ylabel(f"Test Accuracy (%) [{y_col}]", fontsize=11, labelpad=8)
    plt.xlabel("Byzantine Attacker Fraction", fontsize=11, labelpad=8)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.4)

    # Format X-axis ticks as percentages
    rates = sorted(df_matrix["Byzantine Rate"].unique())
    plt.xticks(rates, [f"{int(r * 100)}%" for r in rates], fontsize=10)
    plt.yticks(fontsize=10)

    plt.legend(
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        frameon=True, 
        facecolor='white', 
        framealpha=0.95,
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
