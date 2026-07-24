import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def plot_experiment_results(metrics_path: str, output_dir: str = None):
    """Plots accuracy and loss convergence from a single experiment run metrics file."""
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
    sns.set_style("whitegrid")

    # 1. Accuracy Plot
    plt.figure(figsize=(11, 6))
    plt.plot(np.asarray(df['round']), np.asarray(df['test_accuracy']), label='Global Model Accuracy', marker='o')

    if 'ensemble_test_accuracy' in df.columns:
        plt.plot(np.asarray(df['round']), np.asarray(df['ensemble_test_accuracy']), label='Ensemble Accuracy', marker='s', linestyle='--')

    plt.title('Model Accuracy Convergence')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_convergence.png'), bbox_inches='tight')
    plt.close()

    # 2. Loss Plot
    plt.figure(figsize=(11, 6))
    plt.plot(np.asarray(df['round']), np.asarray(df['test_loss']), label='Global Model Loss', marker='o', color='red')

    if 'ensemble_test_loss' in df.columns:
        plt.plot(np.asarray(df['round']), np.asarray(df['ensemble_test_loss']), label='Ensemble Loss', marker='s', linestyle='--', color='orange')

    plt.title('Model Loss Convergence')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_convergence.png'), bbox_inches='tight')
    plt.close()


def plot_comparison_convergence(experiment_dirs, labels, output_path: str):
    """Plots comparative convergence curves for multiple experiments in a scenario."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))

    for exp_dir, label in zip(experiment_dirs, labels):
        json_path = os.path.join(exp_dir, 'metrics.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)

            plt.plot(np.asarray(df['round']), np.asarray(df['test_accuracy']), label=f'{label} (Global)')

            if 'ensemble_test_accuracy' in df.columns:
                plt.plot(np.asarray(df['round']), np.asarray(df['ensemble_test_accuracy']), label=f'{label} (Ensemble)', linestyle='--')

    plt.title('Experiment Comparison: Test Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_robustness_summary(df_summary: pd.DataFrame, output_path: str, title: str = "Robustness Comparison"):
    """Plots a bar chart comparing final accuracy across topologies and scenarios."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_summary, x="Topology", y="Final Accuracy", hue="Scenario")
    plt.title(title)
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(0, 110)

    for p in ax.patches:
        if isinstance(p, mpatches.Rectangle):
            height = p.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.1f}%',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points',
                            fontsize=8)

    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Scenario", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_byzantine_matrix(df_matrix: pd.DataFrame, output_path: str, title: str = "Byzantine Robustness Matrix"):
    """Plots final accuracy vs Byzantine rate line plot across topologies."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    sns.lineplot(data=df_matrix, x="Byzantine Rate", y="Final Accuracy", hue="Topology", marker="o")
    plt.title(title)
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Byzantine Rate (Proportion of Malicious Clients)")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
