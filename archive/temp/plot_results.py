import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Formal paper style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'serif',
})

ARTIFACT_DIR = r"d:\UROP\Topology-aware-FDL\outputs\postmidterm_visualization_report_v1\new"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def load_csv(filepath):
    return pd.read_csv(filepath)

def plot_partA():
    # Plots E2 and E3
    experiments = [
        ("outputs/postmidterm_final/outputs_partA_v2/exp2_random_with_HE", "partA_E2_random_accuracy.png", "E2: Convergence on Random Partition"),
        ("outputs/postmidterm_final/outputs_partA_v2/exp3_hierarchical_all", "partA_E3_hierarchical_accuracy.png", "E3: Convergence on Hierarchical Partition")
    ]
    
    for folder, out_name, title in experiments:
        plt.figure(figsize=(10, 6))
        
        # Load topologies
        files = {
            "Star (FedAvg)": ("Star_FedAvg_metrics.csv", "test_accuracy"),
            "Star (Ditto)": ("Star_Ditto_metrics.csv", "ensemble_test_accuracy"),
            "Ring": ("Ring_metrics.csv", "test_accuracy"),
            "Gossip": ("Gossip_k3_metrics.csv", "test_accuracy"),
            "HE (Global)": ("HE_Agg-Only_metrics.csv", "test_accuracy"),
            "HE (Ensemble)": ("HE_Ensemble_metrics.csv", "ensemble_test_accuracy")
        }
        
        for label, (fname, metric) in files.items():
            fpath = os.path.join(folder, fname)
            if os.path.exists(fpath):
                data = load_csv(fpath)
                rounds = data['round'].tolist()
                
                # Check if metric exists
                if metric in data.columns:
                    acc = data[metric].fillna(0).tolist()
                else:
                    acc = [0] * len(rounds)
                    
                # Make HE lines stand out
                linewidth = 3 if "HE" in label else 2
                plt.plot(rounds, acc, label=label, linewidth=linewidth, marker='o', markersize=4, markevery=5)
                
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Communication Round", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.legend(title="Topology", fontsize=10)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, out_name), dpi=300)
        plt.close()

def plot_partB():
    # Plots E4 and E5
    experiments = [
        ("outputs/postmidterm_final/output_partB_E4_v2", "partB_E4_robustness.png", "E4: Robustness on Random Partition (Label Flip)"),
        ("outputs/postmidterm_final/output_partB_E5_label_flip", "partB_E5_robustness.png", "E5: Robustness on Hierarchical Partition (Label Flip)")
    ]
    
    rates = ["No_Attack", "0_1", "0_2", "0_3"]
    x_rates = [0.0, 0.1, 0.2, 0.3]
    
    for folder, out_name, title in experiments:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        topos = {
            "Star (FedAvg)": ("Star", "test_accuracy"),
            "Star (Ditto)": ("Star_Ditto,", "ensemble_test_accuracy"),
            "Ring": ("Ring", "test_accuracy"),
            "HE": ("HE", "ensemble_test_accuracy")
        }
        
        for ax, defense_mode, sub_title in zip(axes, ["No_Defense", "Full_Defense"], ["Without Defense", "With Soft Cosine Defense"]):
            ax.set_title(sub_title, fontsize=14)
            ax.set_xlabel("Byzantine Attack Rate (Label Flip)", fontsize=12)
            
            for topo_label, (prefix, metric) in topos.items():
                if topo_label == "Star (Ditto)" and defense_mode == "Full_Defense":
                    # We didn't run Star (Ditto) with Full Defense in these sweeps
                    continue

                y_acc = []
                for rate_str in rates:
                    if rate_str == "No_Attack":
                        fname = f"{prefix}_{defense_mode}_No_Attack_metrics.csv"
                    else:
                        fname = f"{prefix}_{defense_mode}_label_flip_{rate_str}_metrics.csv"
                        
                    fpath = os.path.join(folder, fname)
                    if os.path.exists(fpath):
                        data = load_csv(fpath)
                        if metric in data.columns and not data.empty:
                            final_acc = data[metric].iloc[-1]
                            if pd.isna(final_acc): final_acc = 0.0
                        else:
                            final_acc = 0.0
                        y_acc.append(final_acc)
                    else:
                        y_acc.append(None)
                
                # Check if we have valid data before plotting
                if any(y is not None for y in y_acc):
                    linewidth = 3 if topo_label == "HE" else 2
                    ax.plot(x_rates, y_acc, label=topo_label, linewidth=linewidth, marker='s', markersize=8)
            
            if ax == axes[0]:
                ax.set_ylabel("Final Accuracy (%)", fontsize=12)
            ax.set_ylim(0, 100)
            ax.set_xticks(x_rates)
            ax.legend(title="Topology", fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(os.path.join(ARTIFACT_DIR, out_name), dpi=300)
        plt.close()

if __name__ == "__main__":
    plot_partA()
    plot_partB()
    print("Plots generated successfully in artifact directory.")
