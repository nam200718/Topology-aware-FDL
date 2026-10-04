import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "data_cifar100.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

regime_order = ['iid', 'mild', 'moderate', 'severe', 'extreme']
regime_labels = [r'IID ($\alpha=\infty$)', r'Mild ($\alpha=1.0$)', r'Moderate ($\alpha=0.5$)', r'Severe ($\alpha=0.1$)', r'Extreme ($\alpha=0.05$)']

palette = {
    'FedAvg': '#7f7f7f',
    'FedProx': '#9467bd',
    'Multi-Krum': '#8c564b',
    'SCAFFOLD': '#2ca02c',
    'Ditto': '#ff7f0e',
    'FedRep': '#17becf',
    'FedHEP': '#d62728'
}

for method in df['method'].unique():
    m_data = df[df['method'] == method].set_index('regime').reindex(regime_order)
    lw = 2.8 if 'FedHEP' in method else 1.8
    ms = 8 if 'FedHEP' in method else 6
    alpha = 1.0 if 'FedHEP' in method else 0.8
    zorder = 10 if 'FedHEP' in method else 3
    ax1.plot(regime_labels, m_data['mean_acc'], marker='o', lw=lw, markersize=ms,
             label=method, color=palette.get(method, '#333333'), alpha=alpha, zorder=zorder)
    ax2.plot(regime_labels, m_data['mean_b10'], marker='s', lw=lw, markersize=ms,
             label=method, color=palette.get(method, '#333333'), alpha=alpha, zorder=zorder)

ax1.set_title("CIFAR-100 Personalization Accuracy across Heterogeneity", fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel(r"Dirichlet Skew Regime ($\alpha$)", fontsize=11, labelpad=8)
ax1.set_ylabel("Mean Accuracy (%)", fontsize=11, labelpad=8)
ax1.set_ylim(10, 75)
ax1.legend(loc='upper right', frameon=True, framealpha=0.9)

ax2.set_title("CIFAR-100 Bottom 10% Fairness across Heterogeneity", fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel(r"Dirichlet Skew Regime ($\alpha$)", fontsize=11, labelpad=8)
ax2.set_ylabel("Bottom 10% Client Accuracy (%)", fontsize=11, labelpad=8)
ax2.set_ylim(0, 65)
ax2.legend(loc='upper right', frameon=True, framealpha=0.9)

plt.tight_layout()
out_png = os.path.join(current_dir, "graph_cifar100_personalization.png")
plt.savefig(out_png, dpi=300)
print(f"Graph generated: {out_png}")
