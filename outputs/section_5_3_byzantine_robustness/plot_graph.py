import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "data_byzantine.csv"))

fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
axes = axes.flatten()

attacks = [
    ('label_flip', 'Label Flipping Attack'),
    ('sign_flip', 'Sign Flipping Attack'),
    ('gradient_ascent', 'Gradient Ascent Attack'),
    ('random_noise', 'Gaussian Noise Attack')
]

palette = {
    'FedAvg': '#7f7f7f',
    'FedProx': '#9467bd',
    'Multi-Krum': '#8c564b',
    'SCAFFOLD': '#2ca02c',
    'Ditto': '#ff7f0e',
    'FedRep': '#17becf',
    'FedHEP': '#1f77b4',
    'Defended FedHEP': '#d62728'
}

rates = [0.0, 0.1, 0.2, 0.3, 0.4]

for i, (atk_id, atk_title) in enumerate(attacks):
    ax = axes[i]
    atk_df = df[df['attack'] == atk_id]

    for method in ['FedAvg', 'Multi-Krum', 'Ditto', 'FedHEP', 'Defended FedHEP']:
        m_data = atk_df[atk_df['method'] == method].sort_values('byzantine_rate')
        if len(m_data) > 0:
            is_ours = 'Defended FedHEP' in method
            lw = 3.0 if is_ours else 1.8
            ms = 8 if is_ours else 5
            zorder = 10 if is_ours else 3
            marker = 'D' if is_ours else 'o'
            ax.plot(m_data['byzantine_rate'], m_data['mean_acc'], marker=marker,
                    lw=lw, markersize=ms, label=method, color=palette.get(method, '#333333'),
                    zorder=zorder)

    ax.set_title(atk_title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Byzantine Attacker Fraction ($q$)", fontsize=10, labelpad=6)
    ax.set_ylabel("Accuracy (%)", fontsize=10, labelpad=6)
    ax.set_xticks(rates)
    ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%'])
    ax.set_ylim(0, 55)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)

plt.suptitle("CIFAR-100 Byzantine Robustness Breakdown across Attacks and Byzantine Ratios ($q$)",
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
out_png = os.path.join(current_dir, "graph_byzantine_robustness.png")
plt.savefig(out_png, dpi=300)
print(f"Graph generated: {out_png}")
