import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "data_byzantine.csv"))

# Native 1-column canvas: 3.33 x 2.65 inches (2x2 grid) for AAMAS format
fig, axes = plt.subplots(2, 2, figsize=(3.33, 2.65), dpi=300)
axes = axes.flatten()

attacks = [
    ('label_flip', '(a) Label Flip'),
    ('sign_flip', '(b) Sign Flip'),
    ('gradient_ascent', '(c) Grad Ascent'),
    ('random_noise', '(d) Gaussian Noise')
]

palette = {
    'FedAvg': '#7f7f7f',
    'Multi-Krum': '#8c564b',
    'Ditto': '#ff7f0e',
    'FedHEP': '#d62728'
}

rates = [0.0, 0.1, 0.2, 0.3, 0.4]

for i, (atk_id, atk_title) in enumerate(attacks):
    ax = axes[i]
    atk_df = df[df['attack'] == atk_id]

    for method in ['FedAvg', 'Multi-Krum', 'Ditto', 'Defended FedHEP']:
        m_data = atk_df[atk_df['method'] == method].sort_values('byzantine_rate')
        if len(m_data) > 0:
            is_ours = 'Defended FedHEP' in method
            label = 'FedHEP' if is_ours else method
            lw = 1.3 if is_ours else 0.85
            ms = 3.5 if is_ours else 2.5
            zorder = 10 if is_ours else 3
            marker = 'D' if is_ours else 'o'
            ax.plot(m_data['byzantine_rate'], m_data['mean_acc'], marker=marker,
                    lw=lw, markersize=ms, label=label, color=palette.get(label, '#333333'),
                    zorder=zorder)

    # Subplot label in corner badge
    ax.text(0.5, 0.92, atk_title, transform=ax.transAxes, ha='center', va='top', fontsize=6.0, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='#cccccc', lw=0.5, alpha=0.92))
    ax.set_xticks(rates)
    ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%'], fontsize=5.2)
    ax.set_ylim(0, 62)
    ax.tick_params(axis='both', which='major', labelsize=5.5, pad=1)
    if i in [0, 2]:
        ax.set_ylabel('Acc (%)', fontsize=6.8, labelpad=1.5)
    if i in [2, 3]:
        ax.set_xlabel('Attacker Ratio ($q$)', fontsize=6.2, labelpad=1.5)

# Single shared top legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.92),
           ncol=4, frameon=True, framealpha=0.95, edgecolor='#dddddd',
           fontsize=5.8, handlelength=1.1, handletextpad=0.25, columnspacing=0.8)

plt.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.88, wspace=0.25, hspace=0.32)

out_png = os.path.join(current_dir, "graph_byzantine_robustness.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
paper_png = os.path.join(current_dir, "../../paper/figures/graph_byzantine_robustness.png")
if os.path.exists(os.path.dirname(paper_png)):
    plt.savefig(paper_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f"Graph generated: {out_png}")
