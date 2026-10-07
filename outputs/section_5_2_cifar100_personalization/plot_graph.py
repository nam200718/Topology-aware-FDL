import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "data_cifar100.csv"))

# Native 1-column canvas: 3.33 x 1.75 inches for AAMAS format
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.33, 1.75), dpi=300)

regime_order = ['iid', 'mild', 'moderate', 'severe', 'extreme']
# Multiline x-axis labels to maximize readability
regime_labels = [r'IID' + '\n' + r'($\infty$)', 'Mild\n(1.0)', 'Mod\n(0.5)', 'Sev\n(0.1)', 'Ext\n(0.05)']

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
    lw = 1.3 if 'FedHEP' in method else 0.85
    ms = 3.5 if 'FedHEP' in method else 2.5
    alpha = 1.0 if 'FedHEP' in method else 0.85
    zorder = 10 if 'FedHEP' in method else 3
    ax1.plot(regime_labels, m_data['mean_acc'], marker='o', lw=lw, markersize=ms,
             label=method, color=palette.get(method, '#333333'), alpha=alpha, zorder=zorder)
    ax2.plot(regime_labels, m_data['mean_b10'], marker='s', lw=lw, markersize=ms,
             label=method, color=palette.get(method, '#333333'), alpha=alpha, zorder=zorder)

# Zero top titles: rely on LaTeX caption
ax1.set_ylabel('Mean Acc (%)', fontsize=7.2, labelpad=1.5)
ax1.set_ylim(10, 75)
ax1.tick_params(axis='both', which='major', labelsize=6.2, pad=1)

ax2.set_ylabel('Bottom 10% (%)', fontsize=7.2, labelpad=1.5)
ax2.set_ylim(0, 62)
ax2.tick_params(axis='both', which='major', labelsize=6.2, pad=1)

# Single shared top legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.88),
           ncol=4, frameon=True, framealpha=0.95, edgecolor='#dddddd',
           fontsize=5.6, handlelength=1.1, handletextpad=0.25, columnspacing=0.6)

plt.subplots_adjust(left=0.13, right=0.98, bottom=0.23, top=0.84, wspace=0.34)

out_png = os.path.join(current_dir, "graph_cifar100_personalization.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
paper_png = os.path.join(current_dir, "../../paper/figures/graph_cifar100_personalization.png")
if os.path.exists(os.path.dirname(paper_png)):
    plt.savefig(paper_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f"Graph generated: {out_png}")
