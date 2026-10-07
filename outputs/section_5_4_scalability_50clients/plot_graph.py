import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "data_scale50.csv"))

# Native 1-column canvas: 3.33 x 1.75 inches for AAMAS format
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.33, 1.75), dpi=300)

scenarios = ['Moderate (alpha=0.5)', 'Severe (alpha=0.1)']
scen_labels = ['Moderate\n($\\alpha=0.5$)', 'Severe\n($\\alpha=0.1$)']
methods_orig = ['FedAvg', 'FedRep', 'Ditto', 'Defended FedHEP']
methods_disp = ['FedAvg', 'FedRep', 'Ditto', 'FedHEP']

palette = {
    'FedAvg': '#7f7f7f',
    'FedRep': '#17becf',
    'Ditto': '#ff7f0e',
    'FedHEP': '#d62728'
}

x = np.arange(len(scenarios))
width = 0.19

for idx, (m_orig, m_disp) in enumerate(zip(methods_orig, methods_disp)):
    m_data = df[df['method'] == m_orig].set_index('regime').reindex(scenarios)
    offset = (idx - 1.5) * width
    bars1 = ax1.bar(x + offset, m_data['mean_acc'], width, label=m_disp, color=palette[m_disp], alpha=0.9)
    bars2 = ax2.bar(x + offset, m_data['mean_b10'], width, label=m_disp, color=palette[m_disp], alpha=0.9)

    for bar in bars1:
        yval = bar.get_height()
        if yval > 0:
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{yval:.1f}", ha='center', va='bottom', fontsize=4.8, fontweight='bold')
    for bar in bars2:
        yval = bar.get_height()
        if yval > 0:
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}", ha='center', va='bottom', fontsize=4.8, fontweight='bold')

# Zero titles: rely on LaTeX caption
ax1.set_xticks(x)
ax1.set_xticklabels(scen_labels, fontsize=6.2)
ax1.set_ylabel('Mean Acc (%)', fontsize=7.2, labelpad=1.5)
ax1.set_ylim(0, 48)
ax1.tick_params(axis='both', which='major', labelsize=6.0, pad=1)

ax2.set_xticks(x)
ax2.set_xticklabels(scen_labels, fontsize=6.2)
ax2.set_ylabel('Bottom 10% (%)', fontsize=7.2, labelpad=1.5)
ax2.set_ylim(0, 26)
ax2.tick_params(axis='both', which='major', labelsize=6.0, pad=1)

# Single shared top legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.88),
           ncol=4, frameon=True, framealpha=0.95, edgecolor='#dddddd',
           fontsize=5.6, handlelength=1.1, handletextpad=0.25, columnspacing=0.6)

plt.subplots_adjust(left=0.13, right=0.98, bottom=0.23, top=0.84, wspace=0.34)

out_png = os.path.join(current_dir, "graph_scalability_50clients.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
paper_png = os.path.join(current_dir, "../../paper/figures/graph_scalability_50clients.png")
if os.path.exists(os.path.dirname(paper_png)):
    plt.savefig(paper_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f"Graph generated: {out_png}")
