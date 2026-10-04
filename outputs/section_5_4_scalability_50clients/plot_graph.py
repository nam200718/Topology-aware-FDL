import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "data_scale50.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

scenarios = ['Moderate (alpha=0.5)', 'Severe (alpha=0.1)']
scen_labels = [r'Moderate ($\alpha=0.5$)', r'Severe ($\alpha=0.1$)']
methods = ['FedAvg', 'FedRep', 'Ditto', 'Defended FedHEP']

palette = {
    'FedAvg': '#7f7f7f',
    'FedRep': '#17becf',
    'Ditto': '#ff7f0e',
    'Defended FedHEP': '#d62728'
}

x = np.arange(len(scenarios))
width = 0.18

for idx, m in enumerate(methods):
    m_data = df[df['method'] == m].set_index('regime').reindex(scenarios)
    offset = (idx - 1.5) * width
    bars1 = ax1.bar(x + offset, m_data['mean_acc'], width, label=m, color=palette.get(m, '#333333'), alpha=0.9)
    bars2 = ax2.bar(x + offset, m_data['mean_b10'], width, label=m, color=palette.get(m, '#333333'), alpha=0.9)

    for bar in bars1:
        yval = bar.get_height()
        if yval > 0:
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8.5)
    for bar in bars2:
        yval = bar.get_height()
        if yval > 0:
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8.5)

ax1.set_title("50-Client Scalability: Mean Accuracy ($C_p=0.20$)", fontsize=12, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(scen_labels, fontsize=11)
ax1.set_ylabel("Mean Accuracy (%)", fontsize=11)
ax1.set_ylim(0, 45)
ax1.legend(loc='upper right', frameon=True, framealpha=0.9)

ax2.set_title("50-Client Scalability: Bottom 10% Fairness ($C_p=0.20$)", fontsize=12, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(scen_labels, fontsize=11)
ax2.set_ylabel("Bottom 10% Fairness (%)", fontsize=11)
ax2.set_ylim(0, 30)
ax2.legend(loc='upper right', frameon=True, framealpha=0.9)

plt.suptitle("50-Client Partial Participation Scalability & Fairness on CIFAR-100", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
out_png = os.path.join(current_dir, "graph_scalability_50clients.png")
plt.savefig(out_png, dpi=300)
print(f"Graph generated: {out_png}")
