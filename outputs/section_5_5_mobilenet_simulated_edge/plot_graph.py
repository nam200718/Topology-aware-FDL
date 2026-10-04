import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "data_mobilenet.json"), "r") as f:
    meta = json.load(f)

df_acc = pd.DataFrame(meta["accuracy_benchmarks"])
sim_res = meta["simulated_resource_profile"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

# Panel 1: Simulated VRAM
methods_res = ['FedAvg', 'Ditto', 'FedHEP (Ours)']
vram = [sim_res[m]['simulated_peak_vram_mb'] for m in methods_res]
latency = [sim_res[m]['simulated_latency_ms'] for m in methods_res]
colors = ['#7f7f7f', '#ff7f0e', '#d62728']

x = np.arange(len(methods_res))
bars1 = ax1.bar(x, vram, color=colors, width=0.55, alpha=0.9)
for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 5, f"{y:.1f} MB", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax1.set_title("Simulated Peak VRAM Footprint", fontsize=11, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(methods_res, fontsize=10)
ax1.set_ylabel("Simulated Peak VRAM (MB)", fontsize=10)
ax1.set_ylim(0, 360)

# Panel 2: Simulated Batch Latency
bars2 = ax2.bar(x, latency, color=colors, width=0.55, alpha=0.9)
for bar in bars2:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + 0.6, f"{y:.2f} ms", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax2.set_title("Simulated Batch Latency", fontsize=11, fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(methods_res, fontsize=10)
ax2.set_ylabel("Simulated Latency (ms)", fontsize=10)
ax2.set_ylim(0, 42)

# Panel 3: Personalization Accuracy (Moderate vs Extreme)
scens = ['moderate', 'extreme']
scen_labels = [r'Moderate ($\alpha=0.5$)', r'Extreme ($\alpha=0.05$)']
x_acc = np.arange(len(scens))
width = 0.25

meths_acc = ['FedAvg', 'Ditto', 'FedHEP']
acc_palette = {'FedAvg': '#7f7f7f', 'Ditto': '#ff7f0e', 'FedHEP': '#d62728'}

for idx, m in enumerate(meths_acc):
    sub = df_acc[df_acc['method'] == m].set_index('regime').reindex(scens)
    offset = (idx - 1) * width
    b = ax3.bar(x_acc + offset, sub['mean_acc'], width, label=m, color=acc_palette.get(m, '#333333'), alpha=0.9)
    for bar in b:
        y = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, y + 0.6, f"{y:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3.set_title("MobileNetV3 Personalization Accuracy", fontsize=11, fontweight='bold', pad=10)
ax3.set_xticks(x_acc)
ax3.set_xticklabels(scen_labels, fontsize=10)
ax3.set_ylabel("Mean Accuracy (%)", fontsize=10)
ax3.set_ylim(0, 55)
ax3.legend(loc='upper left', frameon=True, framealpha=0.9)

plt.suptitle("MobileNetV3-Small: Simulated Edge Resource Profiling & Personalization Accuracy (Simulated, Non-Physical Hardware)",
             fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout()
out_png = os.path.join(current_dir, "graph_mobilenet_simulated_edge.png")
plt.savefig(out_png, dpi=300)
print(f"Graph generated: {out_png}")
