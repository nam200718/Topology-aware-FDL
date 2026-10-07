import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "data_mobilenet.json"), "r") as f:
    meta = json.load(f)

sim_res = meta["simulated_resource_profile"]

# Native 1-column canvas: 3.33 x 1.6 inches for AAMAS format
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.33, 1.6), dpi=300)

methods_res = ['FedAvg', 'Ditto', 'FedHEP']
vram = [sim_res[m if m in sim_res else 'FedHEP (Ours)']['simulated_peak_vram_mb'] for m in methods_res]
latency = [sim_res[m if m in sim_res else 'FedHEP (Ours)']['simulated_latency_ms'] for m in methods_res]
colors = ['#7f7f7f', '#ff7f0e', '#d62728']

x = np.arange(len(methods_res))
bars1 = ax1.bar(x, vram, color=colors, width=0.52, alpha=0.9)
for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 6, f"{y:.0f}", ha='center', va='bottom', fontsize=5.2, fontweight='bold')

# Zero top titles: rely on LaTeX caption
ax1.set_xticks(x)
ax1.set_xticklabels(methods_res, fontsize=6.2)
ax1.set_ylabel('Peak VRAM (MB)', fontsize=7.2, labelpad=1.5)
ax1.set_ylim(0, 360)
ax1.tick_params(axis='both', which='major', labelsize=6.0, pad=1)

bars2 = ax2.bar(x, latency, color=colors, width=0.52, alpha=0.9)
for bar in bars2:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + 0.6, f"{y:.1f}", ha='center', va='bottom', fontsize=5.5, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(methods_res, fontsize=6.2)
ax2.set_ylabel('Batch Latency (ms)', fontsize=7.2, labelpad=1.5)
ax2.set_ylim(0, 44)
ax2.tick_params(axis='both', which='major', labelsize=6.0, pad=1)

plt.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.92, wspace=0.34)

out_png = os.path.join(current_dir, "graph_mobilenet_simulated_edge.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
paper_png = os.path.join(current_dir, "../../paper/figures/graph_mobilenet_simulated_edge.png")
if os.path.exists(os.path.dirname(paper_png)):
    plt.savefig(paper_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f"Graph generated: {out_png}")
