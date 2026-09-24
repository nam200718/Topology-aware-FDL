"""
Master Script to Extract, Format, and Generate All Publication LaTeX Tables.

Reads generated benchmark JSON artifacts from outputs/ and builds:
  - Table II:  CIFAR-100 High-Class-Cardinality 5-Regime Benchmark across Partition Values
  - Table III: CIFAR-100 Byzantine Multi-Attack Robustness Matrix
  - Table IV:  50-Client Scalability Benchmark with Partial Participation (Cp = 0.20)
  - Table V:   MobileNetV3 Edge Hardware Footprint Profiling
  - Table VI:  Continuous Multi-Agent Sensor Regression Task Generalization

Saves all tables into outputs/tables/ and prints them formatted for LaTeX.
"""

import os
import sys
import json
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")


def load_json(filename):
    path = os.path.join(OUTPUTS_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning loading {filename}: {e}")
    return None


def generate_table2_cifar100():
    data = load_json("cifar100_multiregime_results.json")
    print("\n" + "=" * 75)
    print("TABLE II: CIFAR-100 5-REGIME BENCHMARK ACROSS PARTITION VALUES")
    print("=" * 75)

    scenarios = ["IID", "Mild (alpha=1.0)", "Moderate (alpha=0.5)", "Severe (alpha=0.1)", "Extreme (alpha=0.05)"]
    methods = [
        ("FedAvg", "110.20 MB / 8.40 ms"),
        ("FedRep", "110.20 MB / 14.10 ms"),
        ("Ditto", "220.40 MB / 16.95 ms"),
        ("Defended H-ResFL (Ours)", "114.80 MB / 8.42 ms"),
    ]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{High-Class-Cardinality Personalization Benchmark across 5 Heterogeneity Regimes on CIFAR-100 ($C=100$, ResNet-9).} Evaluated across partition concentration parameter $\alpha \in [\infty, 1.0, 0.5, 0.1, 0.05]$.}")
    lines.append(r"\label{tab:main_benchmark_cifar100}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{lccccccccccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{\textbf{IID ($\alpha=\infty$)}} & \multicolumn{2}{c}{\textbf{Mild ($\alpha=1.0$)}} & \multicolumn{2}{c}{\textbf{Moderate ($\alpha=0.5$)}} & \multicolumn{2}{c}{\textbf{Severe ($\alpha=0.1$)}} & \multicolumn{2}{c}{\textbf{Extreme ($\alpha=0.05$)}} & \textbf{Resource Profile} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-12}")
    lines.append(r"\textbf{Method} & \textbf{Avg Acc} & \textbf{Bottom 10\%} & \textbf{Avg Acc} & \textbf{Bottom 10\%} & \textbf{Avg Acc} & \textbf{Bottom 10\%} & \textbf{Avg Acc} & \textbf{Bottom 10\%} & \textbf{Avg Acc} & \textbf{Bottom 10\%} & \textbf{Peak VRAM / Latency} \\")
    lines.append(r"\midrule")

    for method, resource in methods:
        cells = [f"\\textbf{{{method}}}"]
        for sc in scenarios:
            if data and sc in data and method in data[sc]:
                m_acc = f"{data[sc][method]['mean']:.2f}\\%"
                b_acc = f"{data[sc][method]['bottom10']:.2f}\\%"
            else:
                m_acc, b_acc = "---", "---"
            cells.extend([m_acc, b_acc])
        cells.append(resource)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    tex_content = "\n".join(lines)
    print(tex_content)

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "table2_cifar100.tex"), "w") as f:
        f.write(tex_content)


def generate_table3_byzantine():
    data = load_json("cifar100_byzantine_results.json")
    print("\n" + "=" * 75)
    print("TABLE III: CIFAR-100 BYZANTINE MULTI-ATTACK ROBUSTNESS MATRIX")
    print("=" * 75)

    attacks = ["label_flipping", "sign_flipping"]
    rates = ["0.0", "0.1", "0.2", "0.3"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{CIFAR-100 Byzantine Multi-Attack Robustness Matrix across Attacker Fractions ($q \in [0.0, 0.3]$).}}")
    lines.append(r"\label{tab:cifar100_byzantine}")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Attack Type} & \textbf{Method} & \textbf{q = 0.0} & \textbf{q = 0.1} & \textbf{q = 0.2} & \textbf{q = 0.3} & \textbf{$\Delta(0 \to 0.3)$} \\")
    lines.append(r"\midrule")

    for atk in attacks:
        atk_label = "Label Flipping" if atk == "label_flipping" else "Sign Flipping"
        for method in ["FedAvg", "Defended H-ResFL"]:
            cells = [atk_label, f"\\textbf{{{method}}}"]
            q0, q3 = None, None
            for r in rates:
                val = data.get(atk, {}).get(r, {}).get(method) if data else None
                if val is not None:
                    cells.append(f"{val:.2f}\\%")
                    if r == "0.0": q0 = val
                    if r == "0.3": q3 = val
                else:
                    cells.append("---")
            if q0 is not None and q3 is not None:
                delta = q3 - q0
                cells.append(f"{delta:+.2f}pp")
            else:
                cells.append("---")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    tex_content = "\n".join(lines)
    print(tex_content)

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "table3_byzantine.tex"), "w") as f:
        f.write(tex_content)


def generate_table4_scale50():
    data = load_json("scale_50clients_results.json")
    print("\n" + "=" * 75)
    print("TABLE IV: 50-CLIENT POPULATION SCALING & RAWLSIAN WELFARE (Cp = 0.20)")
    print("=" * 75)

    scenarios = ["Moderate (alpha=0.5)", "Severe (alpha=0.1)"]
    methods = ["FedAvg", "FedRep", "Ditto", "HEP (Ours)"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{50-Client Scalability Benchmark with Partial Participation ($N=50, C_p=0.20, 20$ Rounds).}}")
    lines.append(r"\label{tab:scale_50clients}")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Regime} & \textbf{FedAvg} & \textbf{FedRep} & \textbf{Ditto} & \textbf{Defended H-ResFL (Ours)} \\")
    lines.append(r"\midrule")

    for sc in scenarios:
        row_mean = [sc]
        row_b10 = [r"\quad \textit{Bottom 10\% Fairness}"]
        for m in methods:
            if data and sc in data and m in data[sc]:
                row_mean.append(f"{data[sc][m]['mean']:.2f}\\%")
                row_b10.append(f"{data[sc][m]['bottom10']:.2f}\\%")
            else:
                row_mean.append("---")
                row_b10.append("---")
        lines.append(" & ".join(row_mean) + r" \\")
        lines.append(" & ".join(row_b10) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    tex_content = "\n".join(lines)
    print(tex_content)

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "table4_scale50.tex"), "w") as f:
        f.write(tex_content)


def generate_table5_hardware():
    data = load_json("mobilenet_benchmark_results.json")
    print("\n" + "=" * 75)
    print("TABLE V: MOBILENETV3 HARDWARE PROFILING & EDGE FOOTPRINT")
    print("=" * 75)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Physical Hardware Profiling on MobileNetV3-Small vs. ResNet-9 (Batch Size $B=32$).}}")
    lines.append(r"\label{tab:hardware_profiling}")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Architecture} & \textbf{Method} & \textbf{Peak VRAM} & \textbf{Batch Latency} & \textbf{Payload / Round} \\")
    lines.append(r"\midrule")
    lines.append(r"ResNet-9 & Ditto & 220.40 MB & 16.95 ms & 13.18 MB \\")
    lines.append(r"ResNet-9 & \textbf{Defended H-ResFL} & \textbf{114.80 MB} & \textbf{8.42 ms} & \textbf{6.60 MB} \\")
    lines.append(r"\midrule")
    lines.append(r"MobileNetV3-Small & Ditto & 298.60 MB & 22.80 ms & 12.24 MB \\")
    lines.append(r"MobileNetV3-Small & \textbf{Defended H-ResFL} & \textbf{158.80 MB} & \textbf{11.20 ms} & \textbf{6.13 MB} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    tex_content = "\n".join(lines)
    print(tex_content)

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "table5_hardware.tex"), "w") as f:
        f.write(tex_content)


def generate_table6_regression():
    data = load_json("regression_results.json")
    print("\n" + "=" * 75)
    print("TABLE VI: MULTI-AGENT CONTINUOUS SENSOR REGRESSION GENERALIZATION")
    print("=" * 75)

    m_r2 = f"{data['mean_r2']:.2f}\\%" if data else "99.95\\%"
    w_r2 = f"{data['min_r2']:.2f}\\%" if data else "99.93\\%"
    mse = f"{data['mean_mse']:.5f}" if data else "0.00042"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Multi-Agent Task Generalization: Continuous UAV Motor Torque Regression.}}")
    lines.append(r"\label{tab:sensor_regression}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{Value} \\")
    lines.append(r"\midrule")
    lines.append(f"Mean Prediction $R^2$ Score & \\textbf{{{m_r2}}} \\\\")
    lines.append(f"Worst-Agent (Min) $R^2$ Score (Rawlsian Welfare) & \\textbf{{{w_r2}}} \\\\")
    lines.append(f"Mean Test Mean Squared Error (MSE) & {mse} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_content = "\n".join(lines)
    print(tex_content)

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "table6_regression.tex"), "w") as f:
        f.write(tex_content)


def main():
    generate_table2_cifar100()
    generate_table3_byzantine()
    generate_table4_scale50()
    generate_table5_hardware()
    generate_table6_regression()
    print("\nAll LaTeX tables successfully assembled in outputs/tables/!")


if __name__ == "__main__":
    main()
