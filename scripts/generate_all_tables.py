"""Master script to extract, format, and generate LaTeX tables from Tier A rerun artifacts.

Covers:
  - Table III: Main Personalization Benchmark (5 regimes x 10 methods, single-seed 42)
  - Table V: Byzantine Label-Flipping Fault Tolerance (5 rates x 4 methods)
  - Table VI: Component Ablations (IID, Moderate, Extreme)
  - Table VII: K=1 Bipartite Certification (5 regimes)
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics(metrics_file):
    hist = load_json(metrics_file)
    if not hist or not isinstance(hist, list):
        return None
    evals = [h for h in hist if h.get("evaluated") and ("ensemble_test_accuracy" in h or "test_accuracy" in h)]
    if not evals:
        return None
    accs = [h.get("ensemble_test_accuracy", h.get("test_accuracy", 0.0)) for h in evals]
    final = evals[-1]
    final_acc = float(accs[-1])
    # If evaluated on every round (>= 15 evals), use last-5 rounds average; otherwise use final round
    if len(evals) >= 15:
        last5 = float(np.mean(accs[-5:]))
    else:
        last5 = final_acc
    b10 = final.get("bottom10_fairness", None)
    return {"last5": last5, "final": final_acc, "bot10": b10, "eval_count": len(evals)}


def parse_comparison_tier_a():
    data = defaultdict(dict)
    
    scen_map = {
        "iid": "IID",
        "non_iid_alpha_1.0": "Mild",
        "non_iid_alpha_0.5": "Moderate",
        "non_iid_alpha_0.1": "Severe",
        "non_iid_alpha_0.05": "Extreme"
    }

    search_dirs = [
        os.path.join(PROJECT_ROOT, "outputs", "baseline_fidelity"),
        os.path.join(PROJECT_ROOT, "outputs", "remaining_queue"),
        os.path.join(PROJECT_ROOT, "outputs"),
        os.path.join(PROJECT_ROOT, "outputs", "tier_a_rerun", "comparison"),
    ]

    for base_dir in search_dirs:
        pattern = os.path.join(base_dir, "**", "metrics", "*", "metrics.json")
        for mf in glob.glob(pattern, recursive=True):
            exp_name = os.path.basename(os.path.dirname(mf))
            scen_key = None
            for k in scen_map.keys():
                if exp_name.endswith(k):
                    scen_key = k
                    break
            if not scen_key:
                continue

            topo_part = exp_name[:-len(scen_key)].rstrip("_").lower()
            norm_name = None
            if "hierarchical_ensemble" in topo_part or "hep" in topo_part:
                norm_name = "HEP (Ours)"
            elif "ditto" in topo_part:
                norm_name = "Ditto"
            elif "apfl" in topo_part:
                norm_name = "APFL"
            elif "fedrep" in topo_part:
                norm_name = "FedRep"
            elif "fedper" in topo_part:
                norm_name = "FedPer"
            elif "fedbabu" in topo_part:
                norm_name = "FedBABU"
            elif "fedala" in topo_part:
                norm_name = "FedALA"
            elif "cfl" in topo_part:
                norm_name = "CFL"
            elif "local" in topo_part:
                norm_name = "Local-Only"
            elif "fedavg" in topo_part or topo_part == "star":
                norm_name = "FedAvg"

            if norm_name:
                res = extract_metrics(mf)
                if res:
                    data[norm_name][scen_key] = res

    return data


def generate_table3():
    data = parse_comparison_tier_a()

    methods_order = [
        ("A. Global Consensus FL (No Personalization)", ["FedAvg"]),
        ("B. Dual-Model Regularization (Full Model Duplication)", ["APFL", "Ditto"]),
        ("C. Decoupled / Split-Head Paradigms (Single Backbone, Local Heads)", ["Local-Only", "FedPer", "FedRep", "FedBABU"]),
        ("D. Clustered & Adaptive-Aggregation Paradigms", ["FedALA", "CFL"]),
        ("E. Hierarchical Ensemble Personalization (Proposed)", ["HEP (Ours)"]),
    ]

    scenarios = ["iid", "non_iid_alpha_1.0", "non_iid_alpha_0.5", "non_iid_alpha_0.1", "non_iid_alpha_0.05"]

    resource_profiles = {
        "FedAvg": "108.58 MB / 15.1s",
        "APFL": "217.15 MB / 24.5s",
        "Ditto": "217.15 MB / 25.8s",
        "Local-Only": "108.58 MB / 6.2s",
        "FedPer": "108.58 MB / 15.8s",
        "FedRep": "108.58 MB / 16.0s",
        "FedBABU": "108.58 MB / 15.5s",
        "FedALA": "116.20 MB / 15.6s",
        "CFL": "108.58 MB / 15.1s",
        "HEP (Ours)": "113.42 MB / 16.5s"
    }

    citations = {
        "FedAvg": r"\cite{mcmahan2017communication}",
        "APFL": r"\cite{deng2020adaptive}",
        "Ditto": r"\cite{li2021ditto}",
        "Local-Only": "",
        "FedPer": r"\cite{arivazhagan2019federated}",
        "FedRep": r"\cite{collins2021exploiting}",
        "FedBABU": r"\cite{oh2021fedbabu}",
        "FedALA": r"\cite{zhang2023fedala}",
        "CFL": r"\cite{sattler2020clustered}",
        "HEP (Ours)": ""
    }

    print("\n=================== TABLE III (MAIN BENCHMARK) ===================")
    rows_tex = []
    for section_title, methods in methods_order:
        rows_tex.append(r"\multicolumn{12}{l}{\textit{\textbf{" + section_title + r"}}} \\")
        for method in methods:
            cite = citations.get(method, "")
            lbl = f"\\textbf{{{method} {cite}}}" if cite else f"\\textbf{{{method}}}"
            cells = [lbl]
            for sc in scenarios:
                val = data.get(method, {}).get(sc)
                if val:
                    last5_str = f"{val['last5']:.2f}\\%"
                    b10_str = f"{val['bot10']:.2f}\\%" if val['bot10'] is not None else "---"
                    cells.extend([last5_str, b10_str])
                else:
                    cells.extend(["---", "---"])
            cells.append(resource_profiles.get(method, "---"))
            rows_tex.append(" & ".join(cells) + r" \\")
        rows_tex.append(r"\midrule")

    full_tab3 = "\n".join(rows_tex)
    print(full_tab3)
    return full_tab3


def generate_table5_byz():
    pattern = os.path.join(PROJECT_ROOT, "outputs", "tier_a_rerun", "byz_label_flip", "*", "matrix_results.csv")
    csv_files = glob.glob(pattern)
    data = defaultdict(dict)
    rates = [0.0, 0.1, 0.2, 0.3, 0.4]

    if csv_files:
        df = pd.read_csv(csv_files[-1])
        for _, row in df.iterrows():
            topo = row["Topology"]
            rate = float(row["Byzantine Rate"])
            acc = float(row["Final Accuracy"])
            
            norm_name = None
            if "HEP" in topo or "Hierarchical" in topo:
                norm_name = "HEP (Ours)"
            elif "Ditto" in topo:
                norm_name = "Ditto"
            elif "FedRep" in topo:
                norm_name = "FedRep"
            elif "FedAvg" in topo or "Star" in topo:
                norm_name = "FedAvg"
                
            if norm_name:
                data[norm_name][rate] = acc

    print("\n=================== TABLE V (BYZANTINE LABEL FLIP) ===================")
    for method in ["FedAvg", "FedRep", "Ditto", "HEP (Ours)"]:
        row = [f"\\textbf{{{method}}}"]
        for r in rates:
            acc = data.get(method, {}).get(r)
            row.append(f"{acc:.2f}\\%" if acc is not None else "---")
        print(" & ".join(row) + r" \\")


def parse_scenario_key(scen_str):
    if "0.05" in scen_str:
        return "extreme"
    elif "0.1" in scen_str:
        return "severe"
    elif "0.5" in scen_str:
        return "moderate"
    elif "1.0" in scen_str:
        return "mild"
    elif "IID" in scen_str:
        return "iid"
    return None


def generate_table6_ablation():
    data = defaultdict(dict)
    
    # HEP Dynamic
    comp_csv = glob.glob(os.path.join(PROJECT_ROOT, "outputs", "tier_a_rerun", "comparison", "*", "comparison_results.csv"))
    if comp_csv:
        df = pd.read_csv(comp_csv[-1])
        pers_df = df[df["Metric"] == "Personalized"]
        for _, row in pers_df.iterrows():
            if "Hierarchical" in row["Topology"]:
                k = parse_scenario_key(row["Scenario"])
                if k in ["iid", "moderate", "extreme"]:
                    data["HEP (Fully Dynamic / Zero Tuning)"][k] = float(row["Final Accuracy"])

    # Ablation IID & Extreme
    ab_csv = glob.glob(os.path.join(PROJECT_ROOT, "outputs", "tier_a_rerun", "ablation", "*", "comparison_results.csv"))
    if ab_csv:
        df = pd.read_csv(ab_csv[-1])
        pers_df = df[df["Metric"] == "Personalized"]
        for _, row in pers_df.iterrows():
            topo = row["Topology"]
            k = parse_scenario_key(row["Scenario"])
            if "Distillation" in topo:
                data["w/ Asymmetric Distillation"][k] = float(row["Final Accuracy"])
            elif "Random" in topo:
                data["w/o Update-Sim (Random Clustering)"][k] = float(row["Final Accuracy"])
            elif "No Entropy" in topo or "Prior" in topo:
                data["w/o Entropy Prior (R_skew)"][k] = float(row["Final Accuracy"])

    # Ablation Moderate
    mod_csv = glob.glob(os.path.join(PROJECT_ROOT, "outputs", "tier_a_rerun", "ablation_moderate", "*", "comparison_results.csv"))
    if mod_csv:
        df = pd.read_csv(mod_csv[-1])
        pers_df = df[df["Metric"] == "Personalized"]
        for _, row in pers_df.iterrows():
            topo = row["Topology"]
            if "Random" in topo:
                data["w/o Update-Sim (Random Clustering)"]["moderate"] = float(row["Final Accuracy"])
            elif "No Entropy" in topo or "Prior" in topo:
                data["w/o Entropy Prior (R_skew)"]["moderate"] = float(row["Final Accuracy"])

    print("\n=================== TABLE VI (ABLATION STUDY) ===================")
    for var in ["HEP (Fully Dynamic / Zero Tuning)", "w/ Asymmetric Distillation", "w/o Update-Sim (Random Clustering)", "w/o Entropy Prior (R_skew)"]:
        v_iid = data.get(var, {}).get('iid')
        v_mod = data.get(var, {}).get('moderate')
        v_ext = data.get(var, {}).get('extreme')
        s_iid = f"{v_iid:.2f}\\%" if v_iid is not None else "---"
        s_mod = f"{v_mod:.2f}\\%" if v_mod is not None else "---"
        s_ext = f"{v_ext:.2f}\\%" if v_ext is not None else "---"
        row = [f"\\textbf{{{var}}}", s_iid, s_mod, s_ext]
        print(" & ".join(row) + r" \\")


def generate_table7_k1():
    data = defaultdict(dict)
    scens = [("iid", "IID"), ("mild", "Mild"), ("moderate", "Moderate"), ("severe", "Severe"), ("extreme", "Extreme")]

    comp_csv = glob.glob(os.path.join(PROJECT_ROOT, "outputs", "tier_a_rerun", "comparison", "*", "comparison_results.csv"))
    if comp_csv:
        df = pd.read_csv(comp_csv[-1])
        pers_df = df[df["Metric"] == "Personalized"]
        for _, row in pers_df.iterrows():
            if "Hierarchical" in row["Topology"]:
                k = parse_scenario_key(row["Scenario"])
                if k:
                    data[3][k] = float(row["Final Accuracy"])

    k1_csv = glob.glob(os.path.join(PROJECT_ROOT, "outputs", "tier_a_rerun", "k1_cert", "*", "comparison_results.csv"))
    if k1_csv:
        df = pd.read_csv(k1_csv[-1])
        pers_df = df[df["Metric"] == "Personalized"]
        for _, row in pers_df.iterrows():
            k = parse_scenario_key(row["Scenario"])
            if k:
                data[1][k] = float(row["Final Accuracy"])

    print("\n=================== TABLE VII (K=1 CERTIFICATION) ===================")
    for skey, sname in scens:
        k1_acc = data.get(1, {}).get(skey, 0.0)
        k3_acc = data.get(3, {}).get(skey, 0.0)
        delta = k1_acc - k3_acc
        delta_str = f"+{delta:.2f}pp" if delta >= 0 else f"{delta:.2f}pp"
        print(f"{sname:15s} | K=1: {k1_acc:.2f}% | K=3: {k3_acc:.2f}% | Delta: {delta_str}")


def main():
    generate_table3()
    generate_table5_byz()
    generate_table6_ablation()
    generate_table7_k1()


if __name__ == "__main__":
    main()
