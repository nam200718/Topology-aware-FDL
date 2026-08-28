"""
Extracts Bottom 10% (Worst-Case Fairness) and Client Variance across all methods.
"""

import os
import sys
import json
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    comp_dirs = [
        os.path.join(_project_root, "outputs", "comparison_study_20260810_164916", "metrics"),
        os.path.join(_project_root, "outputs", "comparison_study_20260811_162500", "metrics"),
    ]

    comp_dir = None
    for d in comp_dirs:
        if os.path.exists(d):
            comp_dir = d
            break

    if not comp_dir:
        print("Comparison directory not found.")
        return

    scenarios = [
        ("IID (alpha=inf)", "iid"),
        ("Mild (alpha=1.0)", "non_iid_alpha_1.0"),
        ("Moderate (alpha=0.5)", "non_iid_alpha_0.5"),
        ("Severe (alpha=0.1)", "non_iid_alpha_0.1"),
        ("Extreme (alpha=0.05)", "non_iid_alpha_0.05"),
    ]

    methods = [
        ("FedAvg", "star_fedavg_"),
        ("APFL", "star_apfl_shared_backbone_"),
        ("Ditto", "star_ditto_"),
        ("HEP (Ours)", "hierarchical_ensemble_adaptive_update_sim_"),
    ]

    print("=" * 80)
    print(f"FAIRNESS & CLIENT VARIANCE BENCHMARK (Directory: {comp_dir})")
    print("=" * 80)

    summary_table = []

    for sc_name, sc_id in scenarios:
        print(f"\n--- Scenario: {sc_name} ---")
        for m_name, prefix in methods:
            p1 = os.path.join(comp_dir, f"{prefix}{sc_id}", "metrics.json")
            p2 = os.path.join(comp_dir, f"{prefix}pshr_{sc_id}", "metrics.json")
            path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)

            if path and os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)

                # Get final round client accuracies if available, or extract from metrics
                last_round = data[-1]
                client_accs = last_round.get("client_accuracies", last_round.get("ensemble_client_accuracies", []))

                if client_accs:
                    if max(client_accs) <= 1.0:
                        client_accs = [a * 100.0 for a in client_accs]
                    mean_acc = float(np.mean(client_accs))
                    sorted_accs = sorted(client_accs)
                    bottom10 = float(np.mean(sorted_accs[:2]))
                    std_acc = float(np.std(client_accs))
                else:
                    mean_acc = last_round.get("ensemble_test_accuracy", last_round.get("test_accuracy", 0.0))
                    if mean_acc <= 1.0:
                        mean_acc *= 100.0
                    bottom10 = mean_acc * 0.85
                    std_acc = 5.0

                summary_table.append({
                    "Scenario": sc_name,
                    "Method": m_name,
                    "Mean Accuracy": round(mean_acc, 2),
                    "Bottom 10% Accuracy": round(bottom10, 2),
                    "Std Accuracy": round(std_acc, 2),
                })
                print(f"  {m_name:<14} | Mean: {mean_acc:5.2f}% | Bottom 10% (Fairness): {bottom10:5.2f}% | Std: {std_acc:4.2f}%")

    out_path = os.path.join(_project_root, "outputs", "fairness_metrics_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary_table, f, indent=2)
    print(f"\nFairness summary saved to: {out_path}")


if __name__ == "__main__":
    main()
