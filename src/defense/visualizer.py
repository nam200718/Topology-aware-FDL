import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from src.defense.trust_tracker import TrustTracker


class DefenseVisualizer:
    """Create visualization for defense analysis."""

    @staticmethod
    def plot_trust_heatmap(
        trust_tracker: TrustTracker,
        byzantine_ids: List[int],
        output_path: str,
        title: str = "Trust Score Evolution per Client",
    ):
        """
        Heatmap of trust score over rounds.
        Byzantine clients have trust score decreasing (red/dark color).

        Args:
            trust_tracker: TrustTracker.
            byzantine_ids: List of client IDs are Byzantine.
            output_path: Image file path.
            title
        """
        client_ids, rounds, matrix = trust_tracker.to_matrix()

        if not client_ids or not rounds:
            print("Warning: No trust data to plot.")
            return

        matrix_np = np.array(matrix)

        fig, ax = plt.subplots(figsize=(max(8, len(rounds) * 0.6), max(4, len(client_ids) * 0.4)))

        # Tạo custom colormap: đỏ (trust thấp) → xanh lá (trust cao)
        cmap = sns.diverging_palette(10, 130, as_cmap=True)

        sns.heatmap(
            matrix_np,
            ax=ax,
            xticklabels=[str(r) for r in rounds],
            yticklabels=[
                f"C{cid} {'⚠ BYZ' if cid in byzantine_ids else ''}"
                for cid in client_ids
            ],
            cmap=cmap,
            vmin=0.0,
            vmax=max(0.5, np.nanmax(matrix_np)),
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            cbar_kws={"label": "Trust Score"},
        )

        ax.set_xlabel("Round")
        ax.set_ylabel("Client")
        ax.set_title(title)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Trust heatmap saved to {output_path}")

    @staticmethod
    def plot_accuracy_vs_byzantine_rate(
        results: List[Dict],
        output_path: str,
        title: str = "Accuracy vs. Byzantine Rate",
    ):
        """
        Compare accuracy with/without defense over attack rates.

        Args:
            results: List of dicts, each dict has:
                - "byzantine_rate": float
                - "defense_mode": str ("none" or "soft_cosine")
                - "final_accuracy": float
                - "label": str (optional, dùng cho legend)
            output_path: Image file path.
            title: Title.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by defense_mode
        groups: Dict[str, Tuple[List[float], List[float]]] = {}
        for r in results:
            mode = r.get("defense_mode", r.get("label", "unknown"))
            if mode not in groups:
                groups[mode] = ([], [])
            groups[mode][0].append(r["byzantine_rate"])
            groups[mode][1].append(r["final_accuracy"])

        # Style mapping
        style_map = {
            "none": {"color": "#e74c3c", "marker": "o", "linestyle": "--", "label": "No Defense (FedAvg)"},
            "soft_cosine": {"color": "#2ecc71", "marker": "s", "linestyle": "-", "label": "Soft Rejection (Cosine)"},
            "soft_norm": {"color": "#3498db", "marker": "^", "linestyle": "-.", "label": "Soft Rejection (Norm)"},
        }

        for mode, (rates, accs) in groups.items():
            # Sort by rate
            sorted_pairs = sorted(zip(rates, accs))
            sorted_rates = [p[0] for p in sorted_pairs]
            sorted_accs = [p[1] for p in sorted_pairs]

            style = style_map.get(mode, {"color": "gray", "marker": "D", "linestyle": ":", "label": mode})
            ax.plot(
                sorted_rates,
                sorted_accs,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                label=style["label"],
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Byzantine Rate", fontsize=12)
        ax.set_ylabel("Final Accuracy (%)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        # Format x-axis as percentages
        ax.set_xticks([r for r in sorted(set(r["byzantine_rate"] for r in results))])
        ax.set_xticklabels(
            [f"{int(r * 100)}%" for r in sorted(set(r["byzantine_rate"] for r in results))]
        )

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Accuracy comparison chart saved to {output_path}")
