from src.experiments.visualizer import (
    plot_experiment_results,
    plot_comparison_convergence as plot_comparison,
    plot_robustness_summary,
    plot_byzantine_matrix,
)

__all__ = [
    "plot_experiment_results",
    "plot_comparison",
    "plot_robustness_summary",
    "plot_byzantine_matrix",
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TopoFDL Visualization Tool")
    parser.add_argument("metrics_path", help="Path to metrics.json or metrics.csv")
    parser.add_argument("--out_dir", help="Directory to save charts", default=None)
    
    args = parser.parse_args()
    plot_experiment_results(args.metrics_path, args.out_dir)
