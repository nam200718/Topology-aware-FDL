import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_experiment_results(metrics_path, output_dir=None):
    """
    Plots accuracy and loss from a metrics.json or metrics.csv file.
    """
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    # Load data
    if metrics_path.endswith('.json'):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(metrics_path)

    if output_dir is None:
        output_dir = os.path.dirname(metrics_path)

    sns.set_style("whitegrid")
    
    # 1. Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['round'], df['test_accuracy'], label='Global Model Accuracy', marker='o')
    
    if 'ensemble_test_accuracy' in df.columns:
        plt.plot(df['round'], df['ensemble_test_accuracy'], label='Ensemble Accuracy', marker='s', linestyle='--')
    
    plt.title('Model Accuracy Convergence')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_convergence.png'))
    plt.close()

    # 2. Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['round'], df['test_loss'], label='Global Model Loss', marker='o', color='red')
    
    if 'ensemble_test_loss' in df.columns:
        plt.plot(df['round'], df['ensemble_test_loss'], label='Ensemble Loss', marker='s', linestyle='--', color='orange')
    
    plt.title('Model Loss Convergence')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_convergence.png'))
    plt.close()

    print(f"Charts saved to {output_dir}")

def plot_comparison(experiment_dirs, labels, output_path):
    """
    Compares multiple experiments in a single plot.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    
    for exp_dir, label in zip(experiment_dirs, labels):
        json_path = os.path.join(exp_dir, 'metrics.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            
            # Plot global accuracy
            plt.plot(df['round'], df['test_accuracy'], label=f'{label} (Global)')
            
            # Plot ensemble if available
            if 'ensemble_test_accuracy' in df.columns:
                plt.plot(df['round'], df['ensemble_test_accuracy'], label=f'{label} (Ensemble)', linestyle='--')
                
    plt.title('Experiment Comparison: Test Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Comparison chart saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TopoFDL Visualization Tool")
    parser.add_argument("metrics_path", help="Path to metrics.json or metrics.csv")
    parser.add_argument("--out_dir", help="Directory to save charts", default=None)
    
    args = parser.parse_args()
    plot_experiment_results(args.metrics_path, args.out_dir)
