"""
Fast Multi-Agent Task Generalization Demo: Continuous Sensor Time-Series Regression.
Demonstrates that the exact same Hierarchical Residual parameterization (H-ResFL)
operates seamlessly beyond classification on continuous physical systems.

Scenario:
10 autonomous UAV / drone sensor agents predicting continuous motor torque
under environmental domain drift (ambient temperature / aerodynamic drag).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.model import HierarchicalResidualLinear


def generate_synthetic_sensor_data(num_agents=10, samples_per_agent=200, input_dim=8, noise_std=0.05, seed=42):
    """
    Generates synthetic multi-agent continuous sensor data with:
    - Global underlying physical law (W_true)
    - Fleet / cluster environmental shifts (e.g. 2 clusters: urban vs rural)
    - Private individual sensor calibration drift
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Global physical relationship
    w_global = torch.randn(input_dim, 1)

    # 2. Cluster environmental shifts (2 clusters of 5 agents each)
    cluster_shifts = [torch.randn(input_dim, 1) * 0.5 for _ in range(2)]

    agent_data = {}
    for i in range(num_agents):
        cluster_id = i % 2
        # Private sensor calibration drift
        w_agent_private = torch.randn(input_dim, 1) * 0.2
        w_agent_total = w_global + cluster_shifts[cluster_id] + w_agent_private

        x = torch.randn(samples_per_agent, input_dim)
        y = x @ w_agent_total + torch.randn(samples_per_agent, 1) * noise_std

        # 80/20 train/test split
        split = int(0.8 * samples_per_agent)
        agent_data[i] = {
            "train_x": x[:split],
            "train_y": y[:split],
            "test_x": x[split:],
            "test_y": y[split:],
            "cluster_id": cluster_id,
        }
    return agent_data, input_dim


def run_regression_experiment():
    print("=" * 75)
    print("MULTI-AGENT CONTINUOUS SENSOR REGRESSION BENCHMARK (TASK GENERALIZATION)")
    print("=" * 75)

    num_agents = 10
    agent_data, input_dim = generate_synthetic_sensor_data(num_agents=num_agents)

    # Initialize Global Model with HierarchicalResidualLinear (num_classes=1 for regression)
    global_model = HierarchicalResidualLinear(in_features=input_dim, num_classes=1, bias=False)

    # Storage for cluster residuals (2 clusters) and local residuals (10 agents)
    cluster_residuals = [torch.zeros(1, input_dim) for _ in range(2)]
    local_residuals = [torch.zeros(1, input_dim) for _ in range(num_agents)]

    num_rounds = 15
    lr = 0.05
    mu_local = 1e-3
    mu_cluster = 1e-4

    print(f"Federated Setup: {num_agents} Agents, 2 Clusters, {num_rounds} Communication Rounds")
    print("Running Federated Training with Hierarchical Residual Parameterization...\n")

    for r in range(1, num_rounds + 1):
        deltas_global = []
        deltas_cluster = {0: [], 1: []}

        for i in range(num_agents):
            cluster_id = agent_data[i]["cluster_id"]
            x_train = agent_data[i]["train_x"]
            y_train = agent_data[i]["train_y"]

            # Load model state with assigned cluster residual and private local residual
            agent_model = HierarchicalResidualLinear(in_features=input_dim, num_classes=1, bias=False)
            agent_model.load_state_dict(global_model.state_dict())
            agent_model.weight_cluster.data.copy_(cluster_residuals[cluster_id])
            agent_model.weight_local.data.copy_(local_residuals[i])

            optimizer = torch.optim.SGD(agent_model.parameters(), lr=lr)

            # Local training passes
            for _ in range(5):
                optimizer.zero_grad()
                pred = agent_model(x_train)
                mse_loss = F.mse_loss(pred, y_train)
                # Bayesian shrinkage on residuals
                shrinkage = (
                    0.5 * mu_local * torch.sum(agent_model.weight_local ** 2)
                    + 0.5 * mu_cluster * torch.sum(agent_model.weight_cluster ** 2)
                )
                total_loss = mse_loss + shrinkage
                total_loss.backward()
                optimizer.step()

            # Save local residual strictly on device
            local_residuals[i].copy_(agent_model.weight_local.data)

            # Record deltas for global and cluster aggregation
            delta_g = agent_model.weight_global.data - global_model.weight_global.data
            delta_c = agent_model.weight_cluster.data - cluster_residuals[cluster_id]

            deltas_global.append(delta_g)
            deltas_cluster[cluster_id].append(delta_c)

        # 1. Global Aggregation: Server averages global head deltas
        avg_delta_g = torch.stack(deltas_global).mean(dim=0)
        global_model.weight_global.data.add_(avg_delta_g)

        # 2. Cluster Aggregation: Cluster heads aggregate coalition residuals
        for c_id in range(2):
            if deltas_cluster[c_id]:
                avg_delta_c = torch.stack(deltas_cluster[c_id]).mean(dim=0)
                cluster_residuals[c_id].add_(avg_delta_c)

    # --- Evaluation across All 10 Agents ---
    r2_scores = []
    mse_scores = []

    for i in range(num_agents):
        cluster_id = agent_data[i]["cluster_id"]
        x_test = agent_data[i]["test_x"]
        y_test = agent_data[i]["test_y"]

        eval_model = HierarchicalResidualLinear(in_features=input_dim, num_classes=1, bias=False)
        eval_model.load_state_dict(global_model.state_dict())
        eval_model.weight_cluster.data.copy_(cluster_residuals[cluster_id])
        eval_model.weight_local.data.copy_(local_residuals[i])

        with torch.no_grad():
            preds = eval_model(x_test)
            mse = F.mse_loss(preds, y_test).item()
            ss_tot = torch.sum((y_test - y_test.mean()) ** 2).item()
            ss_res = torch.sum((y_test - preds) ** 2).item()
            r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

        r2_scores.append(r2)
        mse_scores.append(mse)

    mean_r2 = float(np.mean(r2_scores))
    min_r2 = float(np.min(r2_scores))
    mean_mse = float(np.mean(mse_scores))

    print("-" * 75)
    print(f"Results across {num_agents} Heterogeneous Sensor Agents:")
    print(f"  * Mean R^2 Score:         {mean_r2 * 100:.2f}% (Average Agent Prediction Quality)")
    print(f"  * Worst-Case (Min) R^2:   {min_r2 * 100:.2f}% (Rawlsian Egalitarian Welfare)")
    print(f"  * Mean Test MSE:          {mean_mse:.5f}")
    results = {"mean_r2": round(mean_r2 * 100, 2), "min_r2": round(min_r2 * 100, 2), "mean_mse": round(mean_mse, 5)}
    out_path = os.path.join(_project_root, "outputs", "regression_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Regression results saved to: {out_path}")
    return results


if __name__ == "__main__":
    run_regression_experiment()
