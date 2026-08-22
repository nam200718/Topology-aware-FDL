"""
Direct Engine Verification Script for the Upgraded Hierarchical Ensemble Engine (HEP).
Runs the full HierarchicalEnsembleEngine with:
- Strict IID Gating (w_root=1.0 on IID)
- Active Class Logit Masking (ACLM)
- Temperature Sharpened Dynamic Confidence Inference
- Adaptive Update Similarity Clustering
"""

import os
import sys
import json
import time
import numpy as np
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config import SimulationConfig, ClientConfig, TopologyConfig, EnvironmentConfig, RobustnessConfig
from src.core.aggregator import FedAvgAggregator
from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine
from src.experiments.builder import TopologyEngineFactory, detect_device


def run_hep_regime(dataset="cifar10", non_iid=False, alpha=0.5, num_rounds=20):
    device = detect_device()
    
    env_cfg = EnvironmentConfig(
        seed=42,
        output_dir="./outputs",
        dataset=dataset,
        train_subset=10000,
        test_subset=3000,
        non_iid=non_iid,
        alpha=alpha,
    )
    client_cfg = ClientConfig(
        model_name="resnet9",
        num_clients=15,
        local_lr=0.05,
        local_steps=3,
        total_local_steps=10,
        use_ensemble=True,
        hierarchical_ensemble=True,
        compute_optimization_mode="shared_backbone",
        ensemble_weighting_mode="dynamic_confidence",
        ensemble_distillation=False,
        distillation_lambda=0.0,
        loss_weight_beta=1.0,
    )

    topo_cfg = TopologyConfig(
        type="hierarchical_ensemble",
        params={
            "num_clusters": 3,
            "cluster_method": "update_similarity",
            "warmup_min_rounds": 2,
            "warmup_max_rounds": 10,
            "stability_threshold": 0.30,
            "misalignment_threshold": 0.15,
            "personalization_method": "none",
        }
    )
    sim_cfg = SimulationConfig(
        env=env_cfg,
        clients=client_cfg,
        topology=topo_cfg,
        robustness=RobustnessConfig(),
        num_rounds=num_rounds,
    )

    topology, engine_cls = TopologyEngineFactory.build(sim_cfg)
    aggregator = FedAvgAggregator()
    engine = engine_cls(
        config=sim_cfg,
        topology=topology,
        aggregator=aggregator,
        device=device,
    )


    t0 = time.time()
    for r in range(1, num_rounds + 1):
        engine.run_round(r)
    elapsed = round(time.time() - t0, 2)

    history = engine.metrics.get_history()
    last_round = history[-1] if history else {}
    ens_acc = last_round.get("ensemble_test_accuracy", 0.0)

    # Compute client-level test accuracies for Bottom 10%
    multi_model = engine.updater.multihead_model
    multi_model.eval()
    client_accs = []
    criterion = torch.nn.CrossEntropyLoss()

    for cid in range(client_cfg.num_clients):
        state = engine.clients_state[cid]
        c_test = engine.client_test_datasets[cid]
        if len(c_test) == 0:
            continue
        from src.data.dataset import get_fast_dataloader
        from src.core.model import vector_to_model
        vector_to_model(state.weights.to(device), multi_model)
        if state.parent_head_state is not None:
            multi_model.fc2_parent.load_state_dict(state.parent_head_state)
        if state.local_head_state is not None:
            multi_model.fc2_local.load_state_dict(state.local_head_state)

        loader = get_fast_dataloader(c_test, batch_size=len(c_test), shuffle=False)
        corr, tot = 0, 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                zr, zp, zl = multi_model(imgs, head="all")
                r_sk = getattr(state, "r_skew", 0.5)

                if r_sk >= 0.85:
                    z_blend = zr
                else:
                    pr = torch.softmax(zr, dim=1)
                    pp = torch.softmax(zp, dim=1)
                    pl = torch.softmax(zl, dim=1)

                    hr = -(pr * torch.log(pr + 1e-8)).sum(dim=1).mean().item()
                    hp = -(pp * torch.log(pp + 1e-8)).sum(dim=1).mean().item()
                    hl = -(pl * torch.log(pl + 1e-8)).sum(dim=1).mean().item()

                    score_l = -hl
                    score_p = -hp
                    score_r = -hr

                    alpha_vec = getattr(state, "ensemble_alpha", None)
                    if alpha_vec is not None and len(alpha_vec) == 3:
                        score_l += np.log(max(1e-4, alpha_vec[0]))
                        score_p += np.log(max(1e-4, alpha_vec[1]))
                        score_r += np.log(max(1e-4, alpha_vec[2]))

                    scores = torch.tensor([score_l, score_p, score_r], dtype=torch.float32, device=device)
                    weights = torch.softmax(scores / 1.0, dim=0)
                    wl, wp, wr = weights[0].item(), weights[1].item(), weights[2].item()
                    z_blend = wl * (zl / 0.6) + wp * (zp / 0.8) + wr * (zr / 1.0)

                pred = z_blend.argmax(dim=1)
                corr += (pred == lbls).sum().item()
                tot += lbls.size(0)
        c_acc = (corr / tot * 100.0) if tot > 0 else 0.0
        client_accs.append(c_acc)

    mean_acc = round(float(np.mean(client_accs)), 2)
    bot10 = round(float(np.percentile(client_accs, 10)), 2)
    return mean_acc, bot10, elapsed


def main():
    print("Starting Comprehensive Upgraded HEP Engine Evaluation...")
    results = {}
    
    scenarios = [
        ("IID (inf)", False, 0.5),
        ("Mild (1.0)", True, 1.0),
        ("Moderate (0.5)", True, 0.5),
        ("Severe (0.1)", True, 0.1),
        ("Extreme (0.05)", True, 0.05),
    ]

    for label, non_iid, alpha in scenarios:
        print(f"\n--- Running CIFAR-10 {label} ---")
        mean_acc, bot10, elapsed = run_hep_regime(dataset="cifar10", non_iid=non_iid, alpha=alpha, num_rounds=20)
        results[label] = {
            "mean_acc": mean_acc,
            "bottom_10": bot10,
            "time_s": elapsed
        }
        print(f"[{label}] Result: Mean Acc = {mean_acc}%, Bottom-10% = {bot10}%, Time = {elapsed}s")

    print("\n--- Running CIFAR-100 Extreme (0.05) ---")
    mean100, bot10_100, el100 = run_hep_regime(dataset="cifar100", non_iid=True, alpha=0.05, num_rounds=20)
    results["CIFAR-100 Extreme (0.05)"] = {
        "mean_acc": mean100,
        "bottom_10": bot10_100,
        "time_s": el100
    }
    print(f"[CIFAR-100 Extreme (0.05)] Result: Mean Acc = {mean100}%, Bottom-10% = {bot10_100}%, Time = {el100}s")

    out_path = os.path.join(_project_root, "outputs", "new_upgraded_hep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to: {out_path}")


if __name__ == "__main__":
    main()
