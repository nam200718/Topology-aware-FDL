import os
import sys
import torch
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config import SimulationConfig, TopologyConfig, ClientConfig, EnvironmentConfig, NonIIDConfig, RobustnessConfig
from src.experiments.builder import TopologyEngineFactory, detect_device
from src.core.aggregator import FedAvgAggregator
from src.utils.random import set_seed
from src.core.loss import compute_dynamic_binomial_loss_weights


def run_regime(name, is_non_iid, alpha):
    print(f"\n--- Running Pilot Regime: {name} (non_iid={is_non_iid}, alpha={alpha}) ---")
    set_seed(42)
    device = detect_device()

    sim_config = SimulationConfig(
        experiment_name=f"pilot_{name}",
        num_rounds=3,
        eval_interval=1,
        env=EnvironmentConfig(
            seed=42,
            output_dir="./outputs/pilot_gate",
            dataset="cifar10",
            train_subset=10000,
            test_subset=3000,
        ),
        topology=TopologyConfig(
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
        ),
        clients=ClientConfig(
            model_name="resnet9",
            num_clients=15,
            local_lr=0.05,
            local_steps=3,
            total_local_steps=5,
            use_ensemble=True,
            hierarchical_ensemble=True,
            compute_optimization_mode="shared_backbone",
            ensemble_weighting_mode="dynamic_confidence",
            ensemble_distillation=False,
            distillation_lambda=0.0,
            head_training_schedule="binomial",
            dynamic_parameters=True,
        ),
        robustness=RobustnessConfig(),
        non_iid=NonIIDConfig(enabled=is_non_iid, alpha=alpha if alpha is not None else 0.5),
    )

    topology, engine_cls = TopologyEngineFactory.build(sim_config)
    aggregator = FedAvgAggregator()
    engine = engine_cls(
        config=sim_config,
        topology=topology,
        aggregator=aggregator,
        device=device,
    )

    engine.run()

    r_skews = []
    active_class_counts = []
    binomial_weights = []

    for cid in range(sim_config.clients.num_clients):
        c_state = engine.clients_state[cid]
        c_dataset = engine.client_train_datasets[cid]
        r_skew, active_mask, class_counts = engine.updater._compute_label_stats(c_dataset, 10)
        num_active = int(active_mask.sum().item())
        lr, lp, ll, ar, ap, al = compute_dynamic_binomial_loss_weights(
            r_skew=r_skew, num_classes=10, local_classes=num_active, num_clusters=3
        )
        r_skews.append(r_skew)
        active_class_counts.append(num_active)
        binomial_weights.append((ar, ap, al))

    print(f"Regime {name}:")
    print(f"  R_skew: mean={np.mean(r_skews):.4f}, min={np.min(r_skews):.4f}, max={np.max(r_skews):.4f}")
    print(f"  Active classes: mean={np.mean(active_class_counts):.2f}, min={np.min(active_class_counts)}, max={np.max(active_class_counts)}")
    print(f"  Mean (alpha_r, alpha_p, alpha_l): ({np.mean([w[0] for w in binomial_weights]):.4f}, {np.mean([w[1] for w in binomial_weights]):.4f}, {np.mean([w[2] for w in binomial_weights]):.4f})")
    
    last_metric = engine.metrics.get_history()[-1]
    test_acc = last_metric.get("ensemble_test_accuracy", last_metric.get("test_accuracy", 0.0))
    print(f"  Final Round Test Accuracy: {test_acc:.4f}")

    return {
        "r_skews": r_skews,
        "active_counts": active_class_counts,
        "binomial_weights": binomial_weights,
        "test_acc": test_acc
    }


def main():
    iid_res = run_regime("iid", False, None)
    mod_res = run_regime("moderate_0.5", True, 0.5)
    ext_res = run_regime("extreme_0.05", True, 0.05)

    print("\n================ PILOT GATE AUDIT CHECKS ================")
    # 1. R_skew non-degenerate
    iid_r = np.mean(iid_res["r_skews"])
    mod_r = np.mean(mod_res["r_skews"])
    ext_r = np.mean(ext_res["r_skews"])
    print(f"1. R_skew distribution: IID={iid_r:.4f} (expect ~1.0), Mod={mod_r:.4f} (expect ~0.4-0.6), Ext={ext_r:.4f} (expect ~0.1-0.25)")
    assert iid_r > 0.95, f"IID R_skew {iid_r} too low"
    assert 0.3 <= mod_r <= 0.7, f"Moderate R_skew {mod_r} out of expected range"
    assert ext_r < 0.35, f"Extreme R_skew {ext_r} too high"
    print("   -> PASS: R_skew distribution is non-degenerate across regimes.")

    # 2. ACLM masks non-trivial
    ext_active = np.mean(ext_res["active_counts"])
    print(f"2. ACLM active classes under extreme skew: mean={ext_active:.2f}/10 classes (masked classes > 0)")
    assert ext_active < 10.0, "Expected some classes masked under extreme skew"
    print("   -> PASS: ACLM masks active classes correctly.")

    # 3. Binomial weights differ per regime (lp peaks moderate)
    iid_lp = np.mean([w[1] for w in iid_res["binomial_weights"]])
    mod_lp = np.mean([w[1] for w in mod_res["binomial_weights"]])
    ext_lp = np.mean([w[1] for w in ext_res["binomial_weights"]])
    print(f"3. Binomial lambda_p (parent head weight): IID={iid_lp:.4f}, Mod={mod_lp:.4f}, Ext={ext_lp:.4f}")
    assert mod_lp > iid_lp and mod_lp > ext_lp, f"Expected lambda_p to peak at moderate skew: mod={mod_lp}, iid={iid_lp}, ext={ext_lp}"
    print("   -> PASS: Binomial parent weight lambda_p peaks in moderate skew regime.")

    print("\n PILOT GATE OUTCOME: GO -> Ready for Tier A queue.")


if __name__ == "__main__":
    main()
