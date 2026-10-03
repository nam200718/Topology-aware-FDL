import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.topologies.star import StarTopology
from src.topologies.hierarchical import HierarchicalTopology
from src.core.centralized_engine import CentralizedEngine
from src.core.hierarchical_ensemble_engine import HierarchicalEnsembleEngine
from src.config import ExperimentConfig
from src.core.aggregator import FedAvgAggregator

from temp.ring import RingTopology
from temp.gossip import GossipTopology
from temp.decentralized_engine import DecentralizedEngine
from temp.attacks import check_inf_nan

TOPOLOGY_REGISTRY = {
    "star":      (StarTopology,         CentralizedEngine),
    "ring":      (RingTopology,         DecentralizedEngine),
    "gossip":    (GossipTopology,       DecentralizedEngine),
    "hierarchical":  (HierarchicalTopology, HierarchicalEnsembleEngine),
    "hierarchical_ensemble":  (HierarchicalTopology, HierarchicalEnsembleEngine),
}

class DefendedCentralizedEngine(CentralizedEngine):
    """Centralized Engine with reference_weights passed to defense aggregator."""
    def __init__(self, config, topology, aggregator, device="cpu", defense_config=None):
        super().__init__(config, topology, aggregator, device)
        self.defense_config = defense_config
        if defense_config is not None:
            from src.defense.aggregator import SoftRejectionAggregator
            self.aggregator = SoftRejectionAggregator(defense_config)

    def run_round(self, round_num: int):
        target_clients = self.topology.get_server_connected_clients()
        pre_round_server_weights = self.server_weights.clone()

        # Download
        for client_id in target_clients:
            self.clients_state[client_id].weights = self.server_weights.clone()

        # Update
        current_lr = self.get_current_lr(round_num)
        updated_states = []
        for client_id in target_clients:
            state = self.clients_state[client_id].copy()
            client_ds = self.client_train_datasets[client_id]

            updated_state = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng,
                current_lr=current_lr,
            )
            self.clients_state[client_id] = updated_state
            updated_states.append(updated_state)

        # INF/NAN Sentinel
        for client_id in target_clients:
            state = self.clients_state[client_id]
            if not check_inf_nan(state.weights, pre_round_server_weights):
                state.is_confirmed_malicious = True

        # Aggregate with reference_weights
        if updated_states:
            if hasattr(self.aggregator, 'aggregate'):
                import inspect
                sig = inspect.signature(self.aggregator.aggregate)
                if 'reference_weights' in sig.parameters:
                    self.server_weights = self.aggregator.aggregate(
                        updated_states, reference_weights=pre_round_server_weights
                    )
                else:
                    self.server_weights = self.aggregator.aggregate(updated_states)
            else:
                self.server_weights = self.aggregator.aggregate(updated_states)

        if hasattr(self.aggregator, 'decay_temperature'):
            self.aggregator.decay_temperature()

        # Metrics
        acc, test_loss = self.evaluate_model(self.server_weights)

        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": len(target_clients),
            "total_clients_targeted": len(target_clients),
        }
        if hasattr(self.aggregator, 'current_temperature'):
            round_data["defense_temperature"] = self.aggregator.current_temperature

        has_pers = getattr(self.config.clients, "use_ensemble", False) or (
            getattr(self.config.clients, "personalization_method", "none") in ("ditto", "apfl")
        )
        if has_pers:
            ens_acc, ens_loss = self.evaluate_ensemble()
        else:
            ens_acc, ens_loss = self._evaluate_global_on_client_partitions()

        round_data["ensemble_test_accuracy"] = ens_acc
        round_data["ensemble_test_loss"] = ens_loss

        self.metrics.log_round(round_data)


class DefendedDecentralizedEngine(DecentralizedEngine):
    """Decentralized Engine with reference_weights passed to defense aggregator."""
    def __init__(self, config, topology, aggregator, device="cpu", defense_config=None):
        super().__init__(config, topology, aggregator, device)
        self.defense_config = defense_config
        if defense_config is not None:
            from src.defense.aggregator import SoftRejectionAggregator
            self.aggregator = SoftRejectionAggregator(defense_config)

    def _evaluate_weights_on_client_partitions(self, weights):
        from src.core.model import vector_to_model
        from src.data.dataset import get_fast_dataloader
        model = self.updater.global_model
        vector_to_model(weights.to(self.device), model)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        with torch.no_grad():
            for client_id in range(self.config.clients.num_clients):
                client_test_ds = self.client_test_datasets[client_id]
                if len(client_test_ds) == 0:
                    continue
                loader = get_fast_dataloader(client_test_ds, batch_size=min(len(client_test_ds), 1024), shuffle=False)
                for images, labels in loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    total_loss += criterion(outputs, labels).item()
                    total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    total_samples += labels.size(0)
        if total_samples == 0:
            return 0.0, 0.0
        return (100.0 * total_correct / total_samples), (total_loss / total_samples)

    def run_round(self, round_num: int):
        all_clients = list(range(self.config.clients.num_clients))
        pre_round_weights = {cid: self.clients_state[cid].weights.clone() for cid in all_clients}

        # 1. Exchange: client aggregate neighbors' weights with reference to self
        buffers = {}
        for cid in all_clients:
            neighbors = self.topology.get_neighbors(cid)
            neighbor_states = [self.clients_state[n] for n in neighbors]
            neighbor_states.append(self.clients_state[cid])  # include self
            if hasattr(self.aggregator, 'aggregate'):
                import inspect
                sig = inspect.signature(self.aggregator.aggregate)
                if 'reference_weights' in sig.parameters:
                    buffers[cid] = self.aggregator.aggregate(
                        neighbor_states, reference_weights=pre_round_weights[cid]
                    )
                else:
                    buffers[cid] = self.aggregator.aggregate(neighbor_states)
            else:
                buffers[cid] = self.aggregator.aggregate(neighbor_states)

        if hasattr(self.aggregator, 'decay_temperature'):
            self.aggregator.decay_temperature()

        # 2. Local Update on aggregated weights
        current_lr = self.get_current_lr(round_num)
        for cid in all_clients:
            state = self.clients_state[cid].copy()
            state.weights = buffers[cid]

            client_ds = self.client_train_datasets[cid]
            updated = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng,
                current_lr=current_lr,
            )
            self.clients_state[cid] = updated

        # 3. Global metrics = avg all clients
        avg_weights = torch.mean(
            torch.stack([self.clients_state[c].weights for c in all_clients]), dim=0
        )
        acc, test_loss = self.evaluate_model(avg_weights)

        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "participating_clients": len(all_clients),
            "total_clients_targeted": len(all_clients),
        }
        if hasattr(self.aggregator, 'current_temperature'):
            round_data["defense_temperature"] = self.aggregator.current_temperature

        ens_acc, ens_loss = self._evaluate_weights_on_client_partitions(avg_weights)
        round_data["ensemble_test_accuracy"] = ens_acc
        round_data["ensemble_test_loss"] = ens_loss

        self.metrics.log_round(round_data)


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_single_topology(topo_config: dict, common_config_dict: dict) -> dict:
    topo_type = topo_config["type"]
    label = topo_config["label"]
    print(f"\n{'='*50}\nRunning Topology: {label}\n{'='*50}")

    # Build MainConfig
    config_dict = dict(common_config_dict)
    config_dict["topologies"] = [topo_config]
    config_dict["experiment_type"] = "single"
    
    # We must construct a MainConfig from the dict
    temp_path = "temp_run_config.yaml"
    with open(temp_path, "w") as f:
        yaml.dump(config_dict, f)
        
    exp_config = ExperimentConfig.from_yaml(temp_path)
    main_config = exp_config.build_configs()[0]["config"]
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Instantiate
    TopoClass, EngineClass = TOPOLOGY_REGISTRY[topo_type]
    topo_params = topo_config.get("params", {})
    
    # Configure ensemble and personalization flags based on topology type
    if topo_type == "hierarchical_ensemble":
        main_config.clients.hierarchical_ensemble = True
        main_config.clients.compute_optimization_mode = "shared_backbone"
        main_config.clients.personalization_method = "none"
    else:
        main_config.clients.hierarchical_ensemble = False
        main_config.clients.use_ensemble = False
        main_config.clients.compute_optimization_mode = "none"
        main_config.clients.personalization_method = topo_params.get("personalization_method", "none")
        if "ditto_lambda" in topo_params:
            main_config.clients.ditto_lambda = topo_params["ditto_lambda"]
    
    # Init Topology
    if topo_type == "gossip":
        topology = TopoClass(degree_k=topo_params.get("degree_k", 3))
    else:
        topology = TopoClass()
        
    # Init Engine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    defense_mode = topo_params.get("defense_mode", "none")
    if defense_mode != "none":
        from src.defense.config import DefenseConfig
        defense_cfg = DefenseConfig(
            defense_mode=defense_mode,
            temperature=topo_params.get("temperature", 1.0),
            temperature_decay=topo_params.get("temperature_decay", 0.95),
            temperature_min=topo_params.get("temperature_min", 0.1),
            defense_scope=topo_params.get("defense_scope", "cluster"),
            norm_threshold=topo_params.get("norm_threshold", 2.0),
            norm_bounding_enabled=topo_params.get("norm_bounding_enabled", True),
            norm_bounding_multiplier=topo_params.get("norm_bounding_multiplier", 2.0),
            hard_rejection_enabled=topo_params.get("hard_rejection_enabled", True),
            hard_rejection_threshold=topo_params.get("hard_rejection_threshold", 0.0),
            adaptive_temperature=topo_params.get("adaptive_temperature", True),
        )
        if topo_type in ("hierarchical_ensemble", "hierarchical"):
            from src.defense.engine import DefendedEnsembleEngine
            engine = DefendedEnsembleEngine(
                main_config, topology, FedAvgAggregator(), device, defense_config=defense_cfg
            )
        elif topo_type == "star":
            engine = DefendedCentralizedEngine(
                main_config, topology, None, device, defense_config=defense_cfg
            )
        elif topo_type in ("ring", "gossip"):
            engine = DefendedDecentralizedEngine(
                main_config, topology, None, device, defense_config=defense_cfg
            )
        else:
            from src.defense.aggregator import SoftRejectionAggregator
            aggregator = SoftRejectionAggregator(defense_cfg)
            engine = EngineClass(main_config, topology, aggregator, device)
    else:
        aggregator = FedAvgAggregator()
        engine = EngineClass(main_config, topology, aggregator, device)

    # Set local_batch_size for GPU acceleration
    cfg_batch_size = topo_params.get("local_batch_size", topo_params.get("batch_size", common_config_dict.get("clients", {}).get("local_batch_size", 128)))
    main_config.clients.local_batch_size = cfg_batch_size

    # Hierarchical Partition Injection
    if common_config_dict.get("use_hierarchical_partition", False):
        from temp.hierarchical_partitioner import partition_data_hierarchical
        hp_config = common_config_dict.get("hierarchical_partition_config", {})
        num_clusters = hp_config.get("num_clusters", 5)
        intra_alpha = hp_config.get("intra_alpha", 1.0)
        inter_alpha = hp_config.get("inter_alpha", 0.1)
        
        engine.client_indices, _ = partition_data_hierarchical(
            dataset=engine.train_dataset,
            num_clients=main_config.clients.num_clients,
            num_clusters=num_clusters,
            intra_alpha=intra_alpha,
            inter_alpha=inter_alpha,
            seed=main_config.env.seed,
        )
        engine.client_test_indices, _ = partition_data_hierarchical(
            dataset=engine.test_dataset,
            num_clients=main_config.clients.num_clients,
            num_clusters=num_clusters,
            intra_alpha=intra_alpha,
            inter_alpha=inter_alpha,
            seed=main_config.env.seed + 1,
        )
        from src.data.dataset import ClientDataset
        engine.client_train_datasets = {
            cid: ClientDataset(engine.train_dataset, idxs)
            for cid, idxs in engine.client_indices.items()
        }
        engine.client_test_datasets = {
            cid: ClientDataset(engine.test_dataset, idxs)
            for cid, idxs in engine.client_test_indices.items()
        }
        engine.client_test_indices_t = {
            cid: torch.tensor(idxs, dtype=torch.long, device=engine.device)
            for cid, idxs in engine.client_test_indices.items()
        }
        print(f"  [Hierarchical Partition] clusters={num_clusters}, x={intra_alpha}, β={inter_alpha}")
    
    # Override client config if needed (hierarchical)
    if topo_type == "hierarchical":
        main_config.clients.use_ensemble = False
        main_config.clients.hierarchical_ensemble = False
        main_config.clients.compute_optimization_mode = "none"

    engine.run()
    
    # Extract metrics
    metrics = engine.metrics.history
    final_global_acc = metrics[-1].get("test_accuracy", 0.0) if metrics else 0.0
    final_ens_acc = metrics[-1].get("ensemble_test_accuracy", final_global_acc) if metrics else 0.0
    
    return {
        "label": label,
        "global_acc": final_global_acc,
        "ens_acc": final_ens_acc,
        "metrics": metrics
    }

def run_all(config_path: str):
    config = load_yaml_config(config_path)
    topologies = config.pop("topologies")
    byzantine_rates = config.pop("byzantine_rates", [0.0])
    default_byz_type = config.get("robustness", {}).get("byzantine_type", "none")
    byzantine_types = config.pop("byzantine_types", [default_byz_type])
    
    out_dir = config["env"].get("output_dir", "./outputs/baseline_comparison")
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    all_metrics = {}
    
    for byz_type in byzantine_types:
        for byz_rate in byzantine_rates:
            # Skip duplicate baseline runs (rate=0.0) for subsequent attack types
            if byz_rate == 0.0 and byz_type != byzantine_types[0]:
                continue
                
            if "robustness" not in config:
                config["robustness"] = {}
            config["robustness"]["byzantine_rate"] = byz_rate
            config["robustness"]["byzantine_type"] = byz_type
            
            for topo in topologies:
                res = run_single_topology(topo, config)
                
                # Format label dynamically
                suffix = ""
                if byz_rate > 0:
                    suffix = f" ({byz_type} {byz_rate})"
                elif len(byzantine_rates) > 1 or len(byzantine_types) > 1:
                    suffix = " (No Attack)"
                    
                label = f"{res['label']}{suffix}"
                
                results.append({
                    "Method": label,
                    "Byz Type": byz_type,
                    "Byz Rate": byz_rate,
                    "Global Acc": res["global_acc"],
                    "Ens Acc": res["ens_acc"]
                })
                all_metrics[label] = res["metrics"]
                
                # Save individual metrics as CSV (no JSON)
                topo_safe_name = label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("=", "").replace(".", "_")
                pd.DataFrame(res["metrics"]).to_csv(os.path.join(out_dir, f"{topo_safe_name}_metrics.csv"), index=False)
            
    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
    print("\nSummary:\n", df)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Subplot 1: Global Test Accuracy
    for label, metrics in all_metrics.items():
        rounds = [m["round"] for m in metrics]
        accs = [m.get("test_accuracy", 0) for m in metrics]
        ax1.plot(rounds, accs, label=label, marker='o', markersize=3)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Global Model Accuracy")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Ensemble/Personalized Accuracy  
    for label, metrics in all_metrics.items():
        ens_accs = [m.get("ensemble_test_accuracy", None) for m in metrics]
        if any(a is not None and a > 0 for a in ens_accs):
            rounds = [m["round"] for m in metrics]
            accs = [a if a else 0 for a in ens_accs]
            ax2.plot(rounds, accs, label=label, marker='s', markersize=3)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Ensemble / Personalized Accuracy")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_accuracy_curves.png"), dpi=150)
    plt.close()
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="temp/baseline_comparison.yaml")
    args = parser.parse_args()
    
    run_all(args.config)

