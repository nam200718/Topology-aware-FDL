"""SCAFFOLD Engine: CentralizedEngine subclass managing server-side control variates.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import torch

from src.core.centralized_engine import CentralizedEngine
from src.baselines.scaffold_updater import ScaffoldUpdater
from src.core.interfaces import ClientState


class ScaffoldEngine(CentralizedEngine):
    """Orchestrates SCAFFOLD federated communication and control variate tracking."""

    def __init__(self, config, topology, aggregator, device="cpu"):
        super().__init__(config, topology, aggregator, device)
        model_name = getattr(config.clients, "model_name", "simple_cnn")

        # Replace updater with ScaffoldUpdater
        self.updater = ScaffoldUpdater(
            device=self.device,
            in_channels=self.in_channels,
            model_name=model_name,
            num_classes=self.num_classes,
        )

        num_params = sum(p.numel() for p in self.updater.global_model.parameters())
        self.server_control_variate = torch.zeros(num_params, device=self.device)

    def run_round(self, round_num: int):
        target_clients = self.topology.get_server_connected_clients()

        # Broadcast current global model to clients
        for client_id in target_clients:
            self.clients_state[client_id].weights = self.server_weights.clone()

        current_lr = self.get_current_lr(round_num)
        updated_states: List[ClientState] = []

        for client_id in target_clients:
            state = self.clients_state[client_id].copy()
            # Explicitly propagate control variate across round copies
            state.control_variate = getattr(self.clients_state[client_id], "control_variate", None)
            client_ds = self.client_train_datasets[client_id]

            updated_state = self.updater.update_scaffold(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng,
                current_lr=current_lr,
                server_control_variate=self.server_control_variate,
            )
            self.clients_state[client_id] = updated_state
            updated_states.append(updated_state)

        # Server-side model aggregation
        if updated_states:
            expected = len(self.server_weights)
            valid = [s for s in updated_states if len(s.weights) == expected]
            if valid:
                self.server_weights = self.aggregator.aggregate(valid)

        # Update server control variate c = c + (1 / |S|) * sum(delta_c_i)
        deltas = []
        for s in updated_states:
            dc = getattr(s, "_scaffold_delta_c", None)
            if dc is not None:
                deltas.append(dc.to(self.device))

        if deltas:
            delta_c_avg = torch.stack(deltas).mean(dim=0)
            self.server_control_variate.add_(delta_c_avg)

        # Evaluation & metrics
        if not self.should_evaluate(round_num):
            self.metrics.log_round({
                "round": round_num,
                "participating_clients": len(target_clients),
                "total_clients_targeted": len(target_clients),
                "evaluated": False,
            })
            return

        acc, test_loss = self.evaluate_model(self.server_weights)
        ens_acc, ens_loss = self._evaluate_global_on_client_partitions()

        round_data = {
            "round": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "ensemble_test_accuracy": ens_acc,
            "ensemble_test_loss": ens_loss,
            "participating_clients": len(target_clients),
            "total_clients_targeted": len(target_clients),
            "evaluated": True,
        }

        # Fairness metric at final round
        if round_num == self.config.num_rounds and getattr(self, "_last_per_client_accuracy", None):
            accs = sorted(self._last_per_client_accuracy.values())
            k = max(1, int(np.ceil(0.1 * len(accs))))
            round_data["bottom10_fairness"] = float(np.mean(accs[:k]))
            round_data["per_client_accuracy"] = {
                str(cid): round(a, 4) for cid, a in sorted(self._last_per_client_accuracy.items())
            }

        self.metrics.log_round(round_data)
