"""SCAFFOLD (Karimireddy et al., ICML 2020): Stochastic Controlled Averaging for Federated Learning.

Uses control variates to reduce client drift under Non-IID data distributions.
"""
from typing import Optional, Tuple, List
import torch
from src.core.updater import PyTorchLocalUpdater
from src.core.interfaces import ClientState
from src.core.model import model_to_vector, vector_to_model


class ScaffoldUpdater(PyTorchLocalUpdater):
    """Local updater implementing SCAFFOLD variance-reduced local SGD."""

    def update_scaffold(
        self,
        state: ClientState,
        client_dataset,
        config,
        rng,
        current_lr: float,
        server_control_variate: torch.Tensor,
    ) -> ClientState:
        """Execute SCAFFOLD local training with control variates."""
        initial_weights = state.weights.clone()

        cfg_model_name = getattr(config, "model_name", "simple_cnn")
        if cfg_model_name != self.model_name:
            self._init_models(cfg_model_name, self.in_channels)

        if len(client_dataset) == 0:
            return state

        loader = self._make_loader(client_dataset, config)
        model = self.global_model
        vector_to_model(state.weights.to(self.device), model)

        # Vector of trainable parameters at start of local training
        w_start = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        num_params = w_start.numel()

        # Initialize client control variate c_i if absent
        if getattr(state, "control_variate", None) is None or state.control_variate.numel() != num_params:
            state.control_variate = torch.zeros(num_params, device=self.device)

        c_i = state.control_variate.to(self.device)
        c_server = server_control_variate.to(self.device)

        # Correction term added to local gradient: correction = c_server - c_i
        correction = c_server - c_i

        model.train()
        # SCAFFOLD specifies SGD without momentum to match theoretical convergence
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=current_lr,
            momentum=0.0,
            weight_decay=1e-4,
            foreach=False,
        )

        num_classes = self.num_classes
        is_byz = getattr(state, "is_byzantine", False)
        byz_type = getattr(state, "byzantine_type", "label_flip")
        total_steps = 0

        # Precompute parameter references for zero overhead in mini-batch loop
        model_params = list(model.parameters())
        param_slices = []
        offset = 0
        for p in model_params:
            n = p.numel()
            param_slices.append((p, offset, n, p.shape))
            offset += n

        for epoch in range(config.local_steps):
            for images, labels in loader:
                if images.device != self.device:
                    images, labels = images.to(self.device), labels.to(self.device)

                if is_byz and byz_type == "label_flip":
                    labels = self._flip_labels(labels, num_classes)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = self.criterion(logits, labels)
                loss.backward()

                # Apply SCAFFOLD gradient correction: g_corrected = g_i - c_i + c_server
                for p, off, n, shape in param_slices:
                    if p.grad is not None:
                        p.grad.data.add_(correction[off : off + n].view(shape))

                optimizer.step()
                total_steps += 1

        w_end = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

        # Option II for control variate update (Karimireddy et al., §3):
        # c_i^+ = c_i - c_server + (1 / (K * eta_l)) * (w_start - w_end)
        # delta_c_i = c_i^+ - c_i = -c_server + (1 / (K * eta_l)) * (w_start - w_end)
        effective_step_size = max(1, total_steps) * current_lr
        delta_c = -c_server + (1.0 / effective_step_size) * (w_start - w_end)

        state.control_variate = (c_i + delta_c).detach().cpu()
        state._scaffold_delta_c = delta_c.detach().cpu()
        state.weights = model_to_vector(model).detach()

        return self._apply_byzantine_attack(state, initial_weights)
