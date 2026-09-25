"""FedProx (Li et al., MLSys 2020): Federated Optimization in Heterogeneous Networks.

Extends PyTorchLocalUpdater with proximal L2 regularization anchored to the
received broadcast model parameters:
    min_w F_k(w) + (mu / 2) * ||w - w^t||^2
"""
import torch
from src.core.updater import PyTorchLocalUpdater
from src.core.model import model_to_vector, vector_to_model


class FedProxUpdater(PyTorchLocalUpdater):
    """Local updater adding FedProx proximal regularization."""

    def update(self, state, client_dataset, config, rng, current_lr=None):
        method = getattr(config, "personalization_method", "none")
        if method == "fedprox":
            initial_weights = state.weights.clone()
            cfg_model_name = getattr(config, "model_name", "simple_cnn")
            if cfg_model_name != self.model_name:
                self._init_models(cfg_model_name, self.in_channels)

            local_lr = current_lr if current_lr is not None else config.local_lr
            if len(client_dataset) == 0:
                return state

            loader = self._make_loader(client_dataset, config)
            epochs = config.local_steps
            return self._update_fedprox(state, config, loader, epochs, local_lr, initial_weights)

        # Fallback to parent implementation for other methods
        return super().update(state, client_dataset, config, rng, current_lr)

    def _update_fedprox(self, state, config, loader, epochs, local_lr, initial_weights):
        model = self.global_model
        vector_to_model(state.weights.to(self.device), model)

        # The received server model is the proximal anchor
        w_anchor = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=local_lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-4,
            foreach=False,
        )

        mu = float(getattr(config, "fedprox_mu", 0.01))
        num_classes = self.num_classes
        is_byz = getattr(state, "is_byzantine", False)
        byz_type = getattr(state, "byzantine_type", "label_flip")

        for epoch in range(epochs):
            for images, labels in loader:
                if images.device != self.device:
                    images, labels = images.to(self.device), labels.to(self.device)

                if is_byz and byz_type == "label_flip":
                    labels = self._flip_labels(labels, num_classes)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss_ce = self.criterion(logits, labels)

                # Proximal term: (mu / 2) * ||w - w_anchor||^2
                w_curr = torch.nn.utils.parameters_to_vector(model.parameters())
                loss_prox = 0.5 * mu * torch.sum((w_curr - w_anchor) ** 2)

                loss = loss_ce + loss_prox
                loss.backward()
                optimizer.step()

        state.weights = model_to_vector(model).detach()
        return self._apply_byzantine_attack(state, initial_weights)
