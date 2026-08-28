import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ClientConfig
from src.core.interfaces import ClientState
from src.core.model import (
    SimpleCNN,
    MultiHeadSimpleCNN,
    ResNet9,
    MultiHeadResNet9,
    model_to_vector,
    vector_to_model,
)
from src.core.loss import (
    compute_binomial_loss_weights,
    compute_hill_number_r_skew,
    compute_dynamic_binomial_loss_weights,
)
from src.data.dataset import get_fast_dataloader


def _kl_div(log_p, target):
    """DirectML GPU native KL divergence distillation loss -(target * log_p).sum(dim=1).mean()."""
    return -(target * log_p).sum(dim=1).mean()


class PyTorchLocalUpdater:
    def __init__(self, device="cpu", in_channels=1, model_name="simple_cnn", num_classes=10):
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
        self.zero_loss = torch.tensor(0.0, device=self.device)
        self.model_name = model_name
        self.in_channels = in_channels
        self.num_classes = num_classes

        self._init_models(model_name, in_channels)

    def _init_models(self, model_name, in_channels):
        self.model_name = model_name
        self.in_channels = in_channels
        c = self.num_classes
        if model_name == "resnet9":
            self.global_model = ResNet9(in_channels=in_channels, num_classes=c).to(self.device)
            self.local_model = ResNet9(in_channels=in_channels, num_classes=c).to(self.device)
            self.parent_model = ResNet9(in_channels=in_channels, num_classes=c).to(self.device)
            self.multihead_model = MultiHeadResNet9(in_channels=in_channels, num_classes=c).to(self.device)
        elif model_name == "mobilenetv3":
            from src.core.model import MobileNetV3Small, MultiHeadMobileNetV3Small
            self.global_model = MobileNetV3Small(in_channels=in_channels, num_classes=c).to(self.device)
            self.local_model = MobileNetV3Small(in_channels=in_channels, num_classes=c).to(self.device)
            self.parent_model = MobileNetV3Small(in_channels=in_channels, num_classes=c).to(self.device)
            self.multihead_model = MultiHeadMobileNetV3Small(in_channels=in_channels, num_classes=c).to(self.device)
        else:
            self.global_model = SimpleCNN(in_channels=in_channels, num_classes=c).to(self.device)
            self.local_model = SimpleCNN(in_channels=in_channels, num_classes=c).to(self.device)
            self.parent_model = SimpleCNN(in_channels=in_channels, num_classes=c).to(self.device)
            self.multihead_model = MultiHeadSimpleCNN(in_channels=in_channels, num_classes=c).to(self.device)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _apply_byzantine_attack(self, state: ClientState, initial_weights: torch.Tensor) -> ClientState:
        is_byz = getattr(state, "is_byzantine", False)
        if is_byz:
            byz_type = getattr(state, "byzantine_type", "label_flip")
            full_weights = state.weights
            if byz_type == "sign_flip":
                delta = full_weights - initial_weights
                full_weights = initial_weights - delta * 1.5
            elif byz_type == "gradient_ascent":
                delta = full_weights - initial_weights
                full_weights = initial_weights - delta * 5.0
            elif byz_type == "random_noise":
                full_weights = initial_weights + torch.randn_like(initial_weights) * 2.0
            state.weights = full_weights
        return state

    def _make_loader(self, client_dataset, config: ClientConfig):
        batch_size = min(config.local_batch_size, len(client_dataset))
        return get_fast_dataloader(client_dataset, batch_size=batch_size, shuffle=True)

    def _compute_label_stats(self, client_dataset, num_classes: int):
        """Shannon label-entropy ratio R_skew plus active-class mask / counts.

        Returns defaults (r_skew=0.5, all classes active) when label metadata
        is unavailable (e.g. plain TensorDataset).
        """
        r_skew = 0.5
        active_mask = torch.ones(num_classes, dtype=torch.bool, device=self.device)
        class_counts = torch.ones(num_classes, dtype=torch.float32, device=self.device)
        try:
            if hasattr(client_dataset, "labels") and len(client_dataset) > 0:
                l_tensor = client_dataset.labels
                if not isinstance(l_tensor, torch.Tensor):
                    l_tensor = torch.tensor(l_tensor)
                l_tensor = l_tensor.to(self.device)

                # Active-class statistics restricted to valid class indices
                active_mask = torch.zeros(num_classes, dtype=torch.bool, device=self.device)
                class_counts = torch.zeros(num_classes, dtype=torch.float32, device=self.device)
                valid_idx = l_tensor < num_classes
                u_cls, c_counts = torch.unique(l_tensor[valid_idx], return_counts=True)
                active_mask[u_cls.long()] = True
                class_counts[u_cls.long()] = c_counts.float()

                # Parameter-Free Hill-number Order-1 Perplexity Skew Ratio: R_skew = (exp(H) - 1) / (C - 1)
                total = class_counts.sum()
                if total > 0:
                    probs = class_counts / total
                    r_skew = compute_hill_number_r_skew(probs, num_classes)
                else:
                    r_skew = 0.5
        except Exception:
            r_skew = 0.5
            active_mask = torch.ones(num_classes, dtype=torch.bool, device=self.device)
            class_counts = torch.ones(num_classes, dtype=torch.float32, device=self.device)
        return r_skew, active_mask, class_counts

    def _alloc_steps_piecewise(self, r_skew: float, total_budget: int):
        """Legacy piecewise head-epoch budgets on R_skew thresholds.

        Kept selectable via clients.head_training_schedule="piecewise" for the
        paper's ablation tables. Superseded by the binomial partition-of-unity.
        """
        rem = max(0, total_budget - 3)
        if r_skew >= 0.85:  # Uniform/IID regime: complete Root head routing (closes IID generalization gap)
            e_root, e_parent, e_local = total_budget, 0, 0
        elif r_skew > 0.7:  # Mild Non-IID: prioritize Root & Parent heads
            e_root = 1 + round(rem * 0.6)
            e_parent = 1 + round(rem * 0.3)
            e_local = 1 + round(rem * 0.1)
        elif r_skew < 0.3:  # Severe / Extreme Non-IID: prioritize Local & Parent heads
            e_local = 1 + round(rem * 0.6)
            e_parent = 1 + round(rem * 0.3)
            e_root = 1 + round(rem * 0.1)
        else:  # Moderate Non-IID: balanced budget
            e_root = 1 + round(rem * 0.34)
            e_parent = 1 + round(rem * 0.33)
            e_local = 1 + round(rem * 0.33)
        return e_root, e_parent, e_local

    def _alloc_steps_binomial(self, r_skew: float, total_budget: int, anchor_min: float = None, num_classes: int = 10, local_classes: int = 2, num_clusters: int = 3):
        """Continuous dynamic binomial partition-of-unity loss weighting.

        lambda_r = a_i + (1-a_i)R^2, lambda_p = 2R(1-R), lambda_l = (1-R)^2,
        normalized so sum(lambda) == 1 for every R (constant total loss scale);
        a_i is dynamically computed from local class coverage and cluster count.
        """
        if anchor_min is not None and anchor_min > 0:
            _, _, _, alpha_r, alpha_p, alpha_l = compute_binomial_loss_weights(
                r_skew, anchor_min=anchor_min)
        else:
            _, _, _, alpha_r, alpha_p, alpha_l = compute_dynamic_binomial_loss_weights(
                r_skew, num_classes=num_classes, local_classes=local_classes, num_clusters=num_clusters)
        return alpha_r, alpha_p, alpha_l

    def _flip_labels(self, labels, num_classes: int):
        return (num_classes - 1) - labels

    _HEAD_PARAM_PREFIXES = (
        "fc2_root", "fc2_parent", "fc2_local",
        "classifier_root", "classifier_parent", "classifier_local",
    )

    # Single-head model head prefixes (FedRep/FedPer/FedBABU path)
    _SINGLE_HEAD_PREFIXES = ("fc2", "classifier")

    def _freeze_single_head_backbone(self, model: nn.Module, frozen: bool):
        """Freeze/unfreeze everything except the single classification head
        (used by the FedRep/FedBABU stage schedules)."""
        for name, p in model.named_parameters():
            is_head = name.startswith(self._SINGLE_HEAD_PREFIXES)
            p.requires_grad_(is_head or not frozen)

    def _head_param_mask(self, model: nn.Module) -> torch.Tensor:
        """Boolean mask over model_to_vector ordering marking head coordinates.

        Must walk state_dict (params + buffers) because that is what
        model_to_vector serializes; parameter-only iteration shifts every
        offset after the first BatchNorm layer.
        """
        param_keys = {k for k, _ in model.named_parameters() if any(k.startswith(p) for p in self._SINGLE_HEAD_PREFIXES)}
        total = sum(v.numel() for v in model.state_dict().values())
        mask = torch.zeros(total, dtype=torch.bool, device=self.device)
        offset = 0
        for key, v in model.state_dict().items():
            n = v.numel()
            if key in param_keys:
                mask[offset:offset + n] = True
            offset += n
        return mask

    def _freeze_head_only(self, model: nn.Module):
        """Freeze the classification head, leaving the body trainable
        (canonical FedBABU body phase)."""
        for name, p in model.named_parameters():
            p.requires_grad_(not name.startswith(self._SINGLE_HEAD_PREFIXES))

    def _update_split_head(self, state, config, loader, epochs, local_lr, initial_weights, method: str):
        """FedRep / FedPer / FedBABU via a single-head model with body-only upload.

        All three keep a persistent private classification head and aggregate
        only representation coordinates (head coords zeroed pre-aggregation).
        Schedules differ:
          fedrep : fedrep_head_epochs head-only epochs (frozen body), then joint
                   epochs (total passes == local_steps for compute parity)
          fedper : all-epoch joint training (private head, body-only upload)
          fedbabu: canonical Oh et al. (ICLR 2022) -- head stays frozen at its
                   random init during ALL federated rounds (body-only gradients);
                   the head is trained exclusively at deployment/eval time.
        """
        model = self.global_model
        vector_to_model(state.weights.to(self.device), model)

        # Head handling:
        #   FedRep/FedPer: restore the persistent private head from the
        #     previous round before training.
        #   FedBABU (canonical): the head remains at the client's original
        #     random initialization forever -- it is snapshotted once and
        #     re-restored every round so body-only broadcasts cannot clobber
        #     it; it is only trained at deployment time.
        if method == "fedbabu":
            head_module = model.fc2 if hasattr(model, "fc2") else model.classifier
            if getattr(state, "local_head_state", None) is None:
                state.local_head_state = {k: v.clone()
                                          for k, v in head_module.state_dict().items()}
            head_module.load_state_dict(state.local_head_state)
        elif getattr(state, "local_head_state", None) is not None:
            head_module = model.fc2 if hasattr(model, "fc2") else model.classifier
            head_module.load_state_dict(state.local_head_state)

        num_classes = c = self.num_classes
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)

        head_module_name = "fc2" if hasattr(model, "fc2") else "classifier"
        is_byz = getattr(state, "is_byzantine", False)
        byz_type = getattr(state, "byzantine_type", "label_flip")

        if method == "fedrep":
            e_head = max(1, min(getattr(config, "fedrep_head_epochs", 1), epochs))
            # Cold-start guard: a head-only phase over a frozen, unadapted
            # backbone saturates the classifier before any signal arrives
            # (observed as weight explosion past 1e6 and dead gradients).
            # Warm up jointly until a private head exists, then alternate.
            has_private_head = getattr(state, "local_head_state", None) is not None
            if has_private_head:
                schedule = ["head"] * e_head + ["joint"] * (epochs - e_head)
            else:
                schedule = ["joint"] * epochs
        elif method == "fedbabu":
            schedule = ["body"] * epochs
        else:  # fedper
            schedule = ["joint"] * epochs

        head_params = [p for n, p in model.named_parameters()
                       if n.startswith(self._SINGLE_HEAD_PREFIXES)]

        for phase in schedule:
            head_only = (phase == "head")
            if method != "fedper":
                if phase == "body":
                    self._freeze_head_only(model)
                    for m in model.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            m.train()
                else:
                    self._freeze_single_head_backbone(model, frozen=head_only)
                    # While the body is frozen, pause BatchNorm statistics: a
                    # drifting feature distribution under a stationary head drives
                    # the classifier into softmax saturation (observed as weight
                    # explosion to O(1e6+) and dead gradients).
                    for m in model.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            m.eval() if head_only else m.train()
            for images, labels in loader:
                if images.device != self.device:
                    images, labels = images.to(self.device), labels.to(self.device)
                if is_byz and byz_type == "label_flip":
                    labels = self._flip_labels(labels, num_classes)
                optimizer.zero_grad(set_to_none=True)
                loss = self.criterion(model(images), labels)
                loss.backward()
                if head_only:
                    # Bound the cold-start head update; unclipped steps on
                    # frozen random features saturate the softmax immediately.
                    torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
                optimizer.step()
        model.train()
        # Restore full trainability so frozen phases never leak into the next
        # client or round sharing this persistent model instance.
        for p in model.parameters():
            p.requires_grad_(True)

        full_vec = model_to_vector(model).detach()
        head_mask = self._head_param_mask(model)

        # Persist private head; upload body-only coordinates.
        # FedBABU keeps its head at the random initialization across rounds,
        # so there is no trained private-head state to persist.
        head_module = getattr(model, head_module_name)
        if method != "fedbabu":
            state.local_head_state = {k: v.clone() for k, v in head_module.state_dict().items()}
        state.local_weights = full_vec.clone()          # personalized eval model
        upload = full_vec.clone()
        upload[head_mask.to(upload.device)] = 0.0       # server never sees the head
        state.weights = upload

        return self._apply_byzantine_attack(state, initial_weights)

    def _freeze_backbone(self, model: nn.Module, frozen: bool):
        """Freeze/unfreeze everything outside the three classification heads.

        While frozen, BatchNorm running statistics are paused (module eval)
        so the shared feature extractor cannot drift through buffer updates.
        """
        for name, p in model.named_parameters():
            is_head = name.startswith(self._HEAD_PARAM_PREFIXES)
            p.requires_grad_(is_head or not frozen)
        if frozen:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
        else:
            model.train()

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def update(self, state: ClientState, client_dataset, config: ClientConfig, rng, current_lr: float = None):
        """
        Loads weight tensors into models, trains on client_dataset, and returns updated state weights.
        Supports:
        1. Algorithm Personalization: Ditto (proximal L2), APFL (blended prediction with learnable alpha).
        2. Ensemble Optimization: Shared-backbone, adaptive step allocation, root-anchored asymmetric distillation.
        3. Byzantine Attacks: Label Flip, Sign Flip, Gradient Ascent, Random Noise.
        """
        initial_weights = state.weights.clone()
        cfg_model_name = getattr(config, "model_name", "simple_cnn")
        if cfg_model_name != self.model_name:
            self._init_models(cfg_model_name, self.in_channels)

        local_lr = current_lr if current_lr is not None else config.local_lr

        if len(client_dataset) == 0:
            return state

        loader = self._make_loader(client_dataset, config)
        epochs = config.local_steps

        method = config.personalization_method
        if method == "ditto":
            return self._update_ditto(state, config, loader, epochs, local_lr, initial_weights)
        if method == "apfl":
            return self._update_apfl(state, config, loader, epochs, local_lr, initial_weights)
        if method == "fedala":
            return self._update_fedala(state, config, client_dataset, loader, epochs, local_lr, initial_weights)
        if method in ("fedrep", "fedper", "fedbabu"):
            return self._update_split_head(state, config, loader, epochs, local_lr, initial_weights, method)

        use_shared_backbone = (
            config.compute_optimization_mode == "shared_backbone"
            and (config.use_ensemble or config.hierarchical_ensemble)
        )
        if use_shared_backbone:
            return self._update_hep(state, client_dataset, config, loader, local_lr, initial_weights)

        return self._update_standard(state, config, loader, epochs, local_lr, initial_weights)

    # ------------------------------------------------------------------
    # Baseline: Ditto (Proximal Personalization)
    # ------------------------------------------------------------------

    def _update_ditto(self, state, config, loader, epochs, local_lr, initial_weights):
        global_model = self.global_model
        global_model.train()
        global_optimizer = torch.optim.SGD(global_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)
        num_classes = self.num_classes

        # Step 1: Load broadcast weights using state-dict based approach (includes BN buffers)
        vector_to_model(state.weights.to(self.device), global_model)

        # Canonical Ditto (Li et al., ICML 2021): the proximal anchor of the
        # personalization objective is the RECEIVED global model w^t, not the
        # post-update one. Parameters-only vectorization matches
        # parameters_to_vector(local_model) used inside the penalty.
        w_received = torch.nn.utils.parameters_to_vector(
            global_model.parameters()).detach()

        is_byz = getattr(state, "is_byzantine", False)
        byz_type = getattr(state, "byzantine_type", "label_flip")

        # Standard FedAvg global model local training
        for epoch in range(epochs):
            for images, labels in loader:
                if images.device != self.device:
                    images, labels = images.to(self.device), labels.to(self.device)
                if is_byz and byz_type == "label_flip":
                    labels = self._flip_labels(labels, num_classes)
                global_optimizer.zero_grad(set_to_none=True)
                logits = global_model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                global_optimizer.step()

        updated_global_w = model_to_vector(global_model).detach()
        state.weights = updated_global_w

        w_server_params = w_received

        # Step 2: Personalized model local fine-tuning with Ditto L2 penalty
        local_model = self.local_model
        if state.local_weights is not None and len(state.local_weights) == len(updated_global_w):
            vector_to_model(state.local_weights.to(self.device), local_model)
        else:
            vector_to_model(updated_global_w, local_model)

        local_model.train()
        local_optimizer = torch.optim.SGD(local_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)
        ditto_lambda = config.ditto_lambda

        for epoch in range(epochs):
            for images, labels in loader:
                if images.device != self.device:
                    images, labels = images.to(self.device), labels.to(self.device)
                if is_byz and byz_type == "label_flip":
                    labels = self._flip_labels(labels, num_classes)
                local_optimizer.zero_grad(set_to_none=True)
                logits_p = local_model(images)
                loss_ce = self.criterion(logits_p, labels)

                # Proximal L2 penalty ||w_p - w_server||^2 anchored to the updated global model parameters
                local_params = torch.nn.utils.parameters_to_vector(local_model.parameters())
                loss_prox = 0.5 * ditto_lambda * torch.sum((local_params - w_server_params) ** 2)

                total_loss = loss_ce + loss_prox
                total_loss.backward()
                local_optimizer.step()

        state.local_weights = model_to_vector(local_model).detach()
        return self._apply_byzantine_attack(state, initial_weights)

    # ------------------------------------------------------------------
    # Baseline: APFL (Adaptive Blended Personalization)
    # ------------------------------------------------------------------

    def _apfl_alpha_step(self, alpha, logits_l, logits_g, logits_blend, local_lr):
        grad_alpha = torch.sum((logits_l - logits_g) * F.softmax(logits_blend, dim=1)).item()
        return max(0.01, min(0.99, alpha - local_lr * grad_alpha * 0.1))

    def _update_apfl(self, state, config, loader, epochs, local_lr, initial_weights):
        is_byz = getattr(state, "is_byzantine", False)
        byz_type = getattr(state, "byzantine_type", "label_flip")

        if config.compute_optimization_mode == "shared_backbone" and len(state.weights) == len(model_to_vector(self.multihead_model)):
            multi_model = self.multihead_model
            vector_to_model(state.weights.to(self.device), multi_model)
            if state.local_head_state is not None:
                multi_model.fc2_local.load_state_dict(state.local_head_state)

            num_classes = self.num_classes
            multi_model.train()
            optimizer = torch.optim.SGD(multi_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)
            alpha = getattr(state, "apfl_alpha", config.apfl_alpha)

            for epoch in range(epochs):
                for images, labels in loader:
                    if images.device != self.device:
                        images, labels = images.to(self.device), labels.to(self.device)
                    if is_byz and byz_type == "label_flip":
                        labels = self._flip_labels(labels, num_classes)

                    optimizer.zero_grad(set_to_none=True)
                    logits_g, _, logits_l = multi_model(images, head="all")
                    logits_blend = alpha * logits_l + (1.0 - alpha) * logits_g.detach()

                    loss_g = self.criterion(logits_g, labels)
                    loss_blend = self.criterion(logits_blend, labels)
                    total_loss = loss_g + loss_blend
                    total_loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        alpha = self._apfl_alpha_step(alpha, logits_l, logits_g, logits_blend, local_lr)

            state.apfl_alpha = alpha
            state.weights = model_to_vector(multi_model).detach()
            state.local_head_state = {k: v.clone() for k, v in multi_model.fc2_local.state_dict().items()}
            state.local_weights = state.weights.clone()
            return self._apply_byzantine_attack(state, initial_weights)

        global_model = self.global_model
        local_model = self.local_model

        vector_to_model(state.weights.to(self.device), global_model)
        if state.local_weights is not None and len(state.local_weights) == len(state.weights):
            vector_to_model(state.local_weights.to(self.device), local_model)
        else:
            vector_to_model(state.weights.to(self.device), local_model)

        num_classes = self.num_classes
        global_model.train()
        local_model.train()

        global_opt = torch.optim.SGD(global_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)
        local_opt = torch.optim.SGD(local_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)

        alpha = getattr(state, "apfl_alpha", config.apfl_alpha)

        for epoch in range(epochs):
            for images, labels in loader:
                if images.device != self.device:
                    images, labels = images.to(self.device), labels.to(self.device)
                if is_byz and byz_type == "label_flip":
                    labels = self._flip_labels(labels, num_classes)

                logits_g = global_model(images)
                logits_l = local_model(images)
                logits_blend = alpha * logits_l + (1.0 - alpha) * logits_g.detach()

                loss_g = self.criterion(logits_g, labels)
                loss_blend = self.criterion(logits_blend, labels)

                # Decoupled global and local updates per Deng et al. (2020) Algorithm 1
                global_opt.zero_grad(set_to_none=True)
                loss_g.backward()
                global_opt.step()

                local_opt.zero_grad(set_to_none=True)
                loss_blend.backward()
                local_opt.step()

                # Gradient update for alpha
                with torch.no_grad():
                    alpha = self._apfl_alpha_step(alpha, logits_l, logits_g, logits_blend, local_lr)

        state.apfl_alpha = alpha
        state.weights = model_to_vector(global_model).detach()
        state.local_weights = model_to_vector(local_model).detach()
        return self._apply_byzantine_attack(state, initial_weights)

    # ------------------------------------------------------------------
    # Baseline: FedALA (Adaptive Local Aggregation)
    # ------------------------------------------------------------------

    def _ala_adaptive_local_aggregation(self, state, global_model, local_model,
                                        client_dataset, config):
        """Faithful port of the FedALA ALA module (Zhang et al., TPDS 2023;
        reference implementation TsingZ0/FedALA, system/utils/ALA.py).

        Learns element-wise weights omega in [0,1] interpolating between the
        client's previous local model and the received global model on the top
        `ala_layer_idx` parameter tensors (all lower layers are taken from the
        global model), by gradient descent on locally sampled data.
        Defaults mirror the official configuration: rand_percent=80,
        layer_idx=2, eta=1.0, threshold=0.1, num_pre_loss=10.
        """
        from src.data.dataset import ClientDataset

        device = self.device
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())
        layer_idx = min(max(1, int(getattr(config, "ala_layer_idx", 2))), len(params))

        # Official guard: deactivate ALA at the first communication iteration
        # (received global == local).
        if float(torch.sum(params_g[0] - params[0]).item()) == 0.0:
            return

        eta = float(getattr(config, "ala_eta", 1.0))
        rand_percent = int(getattr(config, "ala_rand_percent", 80))
        threshold = float(getattr(config, "ala_threshold", 0.1))
        num_pre_loss = int(getattr(config, "ala_num_pre_loss", 10))

        n_total = len(client_dataset)
        rand_num = max(2, int(rand_percent * n_total / 100))
        gen = torch.Generator().manual_seed(100000 + int(state.client_id) * 7919)
        start = int(torch.randint(0, n_total - rand_num + 1, (1,), generator=gen).item())
        ala_dataset = ClientDataset(client_dataset, list(range(start, start + rand_num)))
        ala_loader = self._make_loader(ala_dataset, config)

        # Preserve all lower layers from the global model; only the top
        # `layer_idx` parameter tensors are adaptively aggregated.
        for param, param_g in zip(params[:-layer_idx], params_g[:-layer_idx]):
            param.data = param_g.data.clone()

        # Temp copy of the local model used only for weight learning.
        try:
            model_t = copy.deepcopy(local_model)
        except Exception:
            # Some accelerator backends (e.g. DirectML) do not support
            # deepcopying device tensors; roundtrip through CPU instead.
            local_model.to("cpu")
            model_t = copy.deepcopy(local_model)
            local_model.to(device)
            model_t.to(device)
        params_t = list(model_t.parameters())
        params_p = params[-layer_idx:]
        params_gp = params_g[-layer_idx:]
        params_tp = params_t[-layer_idx:]

        for param in params_t[:-layer_idx]:
            param.requires_grad = False

        # lr=0 optimizer: gradients are read manually to update omega.
        optimizer_t = torch.optim.SGD(params_tp, lr=0)

        weights = getattr(state, "ala_weights", None)
        if weights is None or len(weights) != len(params_p):
            weights = [torch.ones_like(p.data).to(device) for p in params_p]

        def interpolate():
            for param_t, param, param_g, w in zip(params_tp, params_p, params_gp, weights):
                param_t.data = param.data + (param_g.data - param.data) * w

        interpolate()

        losses = []
        cnt = 0
        max_epochs = int(getattr(config, "ala_max_epochs", 50))
        while True:
            for x, y in ala_loader:
                if x.device != device:
                    x, y = x.to(device), y.to(device)
                optimizer_t.zero_grad(set_to_none=True)
                output = model_t(x)
                loss_value = self.criterion(output, y)
                loss_value.backward()

                with torch.no_grad():
                    for param_t, param, param_g, w in zip(params_tp, params_p, params_gp, weights):
                        w.data = torch.clamp(
                            w - eta * (param_t.grad * (param_g.data - param.data)), 0, 1)
                    interpolate()

            losses.append(float(loss_value.item()))
            cnt += 1

            if not getattr(state, "ala_start_phase", True):
                break
            if cnt >= max_epochs:
                break
            if len(losses) > num_pre_loss and \
                    float(np.std(losses[-num_pre_loss:])) < threshold:
                break

        state.ala_start_phase = False
        state.ala_weights = [w.detach().clone() for w in weights]

        # Write the interpolated higher layers back as this round's local init.
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()

    def _update_fedala(self, state, config, client_dataset, loader, epochs, local_lr, initial_weights):
        """FedALA baseline: ALA local initialization followed by standard
        local training; uploads the full model (FedAvg aggregation)."""
        is_byz = getattr(state, "is_byzantine", False)
        byz_type = getattr(state, "byzantine_type", "label_flip")

        global_model = self.global_model
        local_model = self.local_model

        vector_to_model(state.weights.to(self.device), global_model)
        if state.local_weights is not None and len(state.local_weights) == len(state.weights):
            vector_to_model(state.local_weights.to(self.device), local_model)
        else:
            vector_to_model(state.weights.to(self.device), local_model)

        self._ala_adaptive_local_aggregation(state, global_model, local_model,
                                             client_dataset, config)

        local_model.train()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=local_lr,
                                    momentum=0.9, nesterov=True, weight_decay=1e-4,
                                    foreach=False)
        for epoch in range(epochs):
            for images, labels in loader:
                if images.device != self.device:
                    images, labels = images.to(self.device), labels.to(self.device)
                if is_byz and byz_type == "label_flip":
                    labels = self._flip_labels(labels, self.num_classes)
                optimizer.zero_grad(set_to_none=True)
                loss = self.criterion(local_model(images), labels)
                loss.backward()
                optimizer.step()

        updated_w = model_to_vector(local_model).detach()
        state.weights = updated_w.clone()
        state.local_weights = updated_w
        return self._apply_byzantine_attack(state, initial_weights)

    # ------------------------------------------------------------------
    # HEP: Shared Backbone Multi-Head Optimization (Hierarchical Ensemble)
    # ------------------------------------------------------------------

    def _update_hep(self, state, client_dataset, config, loader, local_lr, initial_weights):
        multi_model = self.multihead_model
        vector_to_model(state.weights.to(self.device), multi_model)

        # Restore specialized head weights saved from previous round
        if state.parent_head_state is not None:
            multi_model.fc2_parent.load_state_dict(state.parent_head_state)

        if state.local_head_state is not None:
            multi_model.fc2_local.load_state_dict(state.local_head_state)

        multi_model.train()
        optimizer = torch.optim.SGD(multi_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)

        num_classes = self.num_classes
        r_skew, active_mask, class_counts = self._compute_label_stats(client_dataset, num_classes)
        state.r_skew = r_skew

        # Heterogeneity-Calibrated Simplex Prior: [local, parent, root]
        if config.use_heterogeneity_prior:
            pi_root = min(1.0, max(0.0, r_skew ** config.heterogeneity_gamma))
            pi_local = (1.0 - pi_root) * (1.0 - r_skew)
            pi_parent = max(0.0, 1.0 - pi_root - pi_local)
            pi_prior = torch.tensor([pi_local, pi_parent, pi_root], dtype=torch.float32, device=self.device)
        else:
            pi_prior = torch.tensor([0.333, 0.333, 0.334], dtype=torch.float32, device=self.device)

        ensemble_alpha = getattr(state, "ensemble_alpha", [pi_prior[0].item(), pi_prior[1].item(), pi_prior[2].item()])
        if not isinstance(ensemble_alpha, torch.Tensor):
            ensemble_alpha_t = torch.tensor(ensemble_alpha, dtype=torch.float32, device=self.device)
        else:
            ensemble_alpha_t = ensemble_alpha.to(self.device)

        # Head-training schedule -> per-epoch loss multiplier per head.
        # piecewise: integer budgets, active while epoch < budget (legacy).
        # binomial: continuous partition-of-unity weights, constant across epochs.
        total_budget = config.total_local_steps
        if config.head_training_schedule == "binomial":
            num_active = int(active_mask.sum().item()) if active_mask is not None else num_classes
            lw_r, lw_p, lw_l = self._alloc_steps_binomial(
                r_skew, total_budget, config.binomial_anchor_min,
                num_classes=num_classes, local_classes=num_active,
                num_clusters=getattr(config, "num_clusters", 3),
            )
            head_budgets = {"root": lw_r, "parent": lw_p, "local": lw_l}
            max_steps = max(1, total_budget)
        else:
            e_root, e_parent, e_local = self._alloc_steps_piecewise(r_skew, total_budget)
            head_budgets = {"root": float(e_root), "parent": float(e_parent), "local": float(e_local)}
            max_steps = max(e_root, e_parent, e_local)

        state.head_steps = {k: int(round(v)) for k, v in head_budgets.items()}

        def _head_mult(key: str, epoch: int) -> float:
            if config.head_training_schedule == "binomial":
                return head_budgets[key]
            return 1.0 if epoch < int(head_budgets[key]) else 0.0

        # Two-stage high-cardinality schedule:
        # Split E=5 into:
        # - Collaborative Stage (Epochs 0 to 2): Backbone + Root + Parent trained concurrently
        # - Specialization Stage (Epochs 3 to 4): Backbone frozen, Local head trained on local data
        two_stage = (
            config.high_cardinality_two_stage
            and num_classes >= config.two_stage_min_classes
            and max_steps >= 3
        )
        collab_epochs = max(1, max_steps - 2) if two_stage else max_steps
        backbone_frozen = False

        last_loss_root, last_loss_parent, last_loss_local = 0.0, 0.0, 0.0

        # Continuous Dimensional Capacity Factor: anchors shared backbone when num_classes >= 100
        # rho(C) = 1.0 for C=10 (full backbone adaptation for peak PFL accuracy)
        # rho(C) = 0.1 for C=100 (90% StopGrad anchoring to protect 100-class feature extractor)
        dim_capacity = min(1.0, 10.0 / float(num_classes))

        do_distill = config.ensemble_distillation
        distill_lambda = config.distillation_lambda
        alpha_lr_scale = config.ensemble_alpha_lr_scale
        prior_mix = config.prior_mix

        for epoch in range(max_steps):
            # Two-stage freeze transitions (once per epoch boundary)
            if two_stage:
                is_collab = epoch < collab_epochs
                should_freeze = not is_collab
                if should_freeze != backbone_frozen:
                    self._freeze_backbone(multi_model, frozen=should_freeze)
                    backbone_frozen = should_freeze

                if is_collab:
                    m_root = _head_mult("root", epoch)
                    m_parent = _head_mult("parent", epoch)
                    m_local = 0.0
                else:
                    m_root = 0.0
                    m_parent = 0.0
                    m_local = 1.0
            else:
                m_root = _head_mult("root", epoch)
                m_parent = _head_mult("parent", epoch)
                m_local = _head_mult("local", epoch)

            for images, labels in loader:
                if images.device != self.device:
                    images, labels = images.to(self.device), labels.to(self.device)
                if getattr(state, "is_byzantine", False) and getattr(state, "byzantine_type", "label_flip") == "label_flip":
                    labels = self._flip_labels(labels, num_classes)
                optimizer.zero_grad(set_to_none=True)

                if dim_capacity < 1.0 and hasattr(multi_model, "extract_features"):
                    h = multi_model.extract_features(images)
                    logits_root = multi_model.fc2_root(h)
                    logits_parent = multi_model.fc2_parent(h)
                    h_anchored = dim_capacity * h + (1.0 - dim_capacity) * h.detach()
                    logits_local = multi_model.fc2_local(h_anchored)
                else:
                    logits_root, logits_parent, logits_local = multi_model(images, head="all")

                ce_root = self.criterion(logits_root, labels) if m_root > 0 else None
                
                ce_parent = None
                if m_parent > 0:
                    if config.active_class_loss_mask:
                        logits_parent_masked = logits_parent.masked_fill(~active_mask.unsqueeze(0), -1e9)
                        ce_parent = self.criterion(logits_parent_masked, labels)
                    else:
                        ce_parent = self.criterion(logits_parent, labels)

                ce_local = None
                if m_local > 0:
                    if config.active_class_loss_mask:
                        logits_local_masked = logits_local.masked_fill(~active_mask.unsqueeze(0), -1e9)
                    else:
                        logits_local_masked = logits_local
                    if class_counts.sum() > 0:
                        w_cls = 1.0 / (class_counts ** 0.5 + 1e-4)
                        w_cls = w_cls * active_mask.float()
                        w_cls = w_cls / (w_cls[w_cls > 0].mean() + 1e-8)
                        ce_local = F.cross_entropy(logits_local_masked, labels, weight=w_cls)
                    else:
                        ce_local = self.criterion(logits_local_masked, labels)

                loss_root = m_root * ce_root if ce_root is not None else self.zero_loss
                loss_parent = m_parent * ce_parent if ce_parent is not None else self.zero_loss
                loss_local = m_local * ce_local if ce_local is not None else self.zero_loss

                if ce_root is not None:
                    last_loss_root = ce_root.item()
                if ce_parent is not None:
                    last_loss_parent = ce_parent.item()
                if ce_local is not None:
                    last_loss_local = ce_local.item()

                total_loss = loss_root + loss_parent + loss_local

                # Root-Anchored Asymmetric Distillation (Root is detached teacher)
                if do_distill:
                    teacher_target = F.softmax(logits_root.detach(), dim=1)
                    if m_parent > 0:
                        total_loss += distill_lambda * _kl_div(F.log_softmax(logits_parent, dim=1), teacher_target)
                    if m_local > 0:
                        total_loss += distill_lambda * _kl_div(F.log_softmax(logits_local, dim=1), teacher_target)

                total_loss.backward()
                optimizer.step()

                # Calibrated 3-tier ensemble_alpha simplex vector blend (Zero CPU Stalls)
                with torch.no_grad():
                    w_alpha = F.softmax(ensemble_alpha_t, dim=0)
                    logits_blend = w_alpha[0] * logits_local + w_alpha[1] * logits_parent + w_alpha[2] * logits_root
                    p_blend = F.softmax(logits_blend, dim=1)

                    target_onehot = torch.zeros_like(p_blend)
                    target_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
                    grad_logits = (p_blend - target_onehot) / labels.size(0)

                    # Scale gradients by prediction entropy to prevent overfitted logit magnitude explosion
                    norm_factor = torch.clamp(torch.abs(logits_local).mean(), min=1e-4)
                    g_local = torch.sum(grad_logits * (logits_local / norm_factor))
                    g_parent = torch.sum(grad_logits * (logits_parent / norm_factor))
                    g_root = torch.sum(grad_logits * (logits_root / norm_factor))

                    alpha_lr = local_lr * alpha_lr_scale
                    ensemble_alpha_t[0] -= alpha_lr * g_local
                    ensemble_alpha_t[1] -= alpha_lr * g_parent
                    ensemble_alpha_t[2] -= alpha_lr * g_root

                    # Smooth blend with Heterogeneity Prior pi_prior
                    w_updated = F.softmax(ensemble_alpha_t, dim=0)
                    w_final = prior_mix * w_updated + (1.0 - prior_mix) * pi_prior
                    ensemble_alpha_t = torch.log(w_final + 1e-8)

        if two_stage and backbone_frozen:
            self._freeze_backbone(multi_model, frozen=False)

        state.head_losses = {"root": last_loss_root, "parent": last_loss_parent, "local": last_loss_local}
        state.ensemble_alpha = F.softmax(ensemble_alpha_t, dim=0).detach().cpu().tolist()
        state.active_mask = active_mask.detach().cpu()

        # Save shared multihead weights for global aggregation & head states for local specialization
        state.weights = model_to_vector(multi_model).detach()
        state.parent_head_state = {k: v.clone() for k, v in multi_model.fc2_parent.state_dict().items()}
        state.local_head_state = {k: v.clone() for k, v in multi_model.fc2_local.state_dict().items()}
        state.parent_weights = state.weights.clone()
        state.local_weights = state.weights.clone()

        return self._apply_byzantine_attack(state, initial_weights)

    # ------------------------------------------------------------------
    # Standard / Unconstrained Baseline Training
    # ------------------------------------------------------------------

    def _update_standard(self, state, config, loader, epochs, local_lr, initial_weights):
        global_model = self.global_model
        vector_to_model(state.weights.to(self.device), global_model)
        num_classes = self.num_classes
        global_model.train()
        global_optimizer = torch.optim.SGD(global_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)

        is_byz = getattr(state, "is_byzantine", False)
        byz_type = getattr(state, "byzantine_type", "label_flip")

        for epoch in range(epochs):
            for images, labels in loader:
                if images.device != self.device:
                    images, labels = images.to(self.device), labels.to(self.device)
                if is_byz and byz_type == "label_flip":
                    labels = self._flip_labels(labels, num_classes)
                global_optimizer.zero_grad(set_to_none=True)
                logits = global_model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                global_optimizer.step()

        state.weights = model_to_vector(global_model).detach()
        return self._apply_byzantine_attack(state, initial_weights)
