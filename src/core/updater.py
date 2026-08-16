import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.core.interfaces import ClientState
from src.config import ClientConfig
from src.core.model import SimpleCNN, MultiHeadSimpleCNN, ResNet9, MultiHeadResNet9, model_to_vector, vector_to_model

def _kl_div(log_p, target):
    """DirectML GPU native KL divergence distillation loss -(target * log_p).sum(dim=1).mean()."""
    return -(target * log_p).sum(dim=1).mean()

class PyTorchLocalUpdater:
    def __init__(self, device="cpu", in_channels=1, model_name="simple_cnn"):
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
        self.zero_loss = torch.tensor(0.0, device=self.device)
        self.model_name = model_name
        self.in_channels = in_channels
        
        self._init_models(model_name, in_channels)

    def _init_models(self, model_name, in_channels):
        self.model_name = model_name
        self.in_channels = in_channels
        if model_name == "resnet9":
            self.global_model = ResNet9(in_channels=in_channels).to(self.device)
            self.local_model = ResNet9(in_channels=in_channels).to(self.device)
            self.parent_model = ResNet9(in_channels=in_channels).to(self.device)
            self.multihead_model = MultiHeadResNet9(in_channels=in_channels).to(self.device)
        else:
            self.global_model = SimpleCNN(in_channels=in_channels).to(self.device)
            self.local_model = SimpleCNN(in_channels=in_channels).to(self.device)
            self.parent_model = SimpleCNN(in_channels=in_channels).to(self.device)
            self.multihead_model = MultiHeadSimpleCNN(in_channels=in_channels).to(self.device)

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

        local_lr = current_lr if current_lr is not None else getattr(config, "current_lr", getattr(config, "local_lr", 0.03))
        personalization_method = getattr(config, "personalization_method", "none")
        use_ensemble = getattr(config, "use_ensemble", False)
        train_parent = getattr(config, "hierarchical_ensemble", False)
        compute_mode = getattr(config, "compute_optimization_mode", "shared_backbone")
        do_distill = getattr(config, "ensemble_distillation", True)
        distill_lambda = getattr(config, "distillation_lambda", 0.5)
        
        is_byz = getattr(state, "is_byzantine", False)
        byz_type = getattr(state, "byzantine_type", "label_flip")
        
        if len(client_dataset) == 0:
            return state

        batch_size = min(32, len(client_dataset))
        from src.data.dataset import get_fast_dataloader
        dataloader = get_fast_dataloader(client_dataset, batch_size=batch_size, shuffle=True)
        epochs = getattr(config, "local_steps", 1)

        # ---------------------------------------------------------------------
        # ALGORITHM BASELINE: DITTO (Proximal Personalization)
        # ---------------------------------------------------------------------
        if personalization_method == "ditto":
            global_model = self.global_model
            global_model.train()
            global_optimizer = torch.optim.SGD(global_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)
            num_classes = global_model.fc2.out_features if hasattr(global_model, "fc2") else 10
            
            # Step 1: Load broadcast weights using state-dict based approach (includes BN buffers)
            vector_to_model(state.weights.to(self.device), global_model)
            
            # Step 1: Standard FedAvg global model local training
            for epoch in range(epochs):
                for images, labels in dataloader:
                    if images.device != self.device:
                        images, labels = images.to(self.device), labels.to(self.device)
                    if is_byz and byz_type == "label_flip":
                        labels = (num_classes - 1) - labels
                    global_optimizer.zero_grad(set_to_none=True)
                    logits = global_model(images)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    global_optimizer.step()
            
            updated_global_w = model_to_vector(global_model).detach()
            state.weights = updated_global_w
            
            w_server_params = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
            
            # Step 2: Personalized model local fine-tuning with Ditto L2 penalty
            local_model = self.local_model
            if state.local_weights is not None and len(state.local_weights) == len(updated_global_w):
                vector_to_model(state.local_weights.to(self.device), local_model)
            else:
                vector_to_model(updated_global_w, local_model)
                
            local_model.train()
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)
            ditto_lambda = getattr(config, "ditto_lambda", 0.1)
            
            for epoch in range(epochs):
                for images, labels in dataloader:
                    if images.device != self.device:
                        images, labels = images.to(self.device), labels.to(self.device)
                    if is_byz and byz_type == "label_flip":
                        labels = (num_classes - 1) - labels
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

        # ---------------------------------------------------------------------
        # ALGORITHM BASELINE: APFL (Adaptive Blended Personalization)
        # ---------------------------------------------------------------------
        if personalization_method == "apfl":
            if compute_mode == "shared_backbone" and len(state.weights) == len(model_to_vector(self.multihead_model)):
                multi_model = self.multihead_model
                vector_to_model(state.weights.to(self.device), multi_model)
                if state.local_head_state is not None:
                    multi_model.fc2_local.load_state_dict(state.local_head_state)

                num_classes = multi_model.fc2_local.out_features if hasattr(multi_model, "fc2_local") else 10
                multi_model.train()
                optimizer = torch.optim.SGD(multi_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)
                alpha = getattr(state, "apfl_alpha", getattr(config, "apfl_alpha", 0.5))

                for epoch in range(epochs):
                    for images, labels in dataloader:
                        if images.device != self.device:
                            images, labels = images.to(self.device), labels.to(self.device)
                        if is_byz and byz_type == "label_flip":
                            labels = (num_classes - 1) - labels

                        optimizer.zero_grad(set_to_none=True)
                        logits_g, _, logits_l = multi_model(images, head="all")
                        logits_blend = alpha * logits_l + (1.0 - alpha) * logits_g.detach()

                        loss_g = self.criterion(logits_g, labels)
                        loss_blend = self.criterion(logits_blend, labels)
                        total_loss = loss_g + loss_blend
                        total_loss.backward()
                        optimizer.step()

                        with torch.no_grad():
                            grad_alpha = torch.sum((logits_l - logits_g) * F.softmax(logits_blend, dim=1)).item()
                            alpha = max(0.01, min(0.99, alpha - config.local_lr * grad_alpha * 0.1))

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

            num_classes = global_model.fc2.out_features if hasattr(global_model, "fc2") else 10
            global_model.train()
            local_model.train()
            
            global_opt = torch.optim.SGD(global_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)
            local_opt = torch.optim.SGD(local_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)
            
            alpha = getattr(state, "apfl_alpha", getattr(config, "apfl_alpha", 0.5))

            for epoch in range(epochs):
                for images, labels in dataloader:
                    if images.device != self.device:
                        images, labels = images.to(self.device), labels.to(self.device)
                    if is_byz and byz_type == "label_flip":
                        labels = (num_classes - 1) - labels
                    
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
                        grad_alpha = torch.sum((logits_l - logits_g) * F.softmax(logits_blend, dim=1)).item()
                        alpha = max(0.01, min(0.99, alpha - config.local_lr * grad_alpha * 0.1))

            state.apfl_alpha = alpha
            state.weights = model_to_vector(global_model).detach()
            state.local_weights = model_to_vector(local_model).detach()
            return self._apply_byzantine_attack(state, initial_weights)

        # ---------------------------------------------------------------------
        # PATH 1: SHARED BACKBONE MULTI-HEAD OPTIMIZATION (Hierarchical Ensemble)
        # ---------------------------------------------------------------------
        if compute_mode == "shared_backbone" and (use_ensemble or train_parent):
            multi_model = self.multihead_model
            vector_to_model(state.weights.to(self.device), multi_model)
            
            # Restore specialized head weights saved from previous round
            if state.parent_head_state is not None:
                multi_model.fc2_parent.load_state_dict(state.parent_head_state)
                
            if state.local_head_state is not None:
                multi_model.fc2_local.load_state_dict(state.local_head_state)

            multi_model.train()
            optimizer = torch.optim.SGD(multi_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)

            # Compute local dataset label Shannon entropy ratio R_skew
            use_het_prior = getattr(config, "use_heterogeneity_prior", True)
            het_gamma = getattr(config, "heterogeneity_gamma", 1.0)
            num_classes = multi_model.fc2_local.out_features if hasattr(multi_model, "fc2_local") else 10
            
            r_skew = 0.5
            try:
                if hasattr(client_dataset, "labels") and len(client_dataset) > 0:
                    l_tensor = client_dataset.labels if isinstance(client_dataset.labels, torch.Tensor) else torch.tensor(client_dataset.labels)
                    _, counts = torch.unique(l_tensor, return_counts=True)
                    probs = counts.float() / counts.sum()
                    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
                    import math
                    max_entropy = math.log(num_classes)
                    r_skew = min(1.0, max(0.0, entropy / max_entropy))
            except Exception:
                r_skew = 0.5

            state.r_skew = r_skew

            # Compute Heterogeneity-Calibrated Simplex Prior: [local, parent, root]
            if use_het_prior:
                pi_root = min(1.0, max(0.0, r_skew ** het_gamma))
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

            # Adaptive Step Allocation based on R_skew and head activity
            total_budget = getattr(config, "total_local_steps", 5)
            rem = max(0, total_budget - 3)
            
            if r_skew > 0.7:  # IID / Uniform regime: prioritize Root & Parent heads
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

            state.head_steps = {"root": e_root, "parent": e_parent, "local": e_local}
            last_loss_root, last_loss_parent, last_loss_local = 0.0, 0.0, 0.0

            max_steps = max(e_root, e_parent, e_local)
            for epoch in range(max_steps):
                for images, labels in dataloader:
                    if images.device != self.device:
                        images, labels = images.to(self.device), labels.to(self.device)
                    if is_byz and byz_type == "label_flip":
                        labels = (num_classes - 1) - labels
                    optimizer.zero_grad(set_to_none=True)
                    logits_root, logits_parent, logits_local = multi_model(images, head="all")
                    
                    loss_root = self.criterion(logits_root, labels) if epoch < e_root else self.zero_loss
                    loss_parent = self.criterion(logits_parent, labels) if epoch < e_parent else self.zero_loss
                    loss_local = self.criterion(logits_local, labels) if epoch < e_local else self.zero_loss
                    
                    if epoch < e_root:
                        last_loss_root = loss_root.item()
                    if epoch < e_parent:
                        last_loss_parent = loss_parent.item()
                    if epoch < e_local:
                        last_loss_local = loss_local.item()

                    total_loss = loss_root + loss_parent + loss_local

                    # Root-Anchored Asymmetric Distillation (Root is detached teacher)
                    if do_distill:
                        teacher_target = F.softmax(logits_root.detach(), dim=1)
                        if epoch < e_parent:
                            total_loss += distill_lambda * _kl_div(F.log_softmax(logits_parent, dim=1), teacher_target)
                        if epoch < e_local:
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
                        
                        alpha_lr = local_lr * 0.05
                        ensemble_alpha_t[0] -= alpha_lr * g_local
                        ensemble_alpha_t[1] -= alpha_lr * g_parent
                        ensemble_alpha_t[2] -= alpha_lr * g_root
                        
                        # Smooth blend with Heterogeneity Prior pi_prior
                        w_updated = F.softmax(ensemble_alpha_t, dim=0)
                        w_final = 0.5 * w_updated + 0.5 * pi_prior
                        ensemble_alpha_t = torch.log(w_final + 1e-8)

            state.head_losses = {"root": last_loss_root, "parent": last_loss_parent, "local": last_loss_local}
            state.ensemble_alpha = F.softmax(ensemble_alpha_t, dim=0).detach().cpu().tolist()

            # Save shared multihead weights for global aggregation & head states for local specialization
            state.weights = model_to_vector(multi_model).detach()
            state.parent_head_state = {k: v.clone() for k, v in multi_model.fc2_parent.state_dict().items()}
            state.local_head_state = {k: v.clone() for k, v in multi_model.fc2_local.state_dict().items()}
            state.parent_weights = state.weights.clone()
            state.local_weights = state.weights.clone()

            return self._apply_byzantine_attack(state, initial_weights)

        # ---------------------------------------------------------------------
        # PATH 2: STANDARD / UNCONSTRAINED BASELINE TRAINING
        # ---------------------------------------------------------------------
        global_model = self.global_model
        vector_to_model(state.weights.to(self.device), global_model)
        num_classes = global_model.fc2.out_features if hasattr(global_model, "fc2") else 10
        global_model.train()
        global_optimizer = torch.optim.SGD(global_model.parameters(), lr=local_lr, momentum=0.9, nesterov=True, weight_decay=1e-4, foreach=False)

        for epoch in range(epochs):
            for images, labels in dataloader:
                if images.device != self.device:
                    images, labels = images.to(self.device), labels.to(self.device)
                if is_byz and byz_type == "label_flip":
                    labels = (num_classes - 1) - labels
                global_optimizer.zero_grad(set_to_none=True)
                logits = global_model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                global_optimizer.step()

        state.weights = model_to_vector(global_model).detach()
        return self._apply_byzantine_attack(state, initial_weights)
