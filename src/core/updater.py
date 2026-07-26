import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.core.interfaces import ClientState
from src.config import ClientConfig
from src.core.model import SimpleCNN, MultiHeadSimpleCNN, ResNet9, MultiHeadResNet9

def _kl_div(log_p, target):
    """DirectML GPU native KL divergence distillation loss -(target * log_p).sum(dim=1).mean()."""
    return -(target * log_p).sum(dim=1).mean()

class PyTorchLocalUpdater:
    def __init__(self, device="cpu", in_channels=1, model_name="simple_cnn"):
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
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

    def update(self, state: ClientState, client_dataset, config: ClientConfig, rng):
        """
        Loads weight tensors into models, trains on client_dataset, and returns updated state weights.
        Supports:
        1. Algorithm Personalization: Ditto (proximal L2), APFL (blended prediction with learnable alpha).
        2. Ensemble Optimization: Shared-backbone, adaptive step allocation, root-anchored asymmetric distillation.
        """
        cfg_model_name = getattr(config, "model_name", "simple_cnn")
        if cfg_model_name != self.model_name:
            self._init_models(cfg_model_name, self.in_channels)

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
        dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        epochs = getattr(config, "local_steps", 1)

        # ---------------------------------------------------------------------
        # ALGORITHM BASELINE: DITTO (Proximal Personalization)
        # ---------------------------------------------------------------------
        if personalization_method == "ditto":
            global_model = self.global_model
            torch.nn.utils.vector_to_parameters(state.weights.to(self.device), global_model.parameters())
            global_model.train()
            global_optimizer = torch.optim.SGD(global_model.parameters(), lr=config.local_lr)
            
            # Step 1: Standard FedAvg global model local training
            for epoch in range(epochs):
                for images, labels in dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    if is_byz and byz_type == "label_flip":
                        labels = 9 - labels
                    global_optimizer.zero_grad()
                    logits = global_model(images)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    global_optimizer.step()
            
            updated_global_w = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
            state.weights = updated_global_w.cpu()
            
            # Step 2: Personalized model local fine-tuning with Ditto L2 penalty
            local_model = self.local_model
            if state.local_weights is not None and len(state.local_weights) == len(state.weights):
                torch.nn.utils.vector_to_parameters(state.local_weights.to(self.device), local_model.parameters())
            else:
                torch.nn.utils.vector_to_parameters(updated_global_w, local_model.parameters())
                
            local_model.train()
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=config.local_lr)
            ditto_lambda = getattr(config, "ditto_lambda", 0.1)
            
            for epoch in range(epochs):
                for images, labels in dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    if is_byz and byz_type == "label_flip":
                        labels = 9 - labels
                    local_optimizer.zero_grad()
                    logits_p = local_model(images)
                    loss_ce = self.criterion(logits_p, labels)
                    
                    # Proximal L2 penalty ||w_p - w_g||^2
                    local_params = torch.nn.utils.parameters_to_vector(local_model.parameters())
                    loss_prox = 0.5 * ditto_lambda * torch.sum((local_params - updated_global_w) ** 2)
                    
                    total_loss = loss_ce + loss_prox
                    total_loss.backward()
                    local_optimizer.step()
                    
            state.local_weights = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach().cpu()
            return state

        # ---------------------------------------------------------------------
        # ALGORITHM BASELINE: APFL (Adaptive Blended Personalization)
        # ---------------------------------------------------------------------
        if personalization_method == "apfl":
            global_model = self.global_model
            local_model = self.local_model
            
            torch.nn.utils.vector_to_parameters(state.weights.to(self.device), global_model.parameters())
            if state.local_weights is not None and len(state.local_weights) == len(state.weights):
                torch.nn.utils.vector_to_parameters(state.local_weights.to(self.device), local_model.parameters())
            else:
                torch.nn.utils.vector_to_parameters(state.weights.to(self.device), local_model.parameters())

            global_model.train()
            local_model.train()
            
            global_opt = torch.optim.SGD(global_model.parameters(), lr=config.local_lr)
            local_opt = torch.optim.SGD(local_model.parameters(), lr=config.local_lr)
            
            alpha = getattr(state, "apfl_alpha", getattr(config, "apfl_alpha", 0.5))

            for epoch in range(epochs):
                for images, labels in dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    if is_byz and byz_type == "label_flip":
                        labels = 9 - labels
                    
                    global_opt.zero_grad()
                    local_opt.zero_grad()
                    
                    logits_g = global_model(images)
                    logits_l = local_model(images)
                    
                    # Blended logits: alpha * local + (1 - alpha) * global
                    logits_blend = alpha * logits_l + (1.0 - alpha) * logits_g
                    
                    loss_blend = self.criterion(logits_blend, labels)
                    loss_g = self.criterion(logits_g, labels)
                    loss_l = self.criterion(logits_l, labels)
                    
                    total_loss = loss_blend + 0.5 * loss_g + 0.5 * loss_l
                    total_loss.backward()
                    
                    global_opt.step()
                    local_opt.step()
                    
                    # Gradient update for alpha
                    with torch.no_grad():
                        grad_alpha = torch.sum((logits_l - logits_g) * F.softmax(logits_blend, dim=1)).item()
                        alpha = max(0.01, min(0.99, alpha - config.local_lr * grad_alpha * 0.01))

            state.apfl_alpha = alpha
            state.weights = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach().cpu()
            state.local_weights = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach().cpu()
            return state

        # ---------------------------------------------------------------------
        # PATH 1: SHARED BACKBONE MULTI-HEAD OPTIMIZATION (Hierarchical Ensemble)
        # ---------------------------------------------------------------------
        if compute_mode == "shared_backbone" and (use_ensemble or train_parent):
            multi_model = self.multihead_model
            torch.nn.utils.vector_to_parameters(state.weights.to(self.device), multi_model.parameters())
            
            # Load parent/local head weights if explicitly saved in state
            if state.parent_weights is not None and len(state.parent_weights) == len(state.weights):
                parent_temp = ResNet9(in_channels=self.in_channels).to(self.device) if self.model_name == "resnet9" else SimpleCNN(in_channels=self.in_channels).to(self.device)
                torch.nn.utils.vector_to_parameters(state.parent_weights.to(self.device), parent_temp.parameters())
                multi_model.fc2_parent.load_state_dict(parent_temp.fc2.state_dict())
                
            if state.local_weights is not None and len(state.local_weights) == len(state.weights):
                local_temp = ResNet9(in_channels=self.in_channels).to(self.device) if self.model_name == "resnet9" else SimpleCNN(in_channels=self.in_channels).to(self.device)
                torch.nn.utils.vector_to_parameters(state.local_weights.to(self.device), local_temp.parameters())
                multi_model.fc2_local.load_state_dict(local_temp.fc2.state_dict())

            multi_model.train()
            optimizer = torch.optim.SGD(multi_model.parameters(), lr=config.local_lr, foreach=False)

            # Adaptive Step Allocation
            total_budget = getattr(config, "total_local_steps", 5)
            prev_losses = getattr(state, "head_losses", {})
            
            # Calculate step allocation per head out of total_budget (minimum 1 step per head)
            if "root" in prev_losses and "parent" in prev_losses and "local" in prev_losses:
                # Relative loss improvement / activity
                l_root, l_parent, l_local = prev_losses["root"], prev_losses["parent"], prev_losses["local"]
                sum_loss = max(1e-5, l_root + l_parent + l_local)
                rem = max(0, total_budget - 3)
                e_root = 1 + int(round(rem * (l_root / sum_loss)))
                e_parent = 1 + int(round(rem * (l_parent / sum_loss)))
                e_local = 1 + int(round(rem * (l_local / sum_loss)))
            else:
                # Round 1 Fallback Prior (N : N/K : 1 = 15:5:1 -> 3:1:1)
                e_root, e_parent, e_local = 3, 1, 1

            state.head_steps = {"root": e_root, "parent": e_parent, "local": e_local}

            last_loss_root, last_loss_parent, last_loss_local = 0.0, 0.0, 0.0

            max_steps = max(e_root, e_parent, e_local)
            for epoch in range(max_steps):
                for images, labels in dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    if is_byz and byz_type == "label_flip":
                        labels = 9 - labels

                    optimizer.zero_grad()
                    logits_root, logits_parent, logits_local = multi_model(images, head="all")
                    
                    loss_root = self.criterion(logits_root, labels) if epoch < e_root else torch.tensor(0.0, device=self.device)
                    loss_parent = self.criterion(logits_parent, labels) if epoch < e_parent else torch.tensor(0.0, device=self.device)
                    loss_local = self.criterion(logits_local, labels) if epoch < e_local else torch.tensor(0.0, device=self.device)
                    
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

            state.head_losses = {"root": last_loss_root, "parent": last_loss_parent, "local": last_loss_local}

            full_weights = torch.nn.utils.parameters_to_vector(multi_model.parameters()).detach().cpu()
            state.weights = full_weights
            state.parent_weights = full_weights.clone()
            state.local_weights = full_weights.clone()
            return state

        # ---------------------------------------------------------------------
        # PATH 2: STANDARD / UNCONSTRAINED BASELINE TRAINING
        # ---------------------------------------------------------------------
        global_model = self.global_model
        torch.nn.utils.vector_to_parameters(state.weights.to(self.device), global_model.parameters())
        global_model.train()
        global_optimizer = torch.optim.SGD(global_model.parameters(), lr=config.local_lr)

        for epoch in range(epochs):
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                if is_byz and byz_type == "label_flip":
                    labels = 9 - labels
                global_optimizer.zero_grad()
                logits = global_model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                global_optimizer.step()

        state.weights = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach().cpu()
        return state
