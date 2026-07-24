import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.core.interfaces import ClientState
from src.config import ClientConfig
from src.core.model import SimpleCNN, MultiHeadSimpleCNN

class PyTorchLocalUpdater:
    def __init__(self, device="cpu", in_channels=1):
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Reuse standard models
        self.global_model = SimpleCNN(in_channels=in_channels).to(self.device)
        self.local_model = SimpleCNN(in_channels=in_channels).to(self.device)
        self.parent_model = SimpleCNN(in_channels=in_channels).to(self.device)
        
        # Multi-head model for shared_backbone compute optimization
        self.multihead_model = MultiHeadSimpleCNN(in_channels=in_channels).to(self.device)

    def update(self, state: ClientState, client_dataset, config: ClientConfig, rng):
        """
        Loads weight tensors into models, trains on client_dataset, and returns updated state weights.
        Supports compute_optimization_mode ('shared_backbone', 'frozen_root_anchor', 'head_only', 'none')
        and inter-model mutual distillation.
        """
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
        # PATH 1: SHARED BACKBONE MULTI-HEAD OPTIMIZATION
        # ---------------------------------------------------------------------
        if compute_mode == "shared_backbone" and (use_ensemble or train_parent):
            multi_model = self.multihead_model
            # Load weights into multi-head model
            torch.nn.utils.vector_to_parameters(state.weights.to(self.device), multi_model.parameters())
            
            # Load parent/local head weights if explicitly saved in state
            if state.parent_weights is not None and len(state.parent_weights) == len(state.weights):
                parent_temp = SimpleCNN(in_channels=self.multihead_model.in_channels).to(self.device)
                torch.nn.utils.vector_to_parameters(state.parent_weights.to(self.device), parent_temp.parameters())
                multi_model.fc2_parent.load_state_dict(parent_temp.fc2.state_dict())
                
            if state.local_weights is not None and len(state.local_weights) == len(state.weights):
                local_temp = SimpleCNN(in_channels=self.multihead_model.in_channels).to(self.device)
                torch.nn.utils.vector_to_parameters(state.local_weights.to(self.device), local_temp.parameters())
                multi_model.fc2_local.load_state_dict(local_temp.fc2.state_dict())

            multi_model.train()
            optimizer = torch.optim.SGD(multi_model.parameters(), lr=config.local_lr)

            for epoch in range(epochs):
                for images, labels in dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    if is_byz and byz_type == "label_flip":
                        labels = 9 - labels

                    optimizer.zero_grad()
                    logits_root, logits_parent, logits_local = multi_model(images, head="all")
                    
                    loss_root = self.criterion(logits_root, labels)
                    loss_parent = self.criterion(logits_parent, labels)
                    loss_local = self.criterion(logits_local, labels)
                    
                    if is_byz and byz_type == "gradient_ascent":
                        total_loss = -(loss_root + loss_parent + loss_local)
                    else:
                        total_loss = loss_root + loss_parent + loss_local

                    if do_distill:
                        # Soft target ensemble predictions
                        soft_target = (F.softmax(logits_root.detach(), dim=1) + 
                                       F.softmax(logits_parent.detach(), dim=1) + 
                                       F.softmax(logits_local.detach(), dim=1)) / 3.0
                        
                        kd_loss = (F.kl_div(F.log_softmax(logits_local, dim=1), soft_target, reduction='batchmean') +
                                   F.kl_div(F.log_softmax(logits_parent, dim=1), soft_target, reduction='batchmean') +
                                   F.kl_div(F.log_softmax(logits_root, dim=1), soft_target, reduction='batchmean')) / 3.0
                        total_loss += distill_lambda * kd_loss

                    total_loss.backward()
                    optimizer.step()

            # Save updated weights back to state
            full_weights = torch.nn.utils.parameters_to_vector(multi_model.parameters()).detach().cpu()
            state.weights = full_weights
            
            # Export individual head states as 1D tensors for aggregation compatibility
            state.parent_weights = full_weights.clone()
            state.local_weights = full_weights.clone()
            return state

        # ---------------------------------------------------------------------
        # PATH 2: INDEPENDENT / FROZEN / HEAD-ONLY MODEL TRAINING
        # ---------------------------------------------------------------------
        # Setup Global Model
        global_model = self.global_model
        torch.nn.utils.vector_to_parameters(state.weights.to(self.device), global_model.parameters())
        
        if compute_mode == "frozen_root_anchor":
            global_model.eval()
            for p in global_model.parameters():
                p.requires_grad = False
            global_optimizer = None
        else:
            global_model.train()
            if compute_mode == "head_only":
                params = list(global_model.fc2.parameters())
            else:
                params = list(global_model.parameters())
            global_optimizer = torch.optim.SGD(params, lr=config.local_lr)

        # Setup Local Model
        local_model = self.local_model
        local_optimizer = None
        if use_ensemble:
            if state.local_weights is not None:
                torch.nn.utils.vector_to_parameters(state.local_weights.to(self.device), local_model.parameters())
            else:
                torch.nn.utils.vector_to_parameters(state.weights.to(self.device), local_model.parameters())
            local_model.train()
            if compute_mode == "head_only":
                params = list(local_model.fc2.parameters())
            else:
                params = list(local_model.parameters())
            local_optimizer = torch.optim.SGD(params, lr=config.local_lr)

        # Setup Parent Model
        parent_model = self.parent_model
        parent_optimizer = None
        if train_parent:
            if state.parent_weights is not None:
                torch.nn.utils.vector_to_parameters(state.parent_weights.to(self.device), parent_model.parameters())
            else:
                torch.nn.utils.vector_to_parameters(state.weights.to(self.device), parent_model.parameters())
            parent_model.train()
            if compute_mode == "head_only":
                params = list(parent_model.fc2.parameters())
            else:
                params = list(parent_model.parameters())
            parent_optimizer = torch.optim.SGD(params, lr=config.local_lr)

        for epoch in range(epochs):
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                if is_byz and byz_type == "label_flip":
                    labels = 9 - labels

                # Forward passes
                logits_root = global_model(images)
                loss_global = self.criterion(logits_root, labels)

                logits_local = local_model(images) if use_ensemble else None
                loss_local = self.criterion(logits_local, labels) if use_ensemble else 0.0

                logits_parent = parent_model(images) if train_parent else None
                loss_parent = self.criterion(logits_parent, labels) if train_parent else 0.0

                # Distillation loss
                if do_distill and use_ensemble and train_parent:
                    soft_target = (F.softmax(logits_root.detach(), dim=1) + 
                                   F.softmax(logits_parent.detach(), dim=1) + 
                                   F.softmax(logits_local.detach(), dim=1)) / 3.0
                    
                    loss_local += distill_lambda * F.kl_div(F.log_softmax(logits_local, dim=1), soft_target, reduction='batchmean')
                    loss_parent += distill_lambda * F.kl_div(F.log_softmax(logits_parent, dim=1), soft_target, reduction='batchmean')

                # Train Global Model
                if global_optimizer is not None:
                    global_optimizer.zero_grad()
                    if is_byz and byz_type == "gradient_ascent":
                        loss_global = -loss_global
                    loss_global.backward()
                    global_optimizer.step()

                # Train Local Model
                if use_ensemble and local_optimizer is not None:
                    local_optimizer.zero_grad()
                    if is_byz and byz_type == "gradient_ascent":
                        loss_local = -loss_local
                    loss_local.backward()
                    local_optimizer.step()

                # Train Parent Model
                if train_parent and parent_optimizer is not None:
                    parent_optimizer.zero_grad()
                    if is_byz and byz_type == "gradient_ascent":
                        loss_parent = -loss_parent
                    loss_parent.backward()
                    parent_optimizer.step()

        # Save weights back to state
        new_weights = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach().cpu()
        if is_byz:
            if byz_type == "sign_flip":
                new_weights = -new_weights
            elif byz_type == "random_noise":
                new_weights = torch.empty_like(new_weights).normal_(mean=0, std=10.0)
        state.weights = new_weights

        if use_ensemble:
            new_local_weights = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach().cpu()
            if is_byz:
                if byz_type == "sign_flip":
                    new_local_weights = -new_local_weights
                elif byz_type == "random_noise":
                    new_local_weights = torch.empty_like(new_local_weights).normal_(mean=0, std=10.0)
            state.local_weights = new_local_weights

        if train_parent:
            new_parent_weights = torch.nn.utils.parameters_to_vector(parent_model.parameters()).detach().cpu()
            if is_byz:
                if byz_type == "sign_flip":
                    new_parent_weights = -new_parent_weights
                elif byz_type == "random_noise":
                    new_parent_weights = torch.empty_like(new_parent_weights).normal_(mean=0, std=10.0)
            state.parent_weights = new_parent_weights

        return state
