import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.interfaces import ClientState
from src.config import ClientConfig
from src.core.model import SimpleCNN

class PyTorchLocalUpdater:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Reuse models to save overhead
        self.global_model = SimpleCNN().to(self.device)
        self.local_model = SimpleCNN().to(self.device)
        self.parent_model = SimpleCNN().to(self.device)

    def update(self, state: ClientState, client_dataset, config: ClientConfig, rng):
        """
        Loads the 1D weight tensor into the model, trains on client_dataset, and returns new weights.
        """
        use_ensemble = getattr(config, "use_ensemble", False)
        # Check if we should also train the parent model
        train_parent = getattr(config, "hierarchical_ensemble", False)
        
        # Setup Global Model
        global_model = self.global_model
        torch.nn.utils.vector_to_parameters(state.weights.to(self.device), global_model.parameters())
        global_model.train()
        global_optimizer = torch.optim.SGD(global_model.parameters(), lr=config.local_lr)

        # Setup Local Model (if ensemble)
        local_model = self.local_model
        local_optimizer = None
        if use_ensemble:
            if getattr(state, "local_weights", None) is not None:
                torch.nn.utils.vector_to_parameters(state.local_weights.to(self.device), local_model.parameters())
            else:
                # Initialize local weights with initial global weights on the first round
                torch.nn.utils.vector_to_parameters(state.weights.to(self.device), local_model.parameters())
            local_model.train()
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=config.local_lr)

        # Setup Parent Model (if hierarchical ensemble)
        parent_model = self.parent_model
        parent_optimizer = None
        if train_parent:
            if getattr(state, "parent_weights", None) is not None:
                torch.nn.utils.vector_to_parameters(state.parent_weights.to(self.device), parent_model.parameters())
            else:
                # Fallback to current weights if parent_weights not set
                torch.nn.utils.vector_to_parameters(state.weights.to(self.device), parent_model.parameters())
            parent_model.train()
            parent_optimizer = torch.optim.SGD(parent_model.parameters(), lr=config.local_lr)
        batch_size = min(32, len(client_dataset) if len(client_dataset) > 0 else 32)
        if batch_size > 0:
            dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
            epochs = getattr(config, "local_steps", 1)
            is_byz = getattr(state, "is_byzantine", False)
            byz_type = getattr(state, "byzantine_type", "label_flip")

            for epoch in range(epochs):
                for images, labels in dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    if is_byz and byz_type == "label_flip":
                        labels = 9 - labels

                    # Train Global Model
                    global_optimizer.zero_grad()
                    outputs_global = global_model(images)
                    loss_global = self.criterion(outputs_global, labels)
                    if is_byz and byz_type == "gradient_ascent":
                        loss_global = -loss_global
                    loss_global.backward()
                    global_optimizer.step()
                    
                    # Train Local Model
                    if use_ensemble:
                        local_optimizer.zero_grad()
                        outputs_local = local_model(images)
                        loss_local = self.criterion(outputs_local, labels)
                        if is_byz and byz_type == "gradient_ascent":
                            loss_local = -loss_local
                        loss_local.backward()
                        local_optimizer.step()

                    # Train Parent Model
                    if train_parent:
                        parent_optimizer.zero_grad()
                        outputs_parent = parent_model(images)
                        loss_parent = self.criterion(outputs_parent, labels)
                        if is_byz and byz_type == "gradient_ascent":
                            loss_parent = -loss_parent
                        loss_parent.backward()
                        parent_optimizer.step()

        # Extract weights back to 1D tensor
        new_weights = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach().cpu()
        if is_byz:
            if byz_type == "sign_flip":
                new_weights = -new_weights
            elif byz_type == "random_noise":
                new_weights = torch.randn_like(new_weights) * 10.0
        state.weights = new_weights
        
        if use_ensemble:
            new_local_weights = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach().cpu()
            if is_byz:
                if byz_type == "sign_flip":
                    new_local_weights = -new_local_weights
                elif byz_type == "random_noise":
                    new_local_weights = torch.randn_like(new_local_weights) * 10.0
            state.local_weights = new_local_weights

        if train_parent:
            new_parent_weights = torch.nn.utils.parameters_to_vector(parent_model.parameters()).detach().cpu()
            if is_byz:
                if byz_type == "sign_flip":
                    new_parent_weights = -new_parent_weights
                elif byz_type == "random_noise":
                    new_parent_weights = torch.randn_like(new_parent_weights) * 10.0
            state.parent_weights = new_parent_weights
        
        return state
