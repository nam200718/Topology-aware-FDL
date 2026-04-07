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

    def update(self, state: ClientState, client_dataset, config: ClientConfig, rng):
        """
        Loads the 1D weight tensor into the model, trains on client_dataset, and returns new weights.
        """
        model = SimpleCNN().to(self.device)
        # Load weights from the flat tensor in state
        torch.nn.utils.vector_to_parameters(state.weights, model.parameters())

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.local_lr)
        
        # Determine batch size, cap at config or dataset len
        batch_size = min(32, len(client_dataset) if len(client_dataset) > 0 else 32)
        
        if batch_size > 0:
            dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
            
            # Train for specified number of steps or epochs
            # Here config.local_steps corresponds to local epochs
            epochs = getattr(config, "local_steps", 1)
            
            is_byz = getattr(state, "is_byzantine", False)
            byz_type = getattr(state, "byzantine_type", "label_flip")

            for epoch in range(epochs):
                for images, labels in dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    if is_byz and byz_type == "label_flip":
                        # Mathematical invert digits: 0 becomes 9, 1 becomes 8...
                        labels = 9 - labels

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = self.criterion(outputs, labels)
                    
                    if is_byz and byz_type == "gradient_ascent":
                        # Invert loss so SGD maximizes error
                        loss = -loss
                        
                    loss.backward()
                    optimizer.step()

        # Extract weights back to 1D tensor
        new_weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
        
        # Post-training attacks
        if is_byz:
            if byz_type == "sign_flip":
                new_weights = -new_weights
            elif byz_type == "random_noise":
                # Supply purely randomized Gaussian values scaled aggressively
                new_weights = torch.randn_like(new_weights) * 10.0
                
        state.weights = new_weights
        
        return state
