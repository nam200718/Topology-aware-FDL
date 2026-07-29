import torch
from src.core.updater import PyTorchLocalUpdater

class ByzantineUpdater(PyTorchLocalUpdater):
    def update(self, *args, **kwargs):
        # Extract state safely from args or kwargs
        state = kwargs.get('state') if 'state' in kwargs else args[0]
        initial_weights = state.weights.clone()
        
        # Call original train function (Original function extracts is_byz from state)
        state = super().update(*args, **kwargs)
        
        # Apply intense byzantine attacks directly on weight space
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
