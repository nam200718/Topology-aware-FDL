"""
OrehLocalUpdater: Custom local updater extending PyTorchLocalUpdater with
ALIE attack support and targeted tier poisoning (root_only vs parent_only vs all).
"""

from typing import Optional, Literal
import torch

from src.core.interfaces import ClientState
from src.core.updater import PyTorchLocalUpdater
from oreh.alie_attack import generate_alie_delta, compute_alie_z_max


class OrehLocalUpdater(PyTorchLocalUpdater):
    """
    Subclasses PyTorchLocalUpdater to implement:
    1. Targeted Tier Byzantine attacks:
       - 'all': corrupts both root weights and parent weights (and local if present).
       - 'root_only': corrupts only root weights; parent weights remain clean.
       - 'parent_only': corrupts only parent weights; root weights remain clean.
    2. ALIE attack mode (Baruch et al., 2019):
       - Uses pre-computed ALIE deltas (from engine) or localized statistical perturbation.
    """

    def __init__(
        self,
        device="cpu",
        in_channels=3,
        model_name="resnet9",
        num_classes=10,
        attack_target_tier: Literal["all", "root_only", "parent_only"] = "all",
        alie_z_max: Optional[float] = None,
    ):
        super().__init__(
            device=device,
            in_channels=in_channels,
            model_name=model_name,
            num_classes=num_classes,
        )
        self.attack_target_tier = attack_target_tier
        self.alie_z_max = alie_z_max

    def _apply_byzantine_attack(
        self, state: ClientState, initial_weights: torch.Tensor
    ) -> ClientState:
        """
        Overrides PyTorchLocalUpdater._apply_byzantine_attack to support:
        - Targeted tiers ('all', 'root_only', 'parent_only')
        - ALIE stealthy model poisoning
        """
        is_byz = getattr(state, "is_byzantine", False)
        if not is_byz:
            return state

        byz_type = getattr(state, "byzantine_type", "label_flip")
        target_tier = getattr(state, "attack_target_tier", self.attack_target_tier)

        def attack_tensor(w: torch.Tensor, init_w: torch.Tensor, override_delta: Optional[torch.Tensor] = None) -> torch.Tensor:
            if init_w is None or len(w) != len(init_w):
                ref = w
            else:
                ref = init_w

            if override_delta is not None and len(override_delta) == len(ref):
                return ref + override_delta

            delta = w - ref
            if byz_type == "sign_flip":
                return ref - delta * 1.5
            elif byz_type == "gradient_ascent":
                return ref - delta * 5.0
            elif byz_type == "random_noise":
                return ref + torch.randn_like(ref) * 2.0
            elif byz_type == "alie":
                # Fallback if engine didn't inject pre-computed ALIE delta
                z = self.alie_z_max if self.alie_z_max is not None else 1.0
                sigma = torch.abs(delta) * 0.5 + 1e-6
                return ref + (delta - z * sigma)
            return w

        # Extract pre-computed ALIE deltas if provided by OrehDefendedEngine
        alie_root_delta = getattr(state, "alie_root_delta", None)
        alie_parent_delta = getattr(state, "alie_parent_delta", None)

        if byz_type in ("sign_flip", "gradient_ascent", "random_noise", "alie"):
            # 1. Target Root Weights
            if target_tier in ("all", "root_only"):
                state.weights = attack_tensor(state.weights, initial_weights, alie_root_delta)

            # 2. Target Parent Weights
            if target_tier in ("all", "parent_only") and state.parent_weights is not None:
                parent_ref = getattr(state, "parent_reference_weights", initial_weights)
                state.parent_weights = attack_tensor(state.parent_weights, parent_ref, alie_parent_delta)

            # 3. Local weights
            if target_tier == "all" and state.local_weights is not None:
                state.local_weights = attack_tensor(state.local_weights, initial_weights)

        return state

