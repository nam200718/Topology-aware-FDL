"""
OrehLocalUpdater: Custom local updater extending PyTorchLocalUpdater with
targeted tier poisoning, stealthy ALIE/IPM perturbations, Adaptive Cosine Attack,
and Delayed Onset attacks.
"""

from typing import Optional, Literal
import torch

from src.core.interfaces import ClientState
from src.core.updater import PyTorchLocalUpdater
from oreh.attacks import compute_alie_z_max, generate_alie_delta, generate_ipm_delta, generate_adaptive_cosine_delta


class OrehLocalUpdater(PyTorchLocalUpdater):
    """
    Subclasses PyTorchLocalUpdater to implement:
    1. Targeted Tier Byzantine attacks:
       - 'all': corrupts both root weights and parent weights (and local if present).
       - 'root_only': corrupts only root weights; parent weights remain clean.
       - 'parent_only': corrupts only parent weights; root weights remain clean.
    2. Multi-Vector Attack Modes:
       - 'sign_flip', 'gradient_ascent', 'random_noise'
       - 'alie': A Little Is Enough (Baruch et al., 2019)
       - 'ipm': Inner Product Manipulation (Xie et al., 2020)
       - 'adaptive_cosine': Trust-adaptive attack targeting TTT
       - 'delayed_onset': Honeymoon phase followed by attack targeting TTT memory
    """

    def __init__(
        self,
        device="cpu",
        in_channels=3,
        model_name="resnet9",
        num_classes=10,
        attack_target_tier: Literal["all", "root_only", "parent_only"] = "all",
        alie_z_max: Optional[float] = None,
        ipm_epsilon: float = 0.1,
        attack_onset_round: int = 15,
    ):
        super().__init__(
            device=device,
            in_channels=in_channels,
            model_name=model_name,
            num_classes=num_classes,
        )
        self.attack_target_tier = attack_target_tier
        self.alie_z_max = alie_z_max
        self.ipm_epsilon = ipm_epsilon
        self.attack_onset_round = attack_onset_round

    def _apply_byzantine_attack(
        self, state: ClientState, initial_weights: torch.Tensor
    ) -> ClientState:
        """
        Overrides PyTorchLocalUpdater._apply_byzantine_attack to support:
        - Targeted tiers ('all', 'root_only', 'parent_only')
        - Stealthy ALIE, IPM, Adaptive Cosine, and Delayed Onset attacks
        """
        is_byz = getattr(state, "is_byzantine", False)
        if not is_byz:
            return state

        byz_type = getattr(state, "byzantine_type", "label_flip")
        target_tier = getattr(state, "attack_target_tier", self.attack_target_tier)

        # Check Delayed Onset condition
        if byz_type == "delayed_onset":
            curr_round = getattr(state, "current_round", 0)
            onset_round = getattr(state, "attack_onset_round", self.attack_onset_round)
            if curr_round < onset_round:
                # Honeymoon phase: act completely benign to accumulate maximum trust
                return state

        def attack_tensor(
            w: torch.Tensor,
            init_w: torch.Tensor,
            override_delta: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if init_w is None or len(w) != len(init_w):
                ref = w
            else:
                ref = init_w

            if override_delta is not None and len(override_delta) == len(ref):
                return ref + override_delta

            delta = w - ref
            if byz_type in ("sign_flip", "delayed_onset"):
                return ref - delta * 1.5
            elif byz_type == "gradient_ascent":
                return ref - delta * 5.0
            elif byz_type == "random_noise":
                return ref + torch.randn_like(ref) * 2.0
            elif byz_type == "alie":
                z = self.alie_z_max if self.alie_z_max is not None else 1.0
                sigma = torch.abs(delta) * 0.5 + 1e-6
                return ref + (delta - z * sigma)
            elif byz_type == "ipm":
                # Fallback: single-client IPM approximation
                eps = self.ipm_epsilon
                rand_v = torch.randn_like(delta)
                rand_perp = rand_v - (torch.dot(rand_v, delta) / (delta.norm()**2 + 1e-8)) * delta
                rand_perp = rand_perp / (rand_perp.norm() + 1e-8)
                u = eps * (delta / (delta.norm() + 1e-8)) - (1 - eps**2)**0.5 * rand_perp
                return ref + u * delta.norm()
            elif byz_type == "adaptive_cosine":
                prev_trust = getattr(state, "prev_trust_score", 1.0)
                atk_w = 0.85 if prev_trust > 0.7 else (0.5 if prev_trust > 0.4 else 0.05)
                return ref + (1.0 - atk_w) * delta - atk_w * torch.sign(delta) * torch.abs(delta)
            return w

        # Extract pre-computed deltas provided by OrehDefendedEngine / OrehCentralizedEngine
        custom_root_delta = getattr(state, "alie_root_delta", None)
        if custom_root_delta is None:
            custom_root_delta = getattr(state, "ipm_root_delta", None)
        if custom_root_delta is None:
            custom_root_delta = getattr(state, "adaptive_root_delta", None)

        custom_parent_delta = getattr(state, "alie_parent_delta", None)
        if custom_parent_delta is None:
            custom_parent_delta = getattr(state, "ipm_parent_delta", None)
        if custom_parent_delta is None:
            custom_parent_delta = getattr(state, "adaptive_parent_delta", None)

        recognized_attacks = (
            "sign_flip",
            "gradient_ascent",
            "random_noise",
            "alie",
            "ipm",
            "adaptive_cosine",
            "delayed_onset",
        )

        if byz_type in recognized_attacks:
            # 1. Target Root Weights
            if target_tier in ("all", "root_only"):
                state.weights = attack_tensor(state.weights, initial_weights, custom_root_delta)

            # 2. Target Parent Weights
            if target_tier in ("all", "parent_only") and state.parent_weights is not None:
                parent_ref = getattr(state, "parent_reference_weights", initial_weights)
                state.parent_weights = attack_tensor(state.parent_weights, parent_ref, custom_parent_delta)

            # 3. Local weights
            if target_tier == "all" and state.local_weights is not None:
                state.local_weights = attack_tensor(state.local_weights, initial_weights)

        return state
