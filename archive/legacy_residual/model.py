"""
Legacy Additive Hierarchical Residual Parameterization Models (HRC / H-ResFL).
Archived for historical reference and backward compatibility.
The canonical FedHEP architecture uses MultiHeadResNet9 and MultiHeadMobileNetV3Small.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.model import conv_block


class HierarchicalResidualLinear(nn.Module):
    """
    Hierarchical Residual Classifier (HRC) Linear Head.
    Computes logits via a single-pass additive residual projection:
        W_eff = W_global + Delta_W_cluster + Delta_W_local
        b_eff = b_global + Delta_b_cluster + Delta_b_local
        z = x @ W_eff.T + b_eff

    Zero-initialization of residuals:
        W_global ~ Kaiming Uniform
        Delta_W_cluster = 0
        Delta_W_local = 0
    """
    def __init__(self, in_features: int, num_classes: int, bias: bool = True, mode: str = "linear", scale: float = 16.0):
        super(HierarchicalResidualLinear, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.use_bias = bias
        self.mode = mode
        self.scale = scale

        # Tier 1: Global Consensus
        self.weight_global = nn.Parameter(torch.empty(num_classes, in_features))
        if bias:
            self.bias_global = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter('bias_global', None)

        # Tier 2: Cluster Residual (Zero-initialized)
        self.weight_cluster = nn.Parameter(torch.zeros(num_classes, in_features))
        if bias:
            self.bias_cluster = nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_parameter('bias_cluster', None)

        # Tier 3: Local Residual (Zero-initialized)
        self.weight_local = nn.Parameter(torch.zeros(num_classes, in_features))
        if bias:
            self.bias_local = nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_parameter('bias_local', None)

        self.reset_global_parameters()

    def reset_global_parameters(self):
        nn.init.kaiming_uniform_(self.weight_global, a=5**0.5)
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_global)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_global, -bound, bound)

    def get_effective_weights(self):
        w_eff = self.weight_global + self.weight_cluster + self.weight_local
        b_eff = (self.bias_global + self.bias_cluster + self.bias_local) if self.use_bias else None
        return w_eff, b_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff, b_eff = self.get_effective_weights()
        if self.mode == "cosine":
            x_norm = F.normalize(x, p=2, dim=-1)
            w_norm = F.normalize(w_eff, p=2, dim=-1)
            return self.scale * F.linear(x_norm, w_norm)
        return F.linear(x, w_eff, b_eff)


class HierarchicalLoRALinear(nn.Module):
    """
    Hierarchical Low-Rank Adaptation (H-LoRA) Linear Layer for Foundation Models.
    W_eff = W_frozen + (alpha / r) * (B_global @ A_global + B_cluster @ A_cluster + B_local @ A_local)
    """
    def __init__(self, in_features: int, out_features: int, r: int = 4, lora_alpha: float = 8.0):
        super(HierarchicalLoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r

        # Frozen base weights
        self.weight_frozen = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight_frozen, a=5**0.5)

        # Tier 1: Global LoRA
        self.lora_A_global = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B_global = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A_global, a=5**0.5)

        # Tier 2: Cluster LoRA (Zero-initialized)
        self.lora_A_cluster = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B_cluster = nn.Parameter(torch.zeros(out_features, r))

        # Tier 3: Local LoRA (Zero-initialized)
        self.lora_A_local = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B_local = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight_frozen)
        delta = (
            self.lora_B_global @ self.lora_A_global
            + self.lora_B_cluster @ self.lora_A_cluster
            + self.lora_B_local @ self.lora_A_local
        )
        return out + self.scaling * F.linear(x, delta)


class HierarchicalResidualResNet9(nn.Module):
    """
    Streamlined Hierarchical Residual ResNet-9.
    Shares the convolutional feature extractor with a single additive HierarchicalResidualLinear classifier.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_channels: int = 32):
        super(HierarchicalResidualResNet9, self).__init__()
        self.in_channels = in_channels
        c = base_channels
        self.prep = conv_block(in_channels, c)
        self.layer1 = conv_block(c, c * 2, pool=True)
        self.res1 = nn.Sequential(conv_block(c * 2, c * 2), conv_block(c * 2, c * 2))
        
        self.layer2 = conv_block(c * 2, c * 4, pool=True)
        self.layer3 = conv_block(c * 4, c * 8, pool=True)
        self.res2 = nn.Sequential(conv_block(c * 8, c * 8), conv_block(c * 8, c * 8))
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = HierarchicalResidualLinear(c * 8, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prep(x)
        out = self.layer1(out)
        out = self.res1(out) + out
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.res2(out) + out
        out = self.pool(out)
        return out.view(out.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)


class HierarchicalResidualMobileNetV3Small(nn.Module):
    """
    Streamlined Hierarchical Residual MobileNetV3-Small.
    Shares the depthwise-separable convolutional backbone with a HierarchicalResidualLinear classifier.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(HierarchicalResidualMobileNetV3Small, self).__init__()
        import torchvision.models as models
        base_mobilenet = models.mobilenet_v3_small(num_classes=num_classes)
        self.features = base_mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = HierarchicalResidualLinear(576, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)
