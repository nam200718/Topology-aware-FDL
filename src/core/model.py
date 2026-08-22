import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network designed for MNIST and CIFAR-10.
    Keeps parameter count low for fast CPU/GPU simulation while learning effectively.
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc_input_dim = 32 * 4 * 4 if in_channels == 1 else 32 * 5 * 5
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MultiHeadSimpleCNN(nn.Module):
    """
    A multi-head Convolutional Neural Network sharing a single feature backbone
    with three specialized classification heads (Root, Parent, Local).
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(MultiHeadSimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc_input_dim = 32 * 4 * 4 if in_channels == 1 else 32 * 5 * 5
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        
        self.fc2_root = nn.Linear(128, num_classes)
        self.fc2_parent = nn.Linear(128, num_classes)
        self.fc2_local = nn.Linear(128, num_classes)

    def extract_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc_input_dim)
        features = F.relu(self.fc1(x))
        return features

    def forward(self, x, head="root"):
        features = self.extract_features(x)
        if head == "root":
            return self.fc2_root(features)
        elif head == "parent":
            return self.fc2_parent(features)
        elif head == "local":
            return self.fc2_local(features)
        elif head == "all":
            logits_root = self.fc2_root(features)
            logits_parent = self.fc2_parent(features)
            logits_local = self.fc2_local(features)
            return logits_root, logits_parent, logits_local
        else:
            raise ValueError(f"Unknown head '{head}' specified.")


# --- ResNet9 Architecture ---

def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    """
    Standard lightweight 9-layer Residual Network (ResNet9) for CIFAR-10 FL benchmark experiments.
    Supports channel width scaling (base_channels=32) for fast CPU/GPU simulation.
    """
    def __init__(self, in_channels=3, num_classes=10, base_channels=32):
        super(ResNet9, self).__init__()
        self.in_channels = in_channels
        c = base_channels
        self.prep = conv_block(in_channels, c)
        self.layer1 = conv_block(c, c * 2, pool=True)
        self.res1 = nn.Sequential(conv_block(c * 2, c * 2), conv_block(c * 2, c * 2))
        
        self.layer2 = conv_block(c * 2, c * 4, pool=True)
        self.layer3 = conv_block(c * 4, c * 8, pool=True)
        self.res2 = nn.Sequential(conv_block(c * 8, c * 8), conv_block(c * 8, c * 8))
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(c * 8, num_classes)

    def extract_features(self, x):
        out = self.prep(x)
        out = self.layer1(out)
        out = self.res1(out) + out
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.res2(out) + out
        out = self.pool(out)
        return out.view(out.size(0), -1)

    def forward(self, x):
        features = self.extract_features(x)
        return self.fc2(features)


class MultiHeadResNet9(nn.Module):
    """
    Multi-Head version of ResNet9. Shares the entire ResNet9 residual backbone
    with three specialized linear classification heads (Root, Parent, Local).
    Supports channel width scaling (base_channels=32) for fast simulation.
    """
    def __init__(self, in_channels=3, num_classes=10, base_channels=32):
        super(MultiHeadResNet9, self).__init__()
        self.in_channels = in_channels
        c = base_channels
        self.prep = conv_block(in_channels, c)
        self.layer1 = conv_block(c, c * 2, pool=True)
        self.res1 = nn.Sequential(conv_block(c * 2, c * 2), conv_block(c * 2, c * 2))
        
        self.layer2 = conv_block(c * 2, c * 4, pool=True)
        self.layer3 = conv_block(c * 4, c * 8, pool=True)
        self.res2 = nn.Sequential(conv_block(c * 8, c * 8), conv_block(c * 8, c * 8))
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc2_root = nn.Linear(c * 8, num_classes)
        self.fc2_parent = nn.Linear(c * 8, num_classes)
        self.fc2_local = nn.Linear(c * 8, num_classes)

    def extract_features(self, x):
        out = self.prep(x)
        out = self.layer1(out)
        out = self.res1(out) + out
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.res2(out) + out
        out = self.pool(out)
        return out.view(out.size(0), -1)

    def forward(self, x, head="root"):
        features = self.extract_features(x)
        if head == "root":
            return self.fc2_root(features)
        elif head == "parent":
            return self.fc2_parent(features)
        elif head == "local":
            return self.fc2_local(features)
        elif head == "all":
            logits_root = self.fc2_root(features)
            logits_parent = self.fc2_parent(features)
            logits_local = self.fc2_local(features)
            return logits_root, logits_parent, logits_local
        else:
            raise ValueError(f"Unknown head '{head}' specified.")



# --- MobileNetV3 Architecture ---

class MobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small architecture for edge vision benchmarks.
    Uses depthwise-separable convolutions and squeeze-and-excitation blocks.
    """
    def __init__(self, in_channels=3, num_classes=10):
        super(MobileNetV3Small, self).__init__()
        import torchvision.models as models
        base_mobilenet = models.mobilenet_v3_small(num_classes=num_classes)
        self.features = base_mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = base_mobilenet.classifier

    def extract_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)


class MultiHeadMobileNetV3Small(nn.Module):
    """
    Multi-Head MobileNetV3-Small for Hierarchical Ensemble Personalization (HEP).
    Shares the depthwise-separable convolutional backbone (576-dim latent embeddings)
    with three specialized classification heads (Root, Parent, Local).
    """
    def __init__(self, in_channels=3, num_classes=10):
        super(MultiHeadMobileNetV3Small, self).__init__()
        import torchvision.models as models
        base_mobilenet = models.mobilenet_v3_small(num_classes=num_classes)
        self.features = base_mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3-tier specialized classification heads
        self.classifier_root = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )
        self.classifier_parent = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )
        self.classifier_local = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )

    def extract_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    # Head aliases so engine/updater paths can address tiers uniformly
    @property
    def fc2_root(self):
        return self.classifier_root

    @property
    def fc2_parent(self):
        return self.classifier_parent

    @property
    def fc2_local(self):
        return self.classifier_local

    def forward(self, x, head="root"):
        features = self.extract_features(x)
        if head == "root":
            return self.classifier_root(features)
        elif head == "parent":
            return self.classifier_parent(features)
        elif head == "local":
            return self.classifier_local(features)
        elif head == "all":
            logits_root = self.classifier_root(features)
            logits_parent = self.classifier_parent(features)
            logits_local = self.classifier_local(features)
            return logits_root, logits_parent, logits_local
        else:
            raise ValueError(f"Unknown head '{head}' specified.")


def model_to_vector(model: nn.Module) -> torch.Tensor:
    """Serialize all model state (params + buffers like BatchNorm statistics) to a flat vector."""
    with torch.no_grad():
        tensors = [v.data.reshape(-1).float() for v in model.state_dict().values()]
        return torch.cat(tensors)


def vector_to_model(vector: torch.Tensor, model: nn.Module) -> None:
    """Deserialize a flat vector back into a model's state_dict (params + buffers) in-place."""
    with torch.no_grad():
        offset = 0
        for val in model.state_dict().values():
            numel = val.numel()
            val.copy_(vector[offset:offset + numel].reshape(val.shape))
            offset += numel


# --- Streamlined Hierarchical Residual Classifier (HRC / Streamlined HEP) ---

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
    def __init__(self, in_features: int, num_classes: int, bias: bool = True):
        super(HierarchicalResidualLinear, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.use_bias = bias

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
        return F.linear(x, w_eff, b_eff)


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




