import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network designed for MNIST and CIFAR-10.
    Keeps parameter count low for fast CPU/GPU simulation while learning effectively.
    """
    def __init__(self, in_channels=1):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc_input_dim = 32 * 4 * 4 if in_channels == 1 else 32 * 5 * 5
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 10)

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


