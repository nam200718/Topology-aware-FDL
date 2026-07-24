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
        # 16 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        # MNIST (in_channels=1): 32 channels * 4 * 4 spatial dimension after pooling twice from 28x28
        # CIFAR-10 (in_channels=3): 32 channels * 5 * 5 spatial dimension after pooling twice from 32x32
        self.fc_input_dim = 32 * 4 * 4 if in_channels == 1 else 32 * 5 * 5
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc_input_dim) # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MultiHeadSimpleCNN(nn.Module):
    """
    A multi-head Convolutional Neural Network sharing a single feature backbone
    with three specialized classification heads (Root, Parent, Local).
    Dramatically reduces compute/FLOPs for 3-tier ensemble FL on edge devices.
    Generalizes to multi-adapter / multi-head foundation models (e.g., Fed-Multi-LoRA).
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(MultiHeadSimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc_input_dim = 32 * 4 * 4 if in_channels == 1 else 32 * 5 * 5
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        
        # Three specialized classification heads sharing the backbone
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


