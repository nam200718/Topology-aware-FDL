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

