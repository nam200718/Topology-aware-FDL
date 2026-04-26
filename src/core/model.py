import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network designed for MNIST.
    Keeps parameter count low for fast CPU/GPU simulation while learning effectively.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 1 input image channel, 16 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        # 32 channels * 4 * 4 spatial dimension after pooling twice from 28x28
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x shape for MNIST: (batch_size, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4) # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
