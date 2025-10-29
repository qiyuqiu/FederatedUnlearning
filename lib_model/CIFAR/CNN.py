import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dense = nn.Linear(64 * 8 * 8, 2048)
        self.out = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense(x))
        logits = self.out(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
