import torch as torch
import torchvision
import torch.nn as nn
import numpy as np


class testNet(nn.Module):
    def __init__(self, n_in, layer1_conv_out, ):
        super(testNet, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_in, layer1_conv_out, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.linear(x)
        return x