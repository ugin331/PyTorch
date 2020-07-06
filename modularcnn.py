import torch as torch
import torchvision
import torch.nn as nn
import numpy as np


class testNet(nn.Module):
    def __init__(self, n_in, layer1_conv_out, layer2_conv_out, conv_size, linear1_out, linear2_out, final_out):
        super(testNet, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_in, layer1_conv_out, conv_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(layer1_conv_out, layer2_conv_out, conv_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(layer2_conv_out * conv_size * conv_size, linear1_out),
            nn.ReLU(),
            nn.Linear(linear1_out, linear2_out),
            nn.ReLU(),
            nn.Linear(linear2_out, final_out)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.linear(x)
        return x