import torch as torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


dataset = torch.load("/data/trial_7.dat")


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

    def optimize(self, dataset):
        print("stuff")
        # optimize dataset and stuff


model = testNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01, momentum = 0.9)

#variables and things
epochs = 1032342394
split = 0.25
bs = 16

test_loss = []
train_loss = []

# load data and do things here
trainLoader = DataLoader(dataset[:int(split * len(dataset))], batch_size = bs, shuffle = True)
testLoader = DataLoader(dataset[int(split * len(dataset)):], batch_size = bs, shuffle = True)

for epoch in range(epochs):

    train_loss = 0
    test_loss = 0

    for i, (data, target) in enumerate(trainLoader):

        optimizer.zero_grad()
        predict = model(data)
        loss = criterion(data, target)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
