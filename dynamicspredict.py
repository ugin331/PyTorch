import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import sklearn


file = '/data/trial_7.dat'
cur_dir = os.getcwd()
dataset = torch.load(cur_dir+file)


class testNet(nn.Module):
    def __init__(self, n_in, n_hidden1, n_hidden2, n_hidden3, n_predict):
        super(testNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_in, n_hidden1),
            nn.Linear(n_hidden1, n_hidden2),
            nn.Linear(n_hidden2, n_hidden3),
            nn.Linear(n_hidden3, n_predict)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, self.cnntolinearshape)
        x = self.linear(x)
        return x

    def optimize(self, dataset):
        print("stuff")
        # optimize dataset and stuff


model = testNet(torch.numel(dataset[0]), 100, 20, 10, torch.numel(dataset[0]))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01, momentum = 0.9)

#variables and things
epochs = 10
split = 0.25
bs = 16

test_errors = []
train_errors = []

# load data and do things here
trainLoader = DataLoader(dataset[:int(split * len(dataset))], batch_size=bs, shuffle=True)
testLoader = DataLoader(dataset[int(split * len(dataset)):], batch_size=bs, shuffle=True)

for epoch in range(epochs):

    # testing and training loss
    train_error = 0
    test_error = 0

    for i, (data, target) in enumerate(trainLoader):

        optimizer.zero_grad()
        predict = model(data)
        loss = criterion(data, target)
        train_error += loss.item()

        loss.backward()
        optimizer.step()

    for i, (inputs, targets) in enumerate(testLoader):
        outputs = model(inputs)
        loss = criterion(outputs.float(), targets.float())
        test_error += loss.item() / (len(testLoader))

    print(f"    Epoch {epoch + 1}, Train err: {train_error}, Test err: {test_error}")
    train_errors.append(train_error)
    test_errors.append(test_error)

# validation set here or somethign

