import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import sklearn

class simdata(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # load npy files
        actions = np.load(root_dir+'/actions.npy')
        deltas = np.load(root_dir+'/deltas.npy')
        states = np.load(root_dir+'/states.npy')

        # turn them to tensors
        actions = torch.from_numpy(actions)
        deltas = torch.from_numpy(deltas)
        deltas = deltas.unsqueeze(1)
        print(deltas.shape)
        states = torch.from_numpy(states)
        print(states.shape)

        # concat states and deltas
        self.data = torch.cat((states, deltas), dim=1)
        print(self.data.shape)
        self.targets = actions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

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
        x = self.linear(x)
        return x

    def optimize(self, dataset):
        print("stuff")
        # optimize dataset and stuff

dataset = simdata(root_dir = './data/simdata')
model = testNet(7, 100, 20, 10, 4)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# variables and things
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

    for i, data in enumerate(trainLoader):

        inputs, target = data

        optimizer.zero_grad()
        predict = model(inputs)
        loss = criterion(inputs, target)
        train_error += loss.item()

        loss.backward()
        optimizer.step()

    for i, data in enumerate(testLoader):

        inputs, targets = data

        outputs = model(inputs)
        loss = criterion(outputs.float(), targets.float())
        test_error += loss.item() / (len(testLoader))

    print(f"    Epoch {epoch + 1}, Train err: {train_error}, Test err: {test_error}")
    train_errors.append(train_error)
    test_errors.append(test_error)

# validation set here or somethign

