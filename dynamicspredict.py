import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import numpy as np
from collections import OrderedDict
import pandas as pd


def normalize_data(norm_tensor, tensor_max, tensor_min):
    for row in norm_tensor:
        for x in row:
            num = x.item()
            adjusted = (2 * (num - tensor_min) / (tensor_max - tensor_min)) - 1
            x = torch.tensor(adjusted)
            x = x.type(torch.float64)

    return norm_tensor


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
        actions = actions.type(torch.float64)
        deltas = torch.from_numpy(deltas)
        deltas = deltas.unsqueeze(1)
        states = torch.from_numpy(states)

        # concat states and deltas
        data = torch.cat((states, actions), dim=1)
        data_max = torch.max(data)
        data_max = data_max.item()
        data_min = torch.min(data)
        data_min = data_min.item()
        data = normalize_data(data, data_max, data_min)
        targets = deltas
        targets = normalize_data(targets, data_max, data_min)

        self.data = []
        for i in range(0, len(data)):
            temp = [data[i], targets[i]]
            self.data.append(temp)

        # print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        return x, y


class testNet(nn.Module):
    def __init__(self, n_in, hidden_w, depth, n_out):
        super(testNet, self).__init__()
        self.n_in = n_in
        self.hidden_w = hidden_w
        self.depth = depth
        self.activation = nn.ReLU()
        self.n_out = n_out
        layers = []
        layers.append(('dynm_input_lin', nn.Linear(self.n_in, self.hidden_w)))
        layers.append(('dynm_input_act', self.activation))
        for d in range(self.depth):
            layers.append(('dynm_lin_' + str(d), nn.Linear(self.hidden_w, self.hidden_w)))
            layers.append(('dynm_act_' + str(d), self.activation))

        layers.append(('dynm_out_lin', nn.Linear(self.hidden_w, self.n_out)))
        self.features = nn.Sequential(OrderedDict([*layers]))


    def forward(self, x):
        x = self.features(x)
        return x

    def optimize(self, dataset):
        print("stuff")
        # optimize dataset and stuff


def my_collate(batch):
    data = []
    target = []
    for item in batch:
        data_item = item[0].tolist()
        data.append(data_item)
        target_item = item[1].tolist()
        target.append(target_item)

    data = torch.FloatTensor(data)
    target = torch.FloatTensor(target)

    return data, target


# variables and things
epochs = 100
split = 0.7
bs = 16
lr = 0.2

dataset = simdata(root_dir='./data/simdata')
train_set, test_set = random_split(dataset, [int(split*len(dataset)), int((1-split)*len(dataset)+1)])

train_set_len = int(split*len(dataset))

model = testNet(10, 100, 10, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)

test_errors = []
train_errors = []

# load data and do things here
trainLoader = DataLoader(train_set, batch_size=bs, shuffle=True, collate_fn=my_collate)
testLoader = DataLoader(test_set, batch_size=bs, shuffle=True, collate_fn=my_collate)

for epoch in range(epochs):

    # testing and training loss
    train_error = 0
    test_error = 0

    for i, data in enumerate(trainLoader):

        inputs, target = data
        inputs = Variable(inputs, requires_grad=True)
        target = Variable(target, requires_grad=True)

        optimizer.zero_grad()
        predict = model(inputs)
        loss = criterion(inputs, target)
        train_error += loss.item() / len(trainLoader)

        loss.backward()
        optimizer.step()

    for i, data in enumerate(testLoader):

        inputs, targets = data

        outputs = model(inputs)
        loss = criterion(outputs.float(), targets.float())
        test_error += loss.item() / (len(testLoader))

    scheduler.step(test_error)

    print(f"Epoch {epoch + 1}, Train loss: {train_error}, Test loss: {test_error}")
    train_errors.append(train_error)
    test_errors.append(test_error)

correct = 0
total = 0
model.eval()  # prep model for testing

# validation set here or somethign

with torch.no_grad():
    for data, target in testLoader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print('Accuracy of the network on the test set: %d %%' % ((100 * correct / total)))
