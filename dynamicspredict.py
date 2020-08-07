import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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


dataset = simdata(root_dir='./data/simdata')

model = testNet(10, 50, 3, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# variables and things
epochs = 100
split = 0.7
bs = 16

test_errors = []
train_errors = []

# train_set = dataset[:int(split * len(dataset))]
# print("train_set length:")
# print(len(train_set))
# print(train_set)
# test_set = dataset[int(split * len(dataset)):]

# load data and do things here
trainLoader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=my_collate)
testLoader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=my_collate)

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

