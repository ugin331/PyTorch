import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import OrderedDict


# data normalization
# swap to column based
def normalize_data(norm_tensor, col, tensor_max, tensor_min):
    sizetuple = norm_tensor.size()
    numrows = sizetuple[0]
    for i in range(0, numrows):
        # print(norm_tensor[i][col])
        norm_tensor[i][col] -= tensor_min
        norm_tensor[i][col] *= 2
        norm_tensor[i][col] /= (tensor_max - tensor_min)
        norm_tensor[i][col] -= 1
        # print(norm_tensor[i][col])
    return norm_tensor


# get min + max in tensor column for better normalization(tm)
def get_col_minmax(tensor, col):
    data_max = torch.min(tensor)
    data_max = data_max.item()
    data_min = torch.max(tensor)
    data_min = data_min.item()
    for row in tensor:
        if row[col].item() > data_max:
            data_max = row[col].item()
        if row[col].item() < data_min:
            data_min = row[col].item()
    # print("column:", col, "column max:", data_max, "column min: ", data_min)
    return data_max, data_min

def comp_tensor(predict, target, diffpercent):
    correct = 0
    num = torch.numel(predict)
    tempcorrect = 0
    for i in range(0, num):
        predict_val = predict[i].item()
        target_val = target[i].item()
        if abs(abs(target_val - predict_val) / target_val) <= diffpercent:
            tempcorrect += 1

    print(tempcorrect, "correct out of 6")
    if tempcorrect == 6:
        correct += 1
    return correct

# compare two tensors for validation
def compfunc(predict, target, diffpercent):
    correct = 0
    predict_copy = predict.numpy()
    len_outer = len(predict_copy)
    for i in range(0, len_outer):
        correct += comp_tensor(predict[i], target[i], diffpercent)
    return correct


# dataset class for simulated data
class simdata(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # load npy files
        actions = np.load(root_dir+'/actions.npy')
        states = np.load(root_dir+'/states.npy')

        # turn them to tensors
        actions = torch.from_numpy(actions)
        actions = actions.type(torch.float64)
        states = torch.from_numpy(states)
        deltas = states[1:, :] - states[:-1, :]

        # concat states and deltas
        data = torch.cat((states, actions), dim=1)
        data = data[:-1]
        targets = deltas
        # print(deltas.size())
        # print(data.size())

        numcols = data.size()[1]
        for column in range(0, numcols):
            data_max, data_min = get_col_minmax(data, column)
            data = normalize_data(data, column, data_max, data_min)

        numcols = targets.size()[1]
        for column in range(0, numcols):
            data_max, data_min = get_col_minmax(targets, column)
            targets = normalize_data(targets, column, data_max, data_min)

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


# neural network class
class testNet(nn.Module):
    def __init__(self, n_in, hidden_w, depth, n_out):
        super(testNet, self).__init__()
        self.n_in = n_in
        self.hidden_w = hidden_w
        self.depth = depth
        self.activation = nn.Softmax()
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


# custom collate function for dataset
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
epochs = 50
split = 0.7
bs = 16
lr = 0.05

dataset = simdata(root_dir='./data/simdata')
print("dataset length:")
print(len(dataset))
train_set, test_set = random_split(dataset, [int(split*len(dataset)), int((1-split)*len(dataset))])
print("train_set length:")
print(len(train_set))
print("test_set length:")
print(len(test_set))

#train_set_len = int(split*len(dataset))
model = testNet(10, 100, 2, 6)
print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)

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

        inputs, targets = data
        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets, requires_grad=True)

        optimizer.zero_grad()
        predict = model(inputs)
        loss = criterion(predict, targets)
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

train_legend = mlines.Line2D([], [], color='blue', label='Train Loss')
test_legend = mlines.Line2D([], [], color='orange', label='Test Loss')
plt.plot(train_errors, label="train loss")
plt.plot(test_errors, color="orange", label="test loss")
plt.legend(handles=[train_legend, test_legend])
plt.savefig('dynamicslossgraph.png')

correct = 0
total = 0
diffpercent = 0.05
model.eval()  # prep model for testing

# validation set here or somethign
# REWRITE THIS
with torch.no_grad():
    for data, targets in testLoader:
        #print(data)
        outputs = model(data)
        #print("targets")
        #print(targets)
        #print("outputs")
        #print(outputs)
        tgtcpy = targets.numpy()
        tgtlen = len(tgtcpy)
        total += tgtlen
        correct += compfunc(outputs, targets, diffpercent)


print('threshold is  %%%f' % ((diffpercent*100)))
print('Accuracy of the network on the test set: %d out of %d' % (correct, total))
print('Accuracy of the network on the test set: %d %%' % ((100 * correct / total)))
