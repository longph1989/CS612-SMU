import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import ast


class CensusNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        output = x # cross entropy in pytorch already includes softmax
        return output


def load_model(model_class, name):
    model = model_class()
    model.load_state_dict(torch.load(name))

    return model


device = 'cpu'

model = load_model(CensusNet, './week6/exercise3/census.pt')
labels = np.array(ast.literal_eval(open('./week6/exercise3/census/data/labels.txt', 'r').readline()))
labels = torch.Tensor(labels).type(torch.LongTensor)

correct = 0
for i in range(1):
    file_name = './week6/exercise3/census/data/data' + str(i) + '.txt'
    x = np.array(ast.literal_eval(open(file_name, 'r').readline()))
    x = x.reshape(1, 13)
    x = torch.Tensor(x)

    print('Sample x is: {}'.format(x[0].tolist()))
    x.requires_grad = True

    pred_x = model(x)
    print('Prediction of x is: {}'.format(pred_x[0].tolist()))
    print('Label of x is: {}'.format(torch.argmax(pred_x).item()))
    loss = F.cross_entropy(pred_x, labels)
    loss.backward()

    print('Gradient of x: {}'.format(x.grad.data.sign()[0].tolist()))

    x1 = torch.clone(x)
    x1 = x1 + x.grad.data.sign()
    print('Updated sample x1 is: {}'.format(x1[0].tolist()))

    #x1.requires_grad = True
    pred_x1 = model(x1)
    print('Predication of x1 is {}'.format(pred_x1[0].tolist()))
    print('Label of x1 is {}'.format(torch.argmax(pred_x1).item()))

    loss1 = F.cross_entropy(pred_x1, labels)
    loss1.backward()

    print('Gradient of x1: {}'.format(x1.grad.data))

