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


class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2304, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = x # cross entropy in pytorch already includes softmax
        return output


def load_model(model_class, name):
    model = model_class()
    model.load_state_dict(torch.load(name))

    return model


def print_confidence(model, data):
    with torch.no_grad():
        for i in range(20):
            if data == 'train':
                file_name = './week1/exercise5/train/train' + str(i) + '.txt'
            elif data == 'test':
                file_name = './week1/exercise5/test/test' + str(i) + '.txt'

            # TODO: read data, convert to Tensor, apply the model then softmax function to get the confidence

model = load_model(CIFAR10Net, './week1/exercise5/cifar10.pt')

print('\n===================================\n')
print('Confidence with train data')
print_confidence(model, 'train')
print('\n===================================\n')

print('\n===================================\n')
print('Confidence with test data')
print_confidence(model, 'test')
print('\n===================================\n')
