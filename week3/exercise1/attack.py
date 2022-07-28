# This is an implementation of FGSM, an untargeted attack


import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import numpy as np
import ast


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = x # cross entropy in pytorch already includes softmax
        return output


def load_model(model_class, name):
    model = model_class()
    model.load_state_dict(torch.load(name))

    return model


def denormalize(x):
    x = (x * 255).astype('uint8')
    x = x.reshape(28,28)

    return x


def display(x, y, x_adv, y_adv):
    x = denormalize(x)
    x_adv = denormalize(x_adv)

    fig, ax = plt.subplots(1, 2)

    ax[0].set(title='Original. Label is {}'.format(y))
    ax[1].set(title='Adv. sample. Label is {}'.format(y_adv))

    ax[0].imshow(x, cmap='gray')
    ax[1].imshow(x_adv, cmap='gray')
    
    plt.show()


def attack(model, x, y):
    # x is the image sample (Tensor), y is the original label (Tensor)
    pass


model = load_model(MNISTNet, 'mnist.pt')
num_adv = 0

labels = np.array(ast.literal_eval(open('./toattack/labels.txt', 'r').readline()))

num_attack = 5
for i in range(num_attack):
    file_name = './toattack/data' + str(i) + '.txt'
    x = np.array(ast.literal_eval(open(file_name, 'r').readline()))
    x = torch.Tensor(x)
    y = torch.Tensor([labels[i]]).type(torch.LongTensor)

    pred = model(x)
    print('pred img = {}'.format(pred.detach().numpy().reshape(-1)))
    print('lbl imp = {}\n'.format(y.item()))

    if attack(model, x, y): num_adv += 1

    print('\n===========================\n')

print('Adv imgs = {}\n'.format(num_adv))
