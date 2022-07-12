# This is an implementation of C&W, a targeted attack


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


def get_loss(pred, y, target):
    mask = torch.eye(len(pred[0]))[target]
    i, _ = torch.max((1 - mask) * pred, dim=1) 
    t = torch.masked_select(pred, mask.bool())

    return torch.clamp(i - t, min=0.0)


def cw(model, x, y, eps, max_iter, c, target):
    target = torch.Tensor([target]).type(torch.LongTensor)
    if y == target:
        print('\nThe sample is already classified as the target!!! Skip the sample!!!\n')
        return False

    w = torch.zeros(x.size()).detach()
    w.requires_grad = True
    
    optimizer = optim.Adam([w], lr=0.01)

    for _ in range(max_iter):
        delta = 0.5 * (torch.tanh(w) + 1) - x
        l2_loss = torch.linalg.norm(torch.flatten(delta), 2)

        x_adv = torch.clamp(x + w, 0, 1)
        pred = model(x_adv)
        f_loss = get_loss(pred, y, target).sum()

        loss = l2_loss + c * f_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        x_adv = torch.clamp(x + w, 0, 1).detach()

        pred_adv = model(x_adv)
        y_adv = pred_adv.argmax(1)

        if y_adv == target:
            x = x.detach().numpy().reshape(-1)
            x_adv = x_adv.detach().numpy().reshape(-1)

            y, y_adv = y.item(), y_adv.item()

            print('\nFound an adversarial sample!!!\n')

            print('pred adv = {}'.format(pred_adv.detach().numpy().reshape(-1)))
            print('lbl adv = {}\n'.format(y_adv))

            display(x, y, x_adv, y_adv)
            return True

    print('\nCan\'t find adversarial samples!!!\n')
    return False


model = load_model(MNISTNet, 'mnist.pt')
num_adv, eps, max_iter, c, target = 0, 0.1, 100, 100, 0

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

    if cw(model, x, y, eps, max_iter, c, target): num_adv += 1

    print('\n===========================\n')

print('Adv imgs = {}\n'.format(num_adv))
