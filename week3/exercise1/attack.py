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
    
    #The following shows one way of introducing some minor modification to 
    # the adversarial example, i.e, by adding some uniform noise. 
    x_adv = x.detach().clone()
    noise = torch.Tensor(np.random.uniform(-0.05, 0.05, (1,1,28,28)))
    x_adv = torch.clamp(x_adv + noise, 0, 1)

    #TODO1: Modify the scale of the noise in the above method and note the 
    #number of label changes (repeat each setting for at least 5 times and 
    # take the average);
    #TODO2: Invent another way of introducing modification and observe the 
    # label change in the same way.

    pred_adv = model(x_adv)
    y_adv = pred_adv.argmax(1)

    if y_adv != y:
        x = x.detach().numpy().reshape(-1)
        x_adv = x_adv.detach().numpy().reshape(-1)

        y, y_adv = y.item(), y_adv.item()

        #print('\nFound an adversarial sample!!!\n')

        #print('pred adv = {}'.format(pred_adv.detach().numpy().reshape(-1)))
        #print('lbl adv = {}\n'.format(y_adv))

        display(x, y, x_adv, y_adv)
        return True
    else:
        #print('\nCan\'t find adversarial samples!!!\n')
        return False


model = load_model(MNISTNet, './week3/exercise1/mnist.pt')
changes = 0

num_attack = 20
for i in range(num_attack):
    file_name = './week3/exercise1/toattack_adv/adv_data' + str(i) + '.txt'
    x = np.array(ast.literal_eval(open(file_name, 'r').readline()))
    x = torch.Tensor(x)

    pred = model(x)
    y = pred.argmax(1)
    print('pred img = {}'.format(pred.detach().numpy().reshape(-1)))

    if attack(model, x, y): changes += 1

    print('\n===========================\n')

print('Number of changed labels = {}\n'.format(changes))
