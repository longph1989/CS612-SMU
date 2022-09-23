from math import fabs
from operator import truediv
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

def isDiscriminative(x, model):
    print('Sample x is: {}'.format(x[0].tolist()))
    x.requires_grad = True
    pred_x = model(x)
    #print('Prediction of x is: {}'.format(pred_x[0].tolist()))
    #print('Label of x is: {}'.format(torch.argmax(pred_x).item()))

    xp = x.detach().clone()
    xp[0][8] = 1-xp[0][8]
    print('Sample xp is: {}'.format(xp[0].tolist()))
    pred_xp = model(xp)
    #print('Prediction of xp is: {}'.format(pred_xp[0].tolist()))
    #print('Label of xp is: {}'.format(torch.argmax(pred_xp).item()))
    if torch.argmax(pred_x).item() != torch.argmax(pred_xp).item():
        print('This sample is discriminary.')
        return True
    else:
        print('This sample is NOT discriminary.')
        return False 

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

    #The following captures how the global search works
    while isDiscriminative(x,model)==False:
        x.requires_grad = True
        pred_x = model(x)
        loss = F.cross_entropy(pred_x, labels)
        loss.backward()
        print('Gradient of x: {}'.format(x.grad.data.sign()[0].tolist()))
        x = x.detach().clone()
        x = x + x.grad.data.sign()

    print("Discriminative instance found.")
    
    #The following examplifies how local search works
    print('Sample x is: {}'.format(x[0].tolist()))
    x.requires_grad = True
    pred_x = model(x)
    loss = F.cross_entropy(pred_x, labels)
    loss.backward()
    grad = x.grad.data
    print('Gradient of x: {}'.format(grad.tolist()))    
    x = x.detach().clone()
    #TODO: Based on the above-printed grad, update x according to the idea of the local search to find one discirminary instance. 
    #TODO: Your code goes here. Only one/two lines are needed.


    isDiscriminative(x, model)