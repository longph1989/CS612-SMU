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

model = load_model(CensusNet, './week1/exercise4/census.pt')
labels = np.array(ast.literal_eval(open('./week1/exercise4/census/data/labels.txt', 'r').readline()))

correct = 0
for i in range(30):
    file_name = './week1/exercise4/census/data/data' + str(i) + '.txt'
    x = np.array(ast.literal_eval(open(file_name, 'r').readline()))
    x = x.reshape(1, 13)
    x = torch.Tensor(x)
    
    #the following code compares the model prediction to the groudth truth label to get the accuracy
    #TODO: change the code so that it compares the prediction before and after changing the gender feature (the 9th feature)
    if np.argmax(model(x).detach().numpy().reshape(-1)) == labels[i]:
        correct += 1

print('accuracy = {:.2f}%'.format(correct / 30 * 100))
