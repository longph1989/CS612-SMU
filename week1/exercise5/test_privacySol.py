import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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


def save_model(model, name):
    torch.save(model.state_dict(), name)


def load_model(model_class, name):
    model = model_class()
    model.load_state_dict(torch.load(name))

    return model


def print_model(model):
    for name, param in model.named_parameters():
        print(name)
        print(param.data)


def train(model, dataloader, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model.eval()
    test_loss, correct = 0, 0
    cnt, passed = 0, 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    print('passed = {}'.format(passed))
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def print_confidence(model, data):
    with torch.no_grad():
        for i in range(20):
            if data == 'train':
                file_name = './week1/exercise5/train/train' + str(i) + '.txt'
            elif data == 'test':
                file_name = './week1/exercise5/test/test' + str(i) + '.txt'

            # read data, convert to Tensor, apply the model then softmax function to get the confidence

            x = np.array(ast.literal_eval(open(file_name, 'r').readline()))
            x = torch.Tensor(x)

            pred = model(x)
            confidence = F.softmax(pred, 1).numpy().reshape(-1)
            
            print('\nData {}'.format(i))
            print(confidence)
            #print(max(confidence))
            #confidence.sort()
            #print(confidence)


model = load_model(CIFAR10Net, './week1/exercise5/cifar10.pt')

print('\n===================================\n')
print('Confidence with train data')
print_confidence(model, 'train')
print('\n===================================\n')

print('\n===================================\n')
print('Confidence with test data')
print_confidence(model, 'test')
print('\n===================================\n')
