import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import random
import math
import numpy as np


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


def load_model(model_class, name, *args):
    model = model_class(*args)
    model.load_state_dict(torch.load(name, map_location=torch.device('cpu')))

    return model


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
            print('loss: {:.4f} [{}/{}]'.format(loss, current, size))


def test(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    loss, correct = 0.0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()

    loss /= num_batches
    correct /= size
    print('Test Error: \n Accuracy: {:.2f}%, Avg loss: {:.4f}'.format(100 * correct, loss))


def attack(model, dataloader, loss_fn, device, threshold):
    size = 10000
    num_batches = len(dataloader)

    model.eval()
    correct = 0

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = F.softmax(model(x), 1)
            #TODO: update correct to complete this MIA attack.

    return correct / size * 100


device = 'cpu'
train_kwargs = {'batch_size': 1000}
test_kwargs = {'batch_size': 1000}
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10('../data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model = CIFAR10Net().to(device)

# optimizer = optim.Adam(model.parameters(), lr=0.001)
# num_of_epochs = 20

#for epoch in range(num_of_epochs):
#    print('\n------------- Epoch {} -------------\n'.format(epoch))
#    train(model, train_loader, nn.CrossEntropyLoss(), optimizer, device)
#    test(model, test_loader, nn.CrossEntropyLoss(), device)

#save_model(model, 'cifar10.pt')
model = load_model(CIFAR10Net, 'week8/exercise5/cifar10.pt')
threshold = 0.5

print('For train data')
print('MIA accuracy: {:.2f}%\n'.format(attack(model, train_loader, nn.CrossEntropyLoss(), device, threshold)))

print('For test data')
print('MIA accuracy: {:.2f}%\n'.format(100 - attack(model, test_loader, nn.CrossEntropyLoss(), device, threshold)))
