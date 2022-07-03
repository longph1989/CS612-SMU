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
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


device = 'cpu'
test_kwargs = {'batch_size': 1000}
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10('../data', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, **test_kwargs)

test_dataset = datasets.CIFAR10('../data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model = load_model(CIFAR10Net, 'cifar10.pt')

test(model, train_loader, nn.CrossEntropyLoss(), device)
test(model, test_loader, nn.CrossEntropyLoss(), device)
