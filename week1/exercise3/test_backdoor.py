import torch
import random

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



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

model = MNISTNet().to(device)
model = load_model(MNISTNet, 'mnist2.pt')

test_dataset = datasets.MNIST('../data', train=False, transform=transform)
backdoor_test_dataset = datasets.MNIST('../data', train=False, transform=transform)

print('With original data')
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
test(model, test_loader, nn.CrossEntropyLoss(), device)

for i in range(len(backdoor_test_dataset.data)):
    backdoor_test_dataset.data[i][0][0] = 255
    backdoor_test_dataset.data[i][0][1] = 255
    backdoor_test_dataset.data[i][0][2] = 255
    backdoor_test_dataset.data[i][1][0] = 255
    backdoor_test_dataset.data[i][1][1] = 255
    backdoor_test_dataset.data[i][1][2] = 255
    backdoor_test_dataset.data[i][2][0] = 255
    backdoor_test_dataset.data[i][2][1] = 255
    backdoor_test_dataset.data[i][2][2] = 255
    backdoor_test_dataset.targets[i] = 5

print('With backdoored data')
backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_dataset, **test_kwargs)
test(model, backdoor_test_loader, nn.CrossEntropyLoss(), device)
