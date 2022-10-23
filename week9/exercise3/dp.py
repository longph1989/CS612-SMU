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
import math


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


def load_model(model_class, name, *args):
    model = model_class(*args)
    model.load_state_dict(torch.load(name, map_location=torch.device('cpu')))

    return model


def train(model, dataloader, loss_fn, optimizer, device, delta, epsilon):
    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
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

        #The following adds the noise according to differential privacy;
        for name, param in model.named_parameters():
            param.grad.data += np.random.normal(loc=0.0, scale=sigma, size=param.grad.data.size()) / len(y)

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
    print('Test Error: \n Accuracy: {:.2f}%, Avg loss: {:.4f}\n'.format(100 * correct, loss))

def compute_mentr(pred, y):
    pred, y = pred.numpy(), y.numpy()
    mentr = []

    for i in range(len(y)):
        val = 0.0
        for j in range(len(pred[i])):
            if j == y[i]:
                val -= (1 - pred[i][j]) * math.log(pred[i][j])
            elif pred[i][j] < 1:
                val -= pred[i][j] * math.log(1 - pred[i][j])
        mentr.append(val)

    return np.array(mentr)

def attack(model, dataloader, loss_fn, device, threshold):
    size = 10000
    num_batches = len(dataloader)

    model.eval()
    correct = 0

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = F.softmax(model(x), 1)
            mentr = compute_mentr(pred, y)
            correct += (mentr < threshold).sum()
            if (batch + 1) * len(y) == size: break

    return correct / size * 100


device = 'cpu'
train_kwargs = {'batch_size': 100}
test_kwargs = {'batch_size': 1000}
transform = transforms.ToTensor()

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model = MNISTNet().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.1)
num_of_epochs = 20
#delta, epsilon = 1e-1, 1e-1 
delta, epsilon = 1e-5, 1 

print("A program demonstrating training with differential privacy.")

for epoch in range(num_of_epochs):
   print('\n------------- Epoch {} -------------\n'.format(epoch))
   train(model, train_loader, nn.CrossEntropyLoss(), optimizer, device, delta, epsilon)
   test(model, test_loader, nn.CrossEntropyLoss(), device)

save_model(model, 'week9/exercise3/mnist_dp.pt')

# model = load_model(MNISTNet, 'mnist_dp.pt')
threshold = 0.9

MIAAttackTrain = attack(model, train_loader, nn.CrossEntropyLoss(), device, threshold)
MIAAttackTest = 100 - attack(model, test_loader, nn.CrossEntropyLoss(), device, threshold)

print('Overall MIA accuracy: {:.2f}%\n'.format((MIAAttackTrain+MIAAttackTest)/2))
print('MIA accuracy on train data: {:.2f}%\n'.format(MIAAttackTrain))
print('MIA accuracy on test data: {:.2f}%\n'.format(MIAAttackTest))
