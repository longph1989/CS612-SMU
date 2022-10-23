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

#The following masks off the confidence except those top three
def maskoff(pred):
    for i in range(len(pred)):
        fst_i, fst_v = -1, -1
        snd_i, snd_v = -1, -1
        thr_i, thr_v = -1, -1
        for j in range(len(pred[i])):
            if pred[i][j] > fst_v:
                thr_i, thr_v = snd_i, snd_v
                snd_i, snd_v = fst_i, fst_v
                fst_i, fst_v = j, pred[i][j]
            elif pred[i][j] > snd_v:
                thr_i, thr_v = snd_i, snd_v
                snd_i, snd_v = j, pred[i][j]
            elif pred[i][j] > thr_v:
                thr_i, thr_v = j, pred[i][j]
        for j in range(len(pred[i])):
            if j != fst_i and j != snd_i and j != thr_i:
                pred[i][j] = 0.0
        pred[i][fst_i] = fst_v
        pred[i][snd_i] = snd_v
        pred[i][thr_i] = thr_v
    return pred


def attack(model, dataloader, device, threshold):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = F.softmax(model(x), 1)
            pred = maskoff(pred)
            mentr = compute_mentr(pred, y)
            correct += (mentr < threshold).sum()
            #pred = torch.max(pred, 1).values
            #correct += (pred >= threshold).type(torch.int).sum().item()

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

model = load_model(CIFAR10Net, 'week9/exercise2/cifar10.pt')
threshold = 0.5

MIAAttackTrain = attack(model, train_loader, device, threshold)
MIAAttackTest = 100 - attack(model, test_loader, device, threshold)

print('Overall MIA accuracy: {:.2f}%\n'.format((MIAAttackTrain+MIAAttackTest)/2))
print('MIA accuracy on train data: {:.2f}%\n'.format(MIAAttackTrain))
print('MIA accuracy on test data: {:.2f}%\n'.format(MIAAttackTest))