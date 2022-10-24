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
import numpy as np

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
    print('Test Accuracy: {:.2f}%, Avg loss: {:.4f}'.format(100 * correct, loss))

#TODO: implement the following function which prepare the training data 
# and the corresponding labels for extracting the model;
# model is the original model;
# dataloader is the training instances to be labeled;
# atk_mode is a string which has three values, "label", "full", "round";
def attack(model, dataloader, device, atk_mode):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    input_data, pred_data = [], []
    
    #the actual TODO goes here to populate input_data and pred_data.

    return input_data, pred_data


def fidelity(model1, model2, dataloader, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model1.eval()
    model2.eval()

    score = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred1 = F.softmax(model1(x), 1)
            pred2 = F.softmax(model2(x), 1)

            pred1 = torch.argmax(pred1, dim=1).numpy()
            pred2 = torch.argmax(pred2, dim=1).numpy()

            score += np.sum(pred1 == pred2)

    print('Model fidelity = {:.2f}'.format(score / size))


device = 'cpu'
train_kwargs = {'batch_size': 100}
test_kwargs = {'batch_size': 1000}
transform = transforms.ToTensor()

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, transform=transform)

sample_indexes = random.sample(range(60000), 6000)
train_dataset.data = train_dataset.data[sample_indexes]
train_dataset.targets = train_dataset.targets[sample_indexes]

train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model1 = load_model(MNISTNet, 'week9/exercise5/mnist.pt')
print('Original model')
test(model1, test_loader, nn.CrossEntropyLoss(), device)

atk_mode = 'label' 
#atk_mode = 'full'
#atk_mode = 'round'
input_data, pred_data = attack(model1, train_loader, device, atk_mode)

input_data_tensor = torch.Tensor(np.array(input_data))
pred_data_tensor = torch.Tensor(np.array(pred_data))

if atk_mode == 'label': pred_data_tensor = pred_data_tensor.type(torch.LongTensor)

train_dataset = TensorDataset(input_data_tensor, pred_data_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

model2 = MNISTNet().to(device)

optimizer = optim.SGD(model2.parameters(), lr=0.1)
num_of_epochs = 100

for epoch in range(num_of_epochs):
    print('\n------------- Epoch {} -------------\n'.format(epoch))
    train(model2, train_loader, nn.CrossEntropyLoss(), optimizer, device)
    
save_model(model2, 'week9/exercise5/mnist_atk.pt')

model2 = load_model(MNISTNet, 'week9/exercise5/mnist_atk.pt')
#print('Extract model')
test(model2, test_loader, nn.CrossEntropyLoss(), device)

fidelity(model1, model2, test_loader, device)
