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
import ast
import numpy as np

class ClassifierNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 64)
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


def print_model(model):
    for name, param in model.named_parameters():
        print(name)
        print(param.data)


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

    return correct / size


def precision(model, dataloader, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    pred_member, real_member = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            #TODO: add code here to complete this MIA attack by updating pred_member and real_member
            #note that pred_member is the total number of samples predicted to be a member;
            #real_member is the total number of samples predicted to be a member and are actually a member;

    precision = real_member / pred_member
    print('Precision: {:.2f}%\n'.format(100 * precision))


def recall(model, dataloader, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    real_member, all_member = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            real_member += (torch.logical_and(pred.argmax(1) == 1, pred.argmax(1) == y)).type(torch.int).sum().item()
            all_member += (y == 1).type(torch.int).sum().item()

    recall = real_member / all_member
    print('Recall: {:.2f}%\n'.format(100 * recall))


def get_loader(data_file, kwargs):
    data_file = open(data_file, 'r')
    input_data, label_data = [], []

    for line in data_file.readlines():
        data = ast.literal_eval(line)
        input_data.append(data[:-1])
        label_data.append(data[-1])

    input_data = np.array(input_data)
    label_data = np.array(label_data)

    input_data_tensor = torch.Tensor(input_data)
    label_data_tensor = torch.Tensor(label_data).type(torch.LongTensor)

    dataset = TensorDataset(input_data_tensor, label_data_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, **kwargs)

    return data_loader


device = 'cpu'
train_kwargs = {'batch_size': 100}
test_kwargs = {'batch_size': 1000}
transform = transforms.ToTensor()

train_loader = get_loader('week8/exercise1/train_data.txt', train_kwargs)
test_loader = get_loader('week8/exercise1/test_data.txt', test_kwargs)

model = ClassifierNet().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)
num_of_epochs = 100
best_acc = 0.0

for epoch in range(num_of_epochs):
   print('\n------------- Epoch {} -------------\n'.format(epoch))
   train(model, train_loader, nn.CrossEntropyLoss(), optimizer, device)
   acc = test(model, test_loader, nn.CrossEntropyLoss(), device)

   if acc > best_acc:
        save_model(model, 'week8/exercise1/classifier.pt')
        best_acc = acc

model = load_model(ClassifierNet, 'week8/exercise1/classifier.pt')

test(model, test_loader, nn.CrossEntropyLoss(), device)
precision(model, test_loader, device)
recall(model, test_loader, device)
