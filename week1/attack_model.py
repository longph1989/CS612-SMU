import torch
import autograd.numpy as np

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from scipy.optimize import minimize
from scipy.optimize import Bounds
from autograd import grad

from lib_models import *
from lib_layers import *

import matplotlib.pyplot as plt


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



def denormalize(x):
        x = (x * 255).astype('uint8')
        x = x.reshape(28,28)

        return x


def display(x0, y0, x, y):
    x0 = denormalize(x0)
    x = denormalize(x)

    fig, ax = plt.subplots(1, 2)

    ax[0].set(title='Original. Label is {}'.format(y0))
    ax[1].set(title='Adv. sample. Label is {}'.format(y))

    ax[0].imshow(x0, cmap='gray')
    ax[1].imshow(x, cmap='gray')
    
    plt.show()


def attack(model, x0, y0, eps):
    def obj_func(x, model, y0):
        output = model.apply(x).reshape(-1)
        y0_score = output[y0]

        output_no_y0 = output - np.eye(len(output))[y0] * 1e9
        max_score = np.max(output_no_y0)

        return y0_score - max_score

    lower = np.maximum(x0 - eps, 0.0)
    upper = np.minimum(x0 + eps, 1.0)

    x = x0.copy()
    
    args = (model, y0)
    jac = grad(obj_func)
    bounds = Bounds(lower, upper)

    res = minimize(obj_func, x, args=args, jac=jac, bounds=bounds)

    if res.fun <= 0: # an adversarial sample is generated
        print('\nFound an adversarial sample!!!\n')
        print('adv = {}'.format(res.x[400:420]))

        pred = model.apply(res.x).reshape(-1)
        y_adv = np.argmax(pred)

        print('pred adv = {}'.format(pred))
        print('lbl adv = {}\n'.format(y_adv))

        display(x0, y0, res.x, y_adv)
    else:
        print('\nCan\'t find adversarial samples!!!\n')


def get_layers(model):
    layers, params = list(), list(model.named_parameters())

    for i in range(len(params)):
        name, param = params[i]
        if 'weight' in name:
            weight = np.array(param.data)
        elif 'bias' in name:
            bias = np.array(param.data)

            layers.append(Linear(weight, bias, None))
            if i < len(params) - 1: # last layer
                layers.append(Function('relu', None))

    return layers


def get_formal_model(model, shape, lower, upper):
    lower, upper = lower.copy(), upper.copy()
    layers = get_layers(model)
    
    return Model(shape, lower, upper, layers)


test_kwargs = {'batch_size': 1}
transform = transforms.ToTensor()
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model = load_model(MNISTNet, 'mnist.pt')
formal_model = get_formal_model(model, (1,784), np.zeros(784), np.ones(784))
cnt = 0

for x, y in test_loader:
    x0 = x.numpy().reshape(-1)
    pred = model(x).detach().numpy().reshape(-1)
    y0 = np.argmax(pred)

    print('\nimg = {}'.format(x0[400:420]))
    print('pred img = {}'.format(pred))
    print('lbl imp = {}\n'.format(y0))

    eps = 0.05
    attack(formal_model, x0, y0, eps)
    cnt += 1

    print('\n===========================\n')

    if cnt == 10: break
