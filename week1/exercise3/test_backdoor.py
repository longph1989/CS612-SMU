import torch

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


def load_model(model_class, name):
    model = model_class()
    model.load_state_dict(torch.load(name))

    return model


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


device = 'cpu'
test_kwargs = {'batch_size': 1000}
transform = transforms.ToTensor()

model = MNISTNet().to(device)
model = load_model(MNISTNet, './week1/exercise3/mnist2.pt') #if you run from a different folder, change the path here accordingly.

test_dataset = datasets.MNIST('../data', train=False, transform=transform)
backdoor_test_dataset = datasets.MNIST('../data', train=False, transform=transform)

print('With original data')
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
test(model, test_loader, nn.CrossEntropyLoss(), device)

#for i in range(len(backdoor_test_dataset.data)):
    #TODO: add the 3*3 white square at the top-left corner of each sample
    #TODO: set the target to be 5

#print('With backdoored data')
#backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_dataset, **test_kwargs)
#test(model, backdoor_test_loader, nn.CrossEntropyLoss(), device)
