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
import ast
import scipy.stats



class CensusNet(nn.Module):
    def __init__(self, num_of_features):
        super().__init__()
        self.fc1 = nn.Linear(num_of_features, 64)
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


def load_model(model_class, name, num_of_features):
    model = model_class(num_of_features)
    model.load_state_dict(torch.load(name))

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

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(x)
        #     print('loss: {:.4f} [{}/{}]'.format(loss, current, size))


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


def get_loader(input_file, label_file, kwargs, input_data=None):
    if input_data is None:
        input_file = open(input_file, 'r')
        input_data = []
        for line in input_file.readlines():
            input_data.append(ast.literal_eval(line))
        input_data = np.array(input_data)

    label_file = open(label_file, 'r')
    label_data = np.array(ast.literal_eval(label_file.readline()))

    input_data_tensor = torch.Tensor(input_data)
    label_data_tensor = torch.Tensor(label_data).type(torch.LongTensor)
    dataset = TensorDataset(input_data_tensor, label_data_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    return input_data, label_data, data_loader


def train_model(num_of_features, train_loader, test_loader, device, name):
    model = CensusNet(num_of_features).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    num_of_epochs = 20

    for epoch in range(num_of_epochs):
        print('\n------------- Epoch {} -------------\n'.format(epoch))
        train(model, train_loader, nn.CrossEntropyLoss(), optimizer, device)
        test(model, test_loader, nn.CrossEntropyLoss(), device)

    save_model(model, name)


def find_corr(num_of_features, gender_idx, train_input_data):
    max_rho, max_i = 0.0, -1

    for i in range(num_of_features):
        if i == gender_idx: continue

        feature_i = train_input_data[:,i]
        feature_j = train_input_data[:,gender_idx]

        rho, pvalue = scipy.stats.spearmanr(feature_i, feature_j)
        print(' i = {}, corr = {}'.format(i, rho))

        if abs(rho) > max_rho:
            max_rho = abs(rho)
            max_i = i

    return max_i            


def generate_x(size, lower, upper):
    x = np.random.rand(size)
    x = (upper - lower) * x + lower
    return x


device = 'cpu'
train_kwargs = {'batch_size': 100}
test_kwargs = {'batch_size': 1000}

train_input_data, train_label_data, train_loader = get_loader('week7/exercise4/train/input.txt', 'week7/exercise4/train/label.txt', train_kwargs)
test_input_data, test_label_data, test_loader = get_loader('week7/exercise4/test/input.txt', 'week7/exercise4/test/label.txt', test_kwargs)

num_of_features = 13
gender_idx = 8

# train_model(num_of_features, train_loader, test_loader, device, 'census.pt')
model = load_model(CensusNet, 'week7/exercise4/census.pt', num_of_features)
test(model, test_loader, nn.CrossEntropyLoss(), device)

max_i = find_corr(num_of_features, gender_idx, train_input_data)
print('most corr feature = {}'.format(max_i))

all_idx = np.arange(13)
remove_idx = [gender_idx, max_i]
supp_models = []

for i in range(2):
    all_idx = np.delete(all_idx, remove_idx[i])
    supp_train_input_data = train_input_data[:,all_idx]
    supp_test_input_data = test_input_data[:,all_idx]

    _, _, train_loader = get_loader('week7/exercise4/train/input.txt', 'week7/exercise4/train/label.txt', train_kwargs, supp_train_input_data)
    _, _, test_loader = get_loader('week7/exercise4/test/input.txt', 'week7/exercise4/test/label.txt', test_kwargs, supp_test_input_data)

    file_name = 'week7/exercise4/supp_census' + str(i + 1) + '.pt'
    # train_model(num_of_features - (i + 1), train_loader, test_loader, device, file_name)
    supp_model = load_model(CensusNet, file_name, num_of_features - (i + 1))
    test(supp_model, test_loader, nn.CrossEntropyLoss(), device)

    supp_models.append(supp_model)

lower = np.zeros(num_of_features)
upper = np.ones(num_of_features)

num_of_samples = 10000
female_samples, male_samples = [], []

# for i in range(num_of_samples):
for i in range(len(test_input_data)):
    # x = generate_x(num_of_features, lower, upper)
    x = test_input_data[i]

    if x[gender_idx] < 0.5: # female is 0, male is 1
        female_samples.append(x)
    else:
        male_samples.append(x)

print('no of samples female = {}, male = {}'.format(len(female_samples), len(male_samples)))

all_models = ['week7/exercise4/census.pt', 'week7/exercise4/supp_census1.pt', 'week7/exercise4/supp_census2.pt']


def test_fairness(model, samples, i):
    cnt = 0
    for x in samples:
        x = x.copy()
        for j in range(i):
            x = np.delete(x, remove_idx[j])
        x = x.reshape(1, -1)
        x_tensor = torch.Tensor(x)
        if model(x_tensor).argmax(1).item() == 0: # >50K
            cnt += 1
    return cnt


for i in range(3):
    model = load_model(CensusNet, all_models[i], num_of_features - i)
    
    female_cnt = test_fairness(model, female_samples, i)
    male_cnt = test_fairness(model, male_samples, i)

    pr_female = female_cnt / len(female_samples)
    pr_male = male_cnt / len(male_samples)

    print('|Pr(>50K|Male) - Pr(>50K|Female)| = {}'.format(abs(pr_male - pr_female) * 100))
