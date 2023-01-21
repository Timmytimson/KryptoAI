from tkinter import Variable

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()  # wichtig gegen auswendig lernen!
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 26)  # 26 outputs

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        #print(x.size())  # so kommt man auf 320 input für fc1
                         # Size(64, 20, 4, 4) -> 20*4*4=320
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # ohne Batch-Dimension
        num = 1
        for i in size:
            num *= i
        return num


def rotational_error(char_result, char_target):
    return abs(ord(char_target)-ord(char_result))


def rotational_error_oneway(char_result, char_target):
    diff = ord(char_target)-ord(char_result)
    if diff >= 0:
        return diff
    return diff + 26


kwargs = {'num_workers': 1, 'pin_memory': True}
train_data = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,),(0.3081,))])),
    batch_size=64, shuffle=True, **kwargs)
test_data = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False,
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,),(0.3081,))])),
    batch_size=64, shuffle=True, **kwargs)


def train(epoch):
    my_network.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = data.cuda()
        target = target.cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = my_network(data)
        criterion = F.nll_loss()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id * len(data), len(train_data.dataset), 100. * batch_id / len(train_data), loss.data[0]))


if torch.cuda.is_available():
    print("Cuda: check")
    print("Torch Version: {}".format(torch.__version__))
else:
    print(("Cuda: nay"))
    exit()

my_network = Network()
my_network = my_network.cuda()  # Netz auf Graka

optimizer = optim.SGD(my_network.parameters(), lr=0.1)  # SGD = Stochastic Gradient Descent, lr = learning rate
optimizer.step()

if os.path.isfile('caesar_network.pt'):
    my_network = torch.load('caesar_network.pt')
    my_network = my_network.cuda()

for i in range(100):
    train(i)

torch.save(my_network, 'caesar_network.pt')
