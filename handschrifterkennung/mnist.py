import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

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


class Netz(nn.Module):
    def __int__(self):
        super(Netz, self).__int__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()  # wichtig gegen auswendig lernen!
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)  # 10 Möglichkeiten, also 10 outputs

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


model = Netz()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)


def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = data.cuda()
        target = target.cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.nll_loss()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id * len(data), len(train_data.dataset), 100. *batch_id / len(train_data), loss.data[0]))


'''for epoch in range(1, 30):
    train(epoch)'''