from tkinter import Variable

import os

import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Lineare Schichten
        self.lin1 = nn.Linear(10, 10)  # 10 Input Features, 10 Output
        self.lin2 = nn.Linear(10, 10)

    def forward(self, x):
        x = fun.relu(self.lin1(x))
        x = self.lin2(x)
        return x

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



if torch.cuda.is_available():
    # Gewünscht, da Berechnung auf Graka schneller
    print("Cuda: check")
else:
    print(("Cuda: nay"))
    exit()

#my_network = Network()
#my_network = my_network.cuda()  # Netz auf Graka

if os.path.isfile('caesar_network.pt'):
    my_network = torch.load('caesar_network.pt')
    my_network = my_network.cuda()

for i in range(100):
    x = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1]
    input = Variable(torch.Tensor([x for _ in range(10)]))
    input = input.cuda()
    #print(input)

    output = my_network(input)
    #print(output)

    # Lernen
    x = [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]
    target = Variable(torch.Tensor([x for _ in range(10)]))  # gewünschtes Ergebnis
    target = target.cuda()
    criterion = nn.MSELoss()  # Fehlerfunktion
    loss = criterion(output, target)  # Fehler
    print(loss)
    # print(loss.grad_fn.next_functions[0][0])  # Fehler für bestimmte Lineare Schicht

    my_network.zero_grad()  # Veränderung zurücksetzen
    loss.backward()  # Backproporgation
    optimizer = optim.SGD(my_network.parameters(), lr=0.1)  # SGD = Stochastic Gradient Descent, lr = learning rate
    optimizer.step()

torch.save(my_network, 'caesar_network.pt')
