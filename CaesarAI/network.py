import random
from tkinter import Variable

import os
from os import listdir

import unicodedata
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

'''
Character-RNN (Rekurrentes neuronales Netz) zur Klassifizierung?
-> besonders gut für Folgen von Daten (Sätze, Videos, ...)
'''
dir_train = r'..\Caesar\training data'
dir_test = r'..\Caesar\test data'

letters = string.ascii_letters + ".,:'"  # TODO fehlen noch Buchstaben?


def to_ascii(str):
    return ''.join(
        char for char in unicodedata.normalize('NFD', str)
        if unicodedata.category(char) != 'Mn'  # !='Mn' ... nicht Ascii
        and char in letters
    )


def lines(data):
    file = open(data, encoding='utf-8').read().strip().split('\n')  # Strip entfernt leerzeichen
    return [to_ascii(line) for line in file]


def char_to_index(c):
    return letters.find(c)


def char_to_tensor(c):  # "One Hot Tensor"
    ret = torch.zeros(1, len(letters))  # TODO requires grad
    ret[0][char_to_index(c)] = 1
    return ret


def text_to_tensor(txt):
    ret = torch.zeros(len(txt), 1, len(letters))
    for i, char in enumerate(txt):
        ret[i][0][char_to_index(char)] = 1
    return ret


data = {}
rotations = []
for filename in listdir(dir_train):
    rotation = filename.split('_')[0]
    txt = lines(dir_train + '\\' + filename)
    rotations.append(rotation)
    data[rotation] = txt


class Network(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):  # Layer -> Size/Count?
        super(Network, self).__init__()
        self.hidden_layer = hidden_layer
        self.lin1 = nn.Linear(input_layer + hidden_layer, hidden_layer)  # Todo mehr Schichten oder hidden neuronen?
        self.out = nn.Linear(input_layer + hidden_layer, output_layer)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        hidden = hidden.cuda()
        x = torch.cat((x, hidden), 1)
        newHidden = self.lin1(x)
        output = self.logsoftmax(self.out(x))
        return output, newHidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_layer))


def rotation_from_output(out):
    _, i = out.data.topk(1)
    return rotations[i[0][0]]


def get_train_data():
    rotation = int(random.choice(rotations))
    sample = random.choice(rotations[rotation])
    sample_tensor = Variable(text_to_tensor(sample))
    rotation_tensor = Variable(torch.LongTensor([rotations.index(str(rotation))]))
    return rotation, sample, rotation_tensor, sample_tensor


def train(rotation_tensor, sample_tensor, learning_rate):
    rotation_tensor = rotation_tensor.cuda()
    sample_tensor = sample_tensor.cuda()
    hidden = caesar_network.init_hidden()
    caesar_network.zero_grad()
    for i in range(sample_tensor.size()[0]):
        output, hidden = caesar_network(sample_tensor[i], hidden)
    loss = criterion(output, rotation_tensor)
    loss.backward()
    for param in caesar_network.parameters():
        param.data.add_(-learning_rate, param.grad.data)  # ursprünglich i.grad.data

    return output, loss


'''
Hauptprogramm
'''

if torch.cuda.is_available():
    print("Cuda: check")
    print("Torch Version: {}".format(torch.__version__))
else:
    print(("Cuda: nay"))
    exit()


if os.path.isfile('caesar_network.pt'):
    caesar_network = torch.load('caesar_network.pt')
    caesar_network = caesar_network.cuda()
else:
    caesar_network = Network(len(letters), 128, len(data))
    caesar_network = caesar_network.cuda()

criterion = nn.NLLLoss()
learningRate = 0.1

avg = []
sum = 0
steps = 10000
learningRate = 0.1  # Todo ausprobieren
for i in range(1, steps):
    rotation, sample, rotationTensor, sampleTensor = get_train_data()
    output, loss = train(rotationTensor, sampleTensor, learningRate)
    print("loss: ", loss.data[0])
    sum += loss.data[0]

    if i % (steps/100) == 0:
        learningRate /= 2  # Todo ausprobieren
        avg.append(sum/(steps/100))
        sum = 0
        print(i/(steps/100), "% done")

# torch.save(caesar_network, 'caesar_network.pt')

plt.figure()
plt.plot(avg)
plt.show()
