import random

import os
from os import listdir
import glob

import unicodedata
import string

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import time
import math

'''
Datenverarbeitung
'''
dir_training = r'..\Caesar\training data\*.txt'
dir_test = r'..\Caesar\test data\*.txt'


def find_files(path):
    return glob.glob(path)


letters = string.ascii_letters + " .,;'"
n_letters = len(letters)


def to_ascii(str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', str)
        if unicodedata.category(c) != 'Mn'  # !='Mn' ... nicht Ascii
        and c in letters
    )


# Build the category_lines dictionary, a list of names per language
rotation_lines = {}
all_rotations = []

# Read a file and split into lines


def read_lines(filename):
    file = open(filename, encoding='utf-8').read().strip().split('\n')
    return [to_ascii(line) for line in file]


for file in find_files(dir_training):  # TODO category -> rotation
    rotation = os.path.splitext(os.path.basename(file))[0]
    rotation = rotation.split('_')[0]
    all_rotations.append(rotation)
    lines = read_lines(file)
    rotation_lines[rotation] = lines

n_categories = len(all_rotations)


def letter_to_index(chr):
    return letters.find(chr)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(chr):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(chr)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

# print(letterToTensor('J'))

# print(lineToTensor('Jones').size())


'''
Netz
'''


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


'''
Training
'''


def rotation_from_output(out):
    top_n, top_i = out.topk(1)
    category_i = top_i[0].item()
    return all_rotations[category_i], category_i


def random_choice(line):
    return line[random.randint(0, len(line) - 1)]


def random_training_example():
    category = random_choice(all_rotations)
    line = random_choice(rotation_lines[category])
    rotation_tensor = torch.tensor([all_rotations.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, rotation_tensor, line_tensor


def train(rotation_tensor, sample_tensor):
    hidden = network.initHidden()

    network.zero_grad()

    for i in range(sample_tensor.size()[0]):
        output, hidden = network(sample_tensor[i], hidden)

    loss = criterion(output, rotation_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in network.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def time_running(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


'''
Main
'''
n_iters = 10000
print_every = n_iters/20
plot_every = 2*print_every


criterion = nn.NLLLoss()

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn


# Keep track of losses for plotting
current_loss = 0
all_losses = []

'''for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)'''

n_hidden = 128
network = Network(n_letters, n_hidden, n_categories)

input = line_to_tensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = network(input[0], hidden)
#print(output)

#print(categoryFromOutput(output))

start = time.time()

n_errors = 0
for iter in range(1, n_iters + 1):
    try:
        rotation, line, category_tensor, line_tensor = random_training_example()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = rotation_from_output(output)
            correct = '✓' if guess == rotation else '✗ (%s)' % rotation
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, time_running(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    except:
        n_errors += 1
        print('Error at iteration ', iter)
        print('rotation tensor ', category_tensor)
        print('line tensor ', line_tensor)
print(n_errors, ' errors occured')
plt.figure()
plt.plot(all_losses)
plt.show()
