import random

import os
import glob

import unicodedata
import string

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import time
import math

'''
Netz
'''


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()

        self.hidden_size = hidden_size

        self.hid = nn.Linear(input_size + hidden_size, hidden_size)
        self.lin1 = nn.Linear(input_size + hidden_size, output_size)
        self.dropout1 = nn.Dropout(p=0.2)
        self.lin2 = nn.Linear(output_size, output_size)
        self.dropout2 = nn.Dropout(p=0.3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.hid(combined)
        output = self.lin1(combined)
        output = self.dropout1(output)
        output = self.lin2(output)
        output = self.dropout2(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


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

def is_null_or_empty(str):
    if not str:
        return True


# Read a file and split into lines


def read_lines(filename):
    file = open(filename, encoding='utf-8').read().strip().split('\n')
    return [to_ascii(line) for line in file]


def init_data(dir):
    # Build the category_lines dictionary, a list of names per language
    rotation_lines = {}
    all_rotations = []

    for file in find_files(dir):
        rotation = os.path.splitext(os.path.basename(file))[0]
        rotation = rotation.split('_')[0]
        all_rotations.append(rotation)
        lines = read_lines(file)
        rotation_lines[rotation] = lines

    n_rotations = len(all_rotations)

    return all_rotations, rotation_lines, n_rotations


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
Training
'''
def rotational_error(output_tensor, target_tensor):
    top_n, top_i = output_tensor.topk(1)
    index_out = top_i[0].item()
    top_n, top_i = target_tensor.topk(1)
    index_target = top_i[0].item()
    return abs(index_out-index_target)

def time_running(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_thresholds(n_iters, min_steps):
    increment = int(n_iters / 2)
    threshold = 0
    thresholds = [threshold]
    while(increment > min_steps):
        threshold += increment
        thresholds.append(threshold)
        increment = int(increment / 2)

    return thresholds

def rotation_from_output(out, all_rotations):
    top_n, top_i = out.topk(1)
    rotation = top_i[0].item()
    return all_rotations[rotation], rotation


def random_choice(line):
    return line[random.randint(0, len(line) - 1)]


def random_training_example(all_rotations, rotation_lines):
    category = random_choice(all_rotations)
    line = random_choice(rotation_lines[category])
    rotation_tensor = torch.tensor([all_rotations.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, rotation_tensor, line_tensor


def train_network(rotation_tensor, sample_tensor, learning_rate, criterion):
    hidden = network.initHidden()
    hidden = hidden.cuda()

    network.zero_grad()
    network.train()

    for i in range(sample_tensor.size()[0]):
        output, hidden = network(sample_tensor[i], hidden)

    loss = criterion(output, rotation_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in network.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def train(all_rotations, print_every, plot_every, lower_learn_rate_thresholds, criterion):
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    plot_losses = []

    start = time.time()
    learning_rate = base_learning_rate
    n_errors = 0
    nan_error_occured = False
    for iter in range(1, n_iters + 1):
        rotation, line, rotation_tensor, line_tensor = random_training_example(all_rotations, rotation_lines)
        if is_null_or_empty(line):
            continue
        rotation_tensor = rotation_tensor.cuda()
        line_tensor = line_tensor.cuda()

        output, loss = train_network(rotation_tensor, line_tensor, learning_rate, criterion)
        current_loss += loss
        all_losses.append(loss)

        if math.isnan(loss):
            nan_error_occured = True
            print('Loss at step', iter, 'is not a number. Last losses were\n', all_losses[len(all_losses)-10:])
            break
        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = rotation_from_output(output, all_rotations)
            correct = '✓' if guess == rotation else '✗ (%s)' % rotation
            #print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, time_running(start), loss, line, guess, correct))
            print(iter, str(round(iter/n_iters * 100)) + '% Loss=' + str(loss))
            print('\tguessed', guess, 'for', line, correct)

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            plot_losses.append(current_loss / plot_every)
            current_loss = 0

        if iter in lower_learn_rate_thresholds:
            learning_rate = learning_rate/2
            print('Lowering learning rate to ', learning_rate)

    print(n_errors, ' errors occured')
    plt.figure()
    plt.plot(plot_losses)
    plt.show()
    return network, nan_error_occured


'''
Testing
'''
def increment_according_to_len(line, list):
    l = len(line)
    if l <= 20:
        list[0] += 1
    elif 20 < l <= 40:
        list[1] += 1
    elif 40 < l <= 60:
        list[2] += 1
    elif 60 < l <= 80:
        list[3] += 1
    elif 80 < l <= 100:
        list[4] += 1
    else:
        list[5] += 1
    return list

def test_network(dir_test, network, n_tests):
    n_correct = [0] * 6
    n_wrong = [0] * 6
    mismatches = []
    network.eval()
    hidden = network.initHidden()
    hidden = hidden.cuda()
    print_every = n_tests/10

    all_rotations, rotation_lines, n_rotations = init_data(dir_test)
    for iters in range(n_tests):
        rotation, line, rotation_tensor, line_tensor = random_training_example(all_rotations, rotation_lines)
        if is_null_or_empty(line):
            continue
        rotation_tensor = rotation_tensor.cuda()
        line_tensor = line_tensor.cuda()

        for i in range(line_tensor.size()[0]):
            output, hidden = network(line_tensor[i], hidden)

        guessed_rotation = rotation_from_output(output, all_rotations)[1]
        if guessed_rotation == int(rotation):
            n_correct = increment_according_to_len(line, n_correct)
        else:
            n_wrong = increment_according_to_len(line, n_wrong)
            mismatches.append((guessed_rotation, rotation, line))
        if iters % print_every == 0:
            print(str(round(iters/n_tests*100)) + '% done')

    return n_correct, n_wrong, mismatches


def evaluate(n_correct, n_wrong, mismatches, print_errors=False):
    sum_correct = sum(n_correct)
    sum_wrong = sum(n_wrong)
    n_all = sum_correct + sum_wrong
    print('Tested the network on', n_all, 'words and sentences.')
    print(sum_correct, 'examples were classified correctly.', '(' + str(sum_correct/n_all*100) + '%)')
    print(sum_wrong, 'examples were classified wrong.', '(' + str(sum_wrong/n_all*100) + '%)')
    print('')

    bot = 1
    top = 20
    for i in range(len(n_correct)):
        n_all = n_correct[i] + n_wrong[i]
        if bot < 100:
            print('For words with a length from', bot, 'to', top)
        else:
            print('For words with a length from', bot, 'onwards')
        print('\t', n_correct[i], 'examples were classified correctly.', '(' + str(n_correct[i]/n_all*100) + '%)')
        print('\t', n_wrong[i], 'examples were classified wrong.', '(' + str(n_wrong[i]/n_all*100) + '%)')
        print('')
        bot += 20
        top += 20


    if print_errors:
        for tuple in mismatches:
            print(tuple)
    return

'''
Main
'''
network_filename = 'caesar_network.pt'#
save_network = True


# existing network
if os.path.isfile(network_filename):
    print('Found already trained network.')
    network = torch.load(network_filename)
    network = network.cuda()
# new network
else:
    print('No network found in directory.')
    print('Training new network.')
    n_iters = 65000  # 2500*26 = 65000
    print_every = n_iters/20
    plot_every = 2*print_every

    all_rotations, rotation_lines, n_rotations = init_data(dir_training)

    n_hidden = 128
    network = Network(n_letters, n_hidden, n_rotations)
    network = network.cuda()
    criterion = nn.NLLLoss()

    base_learning_rate = 0.002 # If you set this too high, it might explode. If too low, it might not learn
    min_steps = 1000
    lower_learn_rate_thresholds = get_thresholds(n_iters, min_steps)

    network, nan_error_occured = train(all_rotations, print_every, plot_every, lower_learn_rate_thresholds, criterion)

    if nan_error_occured:
        exit()
    if save_network:
        torch.save(network, network_filename)

n_tests = 10000

print('Testing RNN with', n_tests, 'words and sentences.')
n_correct, n_wrong, mismatches = test_network(dir_test, network, n_tests)
evaluate(n_correct, n_wrong, mismatches)
