import os

n_test = 0
n_training = 0
for file in os.listdir("./test data"):
    lines = open("./test data/" + file)
    for line in lines:
        n_test += 1

for file in os.listdir("./training data"):
    lines = open("./training data/" + file)
    for line in lines:
        n_training += 1

print('Test', n_test)
print('Train', n_training)

