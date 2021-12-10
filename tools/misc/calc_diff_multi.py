
import json
import sys
import numpy as np


if len(sys.argv) != 4:
    print("Usage: <targetfile> <resultfile1> <resultfile2>  ")
    sys.exit(-1)

targetfile = sys.argv[1]
resultfile1 = sys.argv[2]
resultfile2 = sys.argv[3]

predictions1 = []
for line in open(resultfile1):
    if line == '':
        break
    p = json.loads(line)['predictions'][0]
    predictions1.append(np.e**p - 1)
print('predictions1: {}'.format(len(predictions1)))
print('predictions1 sum = {}'.format(sum(predictions1)))

predictions2 = []
for line in open(resultfile2):
    if line == '':
        break
    p = json.loads(line)['predictions'][0]
    predictions2.append(np.e**p - 1)
print('predictions2: {}'.format(len(predictions2)))
print('predictions2 sum = {}'.format(sum(predictions2)))

assert len(predictions1) == len(predictions2)

predictions = [(x + y) / 2 for (x, y) in zip(predictions1, predictions2)]
print('predictions: {}'.format(len(predictions)))
print('predictions sum = {}'.format(sum(predictions)))

targets = []
for line in open(targetfile):
    if line == '':
        break
    targets.append(np.e**float(line.split()[0][2:]) - 1)

print('targets: {}'.format(len(targets)))
print('targets sum = {}'.format(sum(targets)))

assert len(predictions) == len(targets)

diffs = [abs(x - y) / float(y) for (x, y) in zip(predictions, targets) if y > 5]
print('diff rate = {}, std = {}'.format(np.mean(diffs), np.std(diffs)))
