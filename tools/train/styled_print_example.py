#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


if len(sys.argv) != 2:
    print("Usgae: <meta>")
    sys.exit(-1)


meta_file = sys.argv[1]


fields = []
for line in open(meta_file):
    line = line.strip()
    if line == "" or line[0] == '#':
        continue
    tokens = line.split()
    field = tokens[0]
    name = tokens[1]
    t = tokens[2]

    if field.startswith('user'):
        pos0 = 2
    elif field.startswith('item'):
        pos0 = 3
    elif field.startswith('ctx'):
        pos0 = 4
    elif field.startswith('extra'):
        pos0 = 5
    else:
        print("meta error")
        sys.exit(-1)

    pos1 = int(tokens[3])
    pos2 = int(tokens[4])
    if t == 'int':
        func = float
    elif t == 'float':
        func = float
    else:
        func = str
    fields.append((name, pos0, pos1, pos2, func))


def total_split(input):
    first_tokens = line.split('|')
    tokens = [None for _ in range(len(first_tokens))]
    for first_idx, first_token in enumerate(first_tokens):
        second_tokens = first_token.split('$')
        tokens[first_idx] = [None for _ in range(len(second_tokens))]
        for second_idx, second_token in enumerate(second_tokens):
            third_tokens = second_token.split('\t')
            tokens[first_idx][second_idx] = third_tokens

    return tokens


for line in sys.stdin:
    line = line.strip()
    print("-------------------------")
    tokens = total_split(line)
    for field in fields:
        name, pos0, pos1, pos2, func = field
        v = tokens[pos0][pos1][pos2]
        print("{} = {}".format(name, v))
    print("-------------------------\n")
