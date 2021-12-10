#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random
from utils import logger

if len(sys.argv) != 4:
    logger.info("Usgae: <datafile> <output> <neg_num>")
    sys.exit(-1)

datafile = sys.argv[1]
output = sys.argv[2]
neg_num = int(sys.argv[3])

negs = []
for line in open(datafile):
    line = line.strip()
    tokens = line.split('|')[3:]
    neg = '|'.join(tokens)
    negs.append(neg)


with open(output, 'w') as f:
    for line in open(datafile):
        line = line.strip()
        tokens = line.split('|')
        f.write(line + '\n')
        if tokens[0] == '1':
            part = '0|1.0|' + tokens[2] + '|'
            for _ in range(neg_num):
                idx = random.randint(0, len(negs)-1)
                f.write(part)
                f.write(negs[idx] + '\n')
