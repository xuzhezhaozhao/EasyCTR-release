#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


"""
crop_embedding 程序对于不在词典的值会填 #, 这个程序将 # 替换为默认值
"""

import numpy as np
import sys


if len(sys.argv) != 3:
    print("Usage: <input> <output>")
    sys.exit(-1)

input_file = sys.argv[1]
output = sys.argv[2]

dim = None
with open(output, 'w') as f:
    for idx, line in enumerate(open(input_file)):
        line = line.strip()
        tokens = line.split()
        if idx == 0:
            dim = int(tokens[1])
            f.write(line + '\n')
            continue
        if len(tokens) == 2 and tokens[1] == '#':
            w = 1.0 / dim
            features = np.random.uniform(-w, w, dim)
            features = map(str, features)
            s = ' '.join(features)
            f.write(tokens[0] + ' ' + s + '\n')
        else:
            if dim + 1 != len(tokens):
                raise ValueError("embedding format error")
            f.write(line + '\n')
