#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import sys
import numpy as np

# Limit output to 3 decimal places.
pd.options.display.float_format = '{:,.4f}'.format
pd.options.display.max_rows = 1000


if len(sys.argv) == 3:
    nrows = None
elif len(sys.argv) == 4:
    nrows = int(sys.argv[3])
else:
    print("Usage: <meta_file> <data_file> [nrows]\n")
    sys.exit(-1)

meta_file = sys.argv[1]
data_file = sys.argv[2]


bancols = [
    'u_omgid',
    'u_data_date',
    'u_history',
    'u_h_top_tags',
    'u_h_top_tags_valid_ratio',
    'u_h_top_cat1',
    'u_h_top_cat1_valid_ratio',
    'u_h_top_cat2',
    'u_h_top_cat2_valid_ratio',
    'u_h_top_cat3',
    'u_h_top_cat3_valid_ratio',
    'u_h_neg_tags',
    'u_h_neg_tags_valid_ratio',
    'u_h_neg_cat1',
    'u_h_neg_cat1_valid_ratio',
    'u_h_neg_cat2',
    'u_h_neg_cat2_valid_ratio',
    'u_h_neg_cat3',
    'u_h_neg_cat3_valid_ratio',
    'i_rowkey',
]
columns = ['label', 'weight']
dtypes = {'label': np.float64, 'weight': np.float64}
for line in open(meta_file):
    line = line.strip()
    if line == "":
        continue
    tokens = line.split()
    name = tokens[1]
    t = tokens[2]
    columns.append(tokens[1])
    if t == 'int':
        dtypes[name] = np.float64
    elif t == 'float':
        dtypes[name] = np.float64
    else:
        dtypes[name] = 'category'

total_usecols = []
for col in columns:
    if col not in bancols:
        total_usecols.append(col)

print("columns = \n{}".format(columns))
print("total_usecols = \n{}".format(total_usecols))
print("dtypes = \n{}".format(dtypes))
sep = '\t'

chunk = 20
sz = int((len(total_usecols) + chunk - 1) / chunk)
print('#chunk = {}'.format(sz))
for i in range(sz):
    usecols = total_usecols[i*chunk:(i+1)*chunk]
    usecols.append('label')
    df = pd.read_csv(data_file, sep=sep, header=None, names=columns, dtype=dtypes,
                     usecols=usecols, nrows=nrows)
    percentiles = np.arange(.01, 1.0, .01)
    for name in usecols[:-1]:
        print("****** {} ********".format(name))
        if df[name].dtype == np.number:
            print("label corr = {}".format(df['label'].corr(df[name])))
            print("median = {}".format(df[name].median()))
        print(df[name].describe(percentiles=percentiles))
        print("\n")
