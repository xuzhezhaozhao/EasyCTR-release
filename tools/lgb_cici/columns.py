# coding: utf-8
import argparse
import pandas as pd
import feature as ft
import lightgbm as lgb
import columns as cl
import numpy as np
import time

meta_file = "./data.meta"

columns = ['label', 'weight']
#drop_columns = ['u_omgid', 'i_rowkey', 'c_first_item_id', 'i_kb_media_id', 'e_play_duration', 'e_video_duration', 'e_scene']
drop_columns = ['e_play_duration','e_video_duration','e_scene']
category_columns = []
strlist_columns = []
dtypes = {'label': np.float64, 'weight': np.float64}

def get_top_3(ftname):
    dict_path = './../data/dict/%s.dict' %ftname
    top_three = {}
    count = 1
    with open(dictpath,mode = 'r') as f:
        for line in f:
            line = line.split('|')
            topelem = line[0]
            top_three[topelem] = count
            if len(top_three) == 3:
                return top_three
            count += 1
    return None
    
for line in open(meta_file):
    line = line.strip()
    if line == "":
        continue
    tokens = line.split()
    name = tokens[1]
    t = tokens[2]
    columns.append(tokens[1])
    if t.startswith('e_'):
        drop_columns.append(tokens[1])
    elif t == 'int':
        #dtypes[name] = np.int64
        dtypes[name] = np.string_
    elif t == 'float':
        dtypes[name] = np.float64
    elif t == 'string':
        dtypes[name] = np.string_
        category_columns.append(tokens[1])
    #elif t == 'string_list':
        #dtypes[name] = np.float64
        #strlist_columns.append(name)
        #category_columns.append(tokens[1])
    # turn string list into type that can be accepted by the lgbm
    else:
        drop_columns.append(tokens[1])

print(dtypes)
new_category_columns = []
for column in category_columns:
    if column not in drop_columns:
        new_category_columns.append(column)
category_columns = new_category_columns
target_label = 'label'

