# -*-coding=utf8-*-
import pandas as pd
import columns as cls
import numpy as np
import os
import sys 
import codecs


sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
dict_map = {}


def get_train_data(source_data):
    source_data.columns = cls.columns
    #source_data['u_age'] = source_data['u_age'].map(lambda x: age_discret(x))
    source_data = transform_category(source_data)
    source_data = source_data.drop(cls.drop_columns, axis=1)
    source_data.dropna()
    print(source_data.columns)
    return getXY(source_data)

def transform_category(source_data):
    get_dict_map()
    for column in cls.category_columns:
        if column in cls.drop_columns:
            continue
        if column not in dict_map.keys():
            print("dict not found %s" % column)
            return None
        else:
            print("="*50)
            print("transform %s" %  column)
            print("before :column %s", source_data.head()[column])
            print("-"*50)
            source_data[column] = source_data[column].map(lambda x: get_index(column, x))
            print("after: column %s", source_data.head()[column])
    return source_data

def getXY(source_data):
    Y = source_data['label']
    X = source_data.drop(['label', 'weight'], axis=1)
    X = X.astype(float)
    print(source_data.columns)
    return X, Y

def get_dict_map():
    dict_path = "./../data/dict"
    files = os.listdir(dict_path)
    global dict_map
    for file in files:
        if file.endswith(".dict"):
            index = 0
            value_list = []
            for line in open(os.path.join(dict_path, file), encoding="utf8"):
                if index >= 1000:
                    break
                value_list.append(line.split('|')[0])
                index += 1
            dict_map[file.split('.')[0]] = value_list

def get_index(column, value):
    global dict_map
    value_list = dict_map[column]
    if value not in value_list:
        return 0
    else:
        return dict_map[column].index(value) + 1


def age_discret(age):
    if age < 6:
        return 0
    elif age < 12:
        return 1
    elif age < 15:
        return 2
    elif age < 18:
        return 3
    elif age < 22:
        return 4
    elif age < 26:
        return 5
    elif age < 30:
        return 6
    elif age < 35:
        return 7
    elif age < 40:
        return 8
    else:
        return 9

if __name__ == '__main__':
    source_data = pd.read_csv('./../data/data/train.csv', header=None, sep='\t')
    parsed = get_train_data(source_data)
    pd.set_option('display.max_columns', None)
    print(parsed[0].head())
