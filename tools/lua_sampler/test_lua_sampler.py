#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf


WEIGHT_COL = '_weight_column'

assembler_ops_path = '/usr/local/services/kd_tools_easy_ctr-1.0/ops/libassembler_ops.so'
if len(sys.argv) == 5:
    assembler_ops_path = sys.argv[4]
elif len(sys.argv) != 4:
    print("usage: <conf_path> <lua_sampler_script> <data_path> [assembler_ops_path]")
    sys.exit(-1)
conf_path = sys.argv[1]
lua_sampler_script = sys.argv[2]
data_path = sys.argv[3]


def parse_line(line, use_negative_sampling):
    assembler_ops = tf.load_op_library(assembler_ops_path)
    feature_dict = {}
    if use_negative_sampling:
        feature, label, weight = assembler_ops.assembler_with_negative_sampling(
            input=line,
            conf_path=conf_path,
            nce_items_path='',  # TODO
            nce_count=5,
            use_lua_sampler=True,
            lua_sampler_script=lua_sampler_script)
    else:
        feature, label, weight = assembler_ops.assembler(
            input=line,
            is_predict=False,
            conf_path=conf_path,
            use_lua_sampler=True,
            lua_sampler_script=lua_sampler_script)
    feature_dict[WEIGHT_COL] = weight
    return feature_dict, label


ds = tf.data.TextLineDataset(data_path)
ds = ds.map(
    lambda line: parse_line(line, use_negative_sampling=False))
ds = ds.flat_map(
    lambda feature_dict, label:
    tf.data.Dataset.from_tensor_slices((feature_dict, label)))
iterator = ds.make_initializable_iterator()
features_tensor, labels_tensor = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer)
    idx = 1
    while True:
        try:
            features, labels = sess.run(
                [features_tensor, labels_tensor])
            print("{}: label = {}, weight = {}".format(
                idx, labels[0], features['_weight_column'][0]))
            idx += 1
        except Exception as e:
            break
