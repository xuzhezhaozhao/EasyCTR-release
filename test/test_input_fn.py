#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf
from common import transform
from common.common import parse_scheme
from easyctr import input_fn


class Dummy():
    pass


opts = Dummy()
opts.assembler_ops_path = './build/ops/libassembler_ops.so'
opts.conf_path = './data/conf.json'
opts.eval_batch_size = 1
opts.batch_size = 1
opts.map_num_parallel_calls = 1
opts.prefetch_size = 1
opts.shuffle_batch = False
opts.read_buffer_size_mb = None
opts.use_spark_fuel = False
opts.compression_type = ''
opts.num_parallel_reads = 1
opts.use_lua_sampler = True
opts.lua_sampler_script = './test/lua/sampler.v1.lua'

scheme = parse_scheme(
    conf_path=opts.conf_path,
    ops_path=opts.assembler_ops_path,
    use_spark_fuel=False)

conf = json.loads(open(opts.conf_path).read())

trans = transform.Transform(conf, scheme)

fn = input_fn.input_fn(opts, './data/train.txt', False, False, scheme)
ds = fn()
iterator = ds.make_initializable_iterator()
features_tensor, labels_tensor = iterator.get_next()

input_layer_tensor = tf.feature_column.input_layer(features_tensor, trans.deep_columns)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer)
    for _ in range(2):
        print("##########################################")
        features, labels, input_layer = sess.run([features_tensor, labels_tensor, input_layer_tensor])
        # sess.run(features_tensor)

        # print("features:")
        # for key in features:
            # print("iname = {}, value = {}".format(key, features[key]))

        print("labels = {}".format(labels))
        print("weights = {}".format(features['_weight_column']))

        # print("input_layer:")
        # print(input_layer)
        # print("##########################################")
        # print("\n\n")
