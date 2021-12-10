#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys


if len(sys.argv) != 3:
    print("usage: <model_dir> <shrink_rate>")
    sys.exit(-1)

model_dir = sys.argv[1]
rate = float(sys.argv[2])

checkpoint_path = tf.train.latest_checkpoint(model_dir)

if checkpoint_path is None or checkpoint_path == '':
    print("adagrad_shrinkage: No valid checkpoint")
    sys.exit(0)

reader = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

tensor_map = dict()
var_map = dict()
var_plhdr = dict()

for key in var_to_shape_map:
    tensor_map[key] = reader.get_tensor(key)
    if "/Adagrad" in key:
        print("adagrad_shrinkage: {}".format(key))
        tensor_map[key] = np.maximum(tensor_map[key] * rate, 0.1)

    var_type = tf.dtypes.as_dtype(tensor_map[key].dtype)
    var_map[key] = tf.get_variable(key, var_to_shape_map[key], dtype=var_type)
    var_plhdr[key] = tf.placeholder(dtype=var_type, shape=var_to_shape_map[key])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for key in var_map:
        sess.run(var_map[key].assign(var_plhdr[key]),
                 {var_plhdr[key]: tensor_map[key]})
    tf.train.Saver().save(sess, checkpoint_path,
                          write_meta_graph=False,
                          write_state=False)
