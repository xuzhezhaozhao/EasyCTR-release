#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

if len(sys.argv) != 3:
    print("<Usage>: <assembler_ops_path> <conf>")
    sys.exit(-1)

assembler_ops_path = sys.argv[1]
conf_path = sys.argv[2]

sess = tf.Session()
assembler_ops = tf.load_op_library(assembler_ops_path)

# test_line = ['']
# output = assembler_ops.assembler(test_line, conf_path=conf_path)
# print(sess.run(output))

# output = assembler_ops.assembler_scheme(conf_path=conf_path)
# print(sess.run(output))

serialized = assembler_ops.assembler_serialize(conf_path=conf_path)
serialized = sess.run(serialized)
user_feature = assembler_ops.assembler_dssm_serving(
    user_feature=[''],
    serialized=serialized)

user_feature_output = sess.run([user_feature])
print(user_feature_output.shape)
print(user_feature_output)
