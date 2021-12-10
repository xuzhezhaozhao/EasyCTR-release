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

serialized = assembler_ops.assembler_serialize(conf_path=conf_path)
serialized = sess.run(serialized)

print("dssm serving ...")
user_feature = assembler_ops.assembler_dssm_serving(
    user_feature=[''],
    serialized=serialized)
print("dssm serving done")

user_feature_output = sess.run(user_feature)
print(user_feature_output.shape)
print(user_feature_output)
