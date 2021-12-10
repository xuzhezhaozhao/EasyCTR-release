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

assembler_ops = tf.load_op_library(assembler_ops_path)
test_line = ['u_001	i_003,0,18,16	i_001,0,11,12	i_004,1,19,6	i_008,1,12,11']
sess = tf.Session()
output = assembler_ops.assembler(test_line, conf_path=conf_path)
print(sess.run(output))

serialized = assembler_ops.assembler_serialize(conf_path=conf_path)
serialized = sess.run(serialized)
print(len(serialized))
