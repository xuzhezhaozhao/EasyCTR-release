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

user_meta = "user.age\tuser.gender\tuser.city"
context_meta = "action.time\taction.os\taction.network"
item_meta = "item.click_rate\titem.eplay_rate"

user_feature = "11\t0\tbj"
context_feature = "21\t14.4"
item_feature = ["0.11\t0.45", "0.13\t0.56", ""]
num_valid = 2

serialized = assembler_ops.assembler_serialize(conf_path=conf_path)
with tf.Session() as sess:
    serialized = sess.run(serialized)

output = assembler_ops.assembler_serving(
    user_meta=user_meta,
    context_meta=context_meta,
    item_meta=item_meta,
    user_feature=user_feature,
    context_feature=context_feature,
    item_feature=item_feature,
    num_valid=num_valid,
    serialized=serialized)

with tf.Session() as sess:
    output = sess.run(output)
    print(output)
