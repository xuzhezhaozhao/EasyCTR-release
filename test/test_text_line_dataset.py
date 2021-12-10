#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys


ds = tf.data.TextLineDataset(sys.argv[1:])
iterator = ds.make_initializable_iterator()
line = iterator.get_next()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer)
    while True:
        try:
            output = sess.run(line)
            print(output)
        except tf.errors.OutOfRangeError as e:
            break
