#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

age = tf.feature_column.numeric_column('age')
bucketized = tf.feature_column.bucketized_column(age, [23])
features = {
    'age': tf.constant([[23], [22]]),
}
dense_tensor = tf.feature_column.input_layer(features, [bucketized])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(dense_tensor))
