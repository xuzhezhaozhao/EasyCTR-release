#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def fn1(x):
    return x


ctr1 = tf.feature_column.numeric_column('ctr', normalizer_fn=fn1)
ctr2 = tf.feature_column.numeric_column('ctr_2', normalizer_fn=fn1)
price = tf.feature_column.numeric_column('price', normalizer_fn=fn1)

features = {
    'ctr': tf.constant([[0.1]]),
    'price': tf.constant([[1.1]]),
}
features['ctr_2'] = features['ctr']

dense_tensor = tf.feature_column.input_layer(features, [ctr1, ctr2, price])
print(dense_tensor)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(dense_tensor))
