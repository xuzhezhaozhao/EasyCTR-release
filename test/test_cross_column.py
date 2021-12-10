#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

column1 = tf.feature_column.categorical_column_with_vocabulary_list(
        'name1', [0, 1, 2, 3], default_value=-1)
column2 = tf.feature_column.categorical_column_with_vocabulary_list(
        'name2', [0, 1, 2, 3], default_value=-1)

cross_column = tf.feature_column.crossed_column([column1, column2], 100)
indicator_column = tf.feature_column.indicator_column(cross_column)

features = {
    'name1': tf.constant([[0]]),
    'name2': tf.constant([[0]]),
}

indicator_tensor = tf.feature_column.input_layer(features, [indicator_column])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(indicator_tensor))
