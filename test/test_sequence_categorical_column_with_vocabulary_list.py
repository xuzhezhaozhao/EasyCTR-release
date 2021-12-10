#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

column = tf.feature_column.sequence_categorical_column_with_vocabulary_list(
        'name', [0, 1, 2, 3], default_value=-1)

indicator_column = tf.feature_column.indicator_column(column)

# 测试处理 oov 词的情况
features = {
    'name': tf.constant([[0, 1, 9, 10, 11, 12, 13, 14]]),
}


indicator_tensor = tf.feature_column.input_layer(features, [indicator_column])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(indicator_tensor))
