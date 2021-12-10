#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

name_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'name', [0, 1, 2, 3], default_value=-1)

weighted_column = tf.feature_column.weighted_categorical_column(
    categorical_column=name_column, weight_feature_key='frequencies')

indicator_column = tf.feature_column.indicator_column(weighted_column)
embedding_column = tf.feature_column.embedding_column(weighted_column, 5, combiner='mean')

# 测试处理 oov 词的情况
# features = {
    # 'name': tf.constant([[0, 1, 9, 10, 11, 12, 13, 14]]),
    # 'frequencies': tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
# }

features = {
    'name': tf.constant([[10, 10, -1]]),
    'frequencies': tf.constant([[0.1, 10.0001, 0]])
}

indicator_tensor = tf.feature_column.input_layer(features, [indicator_column])
embedding_tensor = tf.feature_column.input_layer(features, [embedding_column])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(indicator_tensor))
    print(sess.run(embedding_tensor))
