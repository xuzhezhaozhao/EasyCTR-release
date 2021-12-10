#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

name_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'name', ['bob', 'george', 'wanda', 'abc'])

weighted_column = tf.feature_column.weighted_categorical_column(
    categorical_column=name_column, weight_feature_key='frequencies')

indicator_column = tf.feature_column.indicator_column(weighted_column)

features = {
    'name': tf.constant([['']]),
    'frequencies': tf.constant([[0.0]])
}

dense_tensor = tf.feature_column.input_layer(features, [indicator_column])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(dense_tensor))
