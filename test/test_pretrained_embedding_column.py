#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

embed = tf.constant(
    [
        [1., 2., 3., 4., 5.],
        [6., 7., 8., 9., 10.],
        [11., 12., 13., 14., 15.]
    ]
)


def initializer(shape=None, dtype=tf.float32, partition_info=None):
    assert dtype is tf.float32
    return embed


name = tf.feature_column.categorical_column_with_vocabulary_list(
    'name', ['bob', 'george', 'wanda'], default_value=-1)
embedding = tf.feature_column.embedding_column(
    name, 5, initializer=initializer)

features = {
    'name': tf.constant([
        ['bob', '', ''],
        ['bob', 'bob', ''],
        ['xxx', '', ''],
        ['george', '', ''],
        ['george', 'bob', 'wanda'],
        ['wanda', 'george', ''],
    ])
}
dense_tensor = tf.feature_column.input_layer(features, [embedding])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(dense_tensor))
