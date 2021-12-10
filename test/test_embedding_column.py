#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

initial = tf.fill([3, 5], 1.0)
name = tf.feature_column.categorical_column_with_vocabulary_list(
    'name', ['bob', 'george', 'wanda'], default_value=-1)
embedding = tf.feature_column.embedding_column(name, 5, initializer=None)

features = {
    'name': tf.constant([
        ['bob', '', ''],
        ['bob', 'bob', ''],
        ['xxx', '', ''],
        ['george', '', ''],
        ['george', 'george', 'george']])
}
print(embedding)
dense_tensor = tf.feature_column.input_layer(features, [embedding])
dense_tensor2 = tf.feature_column.input_layer(features, [embedding])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(dense_tensor))
    print(sess.run(dense_tensor2))
