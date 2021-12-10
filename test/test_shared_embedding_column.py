#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

initial = tf.fill([3, 5], 1.0)
name1 = tf.feature_column.categorical_column_with_vocabulary_list(
    'name1', ['bob', 'george', 'wanda'], default_value=-1)
name2 = tf.feature_column.categorical_column_with_vocabulary_list(
    'name2', ['bob', 'george', 'wanda'], default_value=-1)

embeddings = tf.feature_column.shared_embedding_columns([name1, name2], 5, initializer=None)

features = {
    'name1': tf.constant([
        ['george', 'george', 'george']]),
    'name2': tf.constant([
        ['bob', '', '']
    ])
}

print(embeddings[0].name)
print(embeddings[1].key)
dense_tensor = tf.feature_column.input_layer(features, embeddings[0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(dense_tensor))
