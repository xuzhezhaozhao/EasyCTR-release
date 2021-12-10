#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.common import add_hidden_layer_summary
from common.common import check_arg
from common.common import get_feature_vectors
from common.common import project


"""
Product-based Neural Networks for User Response Prediction over Multi-field Categorical Data

Kernel FM
"""


def _check_kfm_args(args):
    check_arg(args, 'kfm_use_shared_embedding', bool)
    check_arg(args, 'kfm_use_project', bool)
    check_arg(args, 'kfm_project_size', int)


def get_kfm_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('kfm'):
        _check_kfm_args(extra_options)
        use_shared_embedding = extra_options['kfm_use_shared_embedding']
        use_project = extra_options['kfm_use_project']
        project_size = extra_options['kfm_project_size']

        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        if use_project:
            feature_vectors = project(feature_vectors, project_size)

        y = _kfm(feature_vectors, reduce_sum=True)  # [B, 1]
        with tf.variable_scope('logits') as logits_scope:
            logits = y
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _kfm(feature_vectors, reduce_sum=True):
    """Kernel FM
      feature_vectors: List of shape [B, ?] tensors, size N

    Half-Optimized implementation

    Return:
      Tensor of shape [B, T] if reduce_sum is True, or shape [B, 1], T is the sum
      dimentions of all features.
    """

    with tf.variable_scope('kfm'):
        outputs = []
        x = tf.concat(feature_vectors, axis=1)   # [B, T]
        T = x.shape[1].value
        N = len(feature_vectors)
        for i in range(N):
            vi = feature_vectors[i]
            name = 'kfm_{}'.format(i)
            di = vi.shape[1].value
            U = tf.get_variable(name, [T, di])
            y = tf.matmul(x, U)   # [B, di]
            outputs.append(y)
        y = tf.concat(outputs, axis=1)   # [B, T]
        y = x * y  # [B, T]

        if reduce_sum:
            y = tf.reduce_sum(y, axis=1, keepdims=True)  # [B, 1]

        return y
