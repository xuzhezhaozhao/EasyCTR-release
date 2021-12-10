#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from common.common import add_hidden_layer_summary
from common.common import check_feature_dims
from common.common import check_arg
from common.common import project
from common.common import get_feature_vectors


def _check_fm_args(args):
    check_arg(args, 'fm_use_shared_embedding', bool)
    check_arg(args, 'fm_use_project', bool)
    check_arg(args, 'fm_project_size', int)


def get_fm_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    assert units == 1, "FM units must be 1"

    with tf.variable_scope('fm'):
        _check_fm_args(extra_options)

        use_shared_embedding = extra_options['fm_use_shared_embedding']
        use_project = extra_options['fm_use_project']
        project_size = extra_options['fm_project_size']

        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        if use_project:
            feature_vectors = project(feature_vectors, project_size)

        y = _fm(feature_vectors, reduce_sum=True)  # [B, 1]
        with tf.variable_scope('logits') as logits_scope:
            logits = y
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _fm(feature_vectors, reduce_sum=True):
    """FM
      feature_vectors: List of shape [B, D] tensors, size N

    Return:
      Tensor of shape [B, 1] if reduce_sum is True, or shape of [B, D]
    """
    with tf.variable_scope('fm'):
        check_feature_dims(feature_vectors)
        stack_fm_vectors = tf.stack(feature_vectors, axis=1)   # [B, N, D]
        sum_square = tf.square(tf.reduce_sum(stack_fm_vectors, axis=1))  # [B, D]
        square_sum = tf.reduce_sum(tf.square(stack_fm_vectors), axis=1)  # [B, D]
        y = 0.5 * (tf.subtract(sum_square, square_sum))  # [B, D]

        if reduce_sum:
            y = tf.reduce_sum(y, axis=1, keepdims=True)  # [B, 1]

        return y
