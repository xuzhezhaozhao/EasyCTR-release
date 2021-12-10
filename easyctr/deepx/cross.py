#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from common.common import fc
from common.common import add_hidden_layer_summary
from common.common import check_arg
from common.common import get_feature_vectors
from common.common import project


"""
Deep & Cross Network for Ad Click Predictions
"""


def _check_cross_args(args):
    check_arg(args, 'cross_use_shared_embedding', bool)
    check_arg(args, 'cross_use_project', bool)
    check_arg(args, 'cross_project_size', int)
    check_arg(args, 'cross_num_layers', int)


def get_cross_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('cross'):
        _check_cross_args(extra_options)
        use_shared_embedding = extra_options['cross_use_shared_embedding']
        use_project = extra_options['cross_use_project']
        project_size = extra_options['cross_project_size']
        num_layers = extra_options['cross_num_layers']

        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        if use_project:
            feature_vectors = project(feature_vectors, project_size)

        x = tf.concat(feature_vectors, axis=1)  # [B, T]
        y = _cross_net(x, num_layers)
        with tf.variable_scope('logits') as logits_scope:
            logits = fc(y, units=units, name=logits_scope)
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _cross_net(x, num_layers):
    """Cross Network
      x: Input tensor, shape [B, T]
      num_layers: number of cross layers

    Return:
      Tensor of shape [B, T]
    """
    with tf.variable_scope('cross'):
        dim = x.shape[-1].value
        x0 = tf.expand_dims(x, axis=2)  # [B, T, 1]
        xl = x0  # [B, T, 1]
        for layer_id in range(num_layers):
            with tf.variable_scope('cross_{}'.format(layer_id)):
                w_name = 'cross_w_{}'.format(layer_id)
                bias_name = 'cross_bias_{}'.format(layer_id)
                w = tf.get_variable(w_name, [dim, 1])
                bias = tf.get_variable(bias_name,
                                       shape=[dim, 1],
                                       initializer=tf.initializers.zeros)
                y = tf.tensordot(xl, w, axes=(1, 0))  # [B, 1, 1]
                y = tf.matmul(x0, y)  # [B, T, 1]
                y = y + bias + xl  # [B, T, 1]
                xl = y  # [B, T, 1]
        xl = tf.squeeze(xl, axis=2)  # [B, T]

        return xl
