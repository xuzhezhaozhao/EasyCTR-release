#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.common import add_hidden_layers
from common.common import add_hidden_layer_summary
from common.common import fc
from common.common import check_arg
from common.common import get_feature_vectors
from common.common import project
from common.common import get_activation_fn


"""
Product-based Neural Networks for User Response Prediction over Multi-field Categorical Data
"""


def _check_nifm_args(args):
    check_arg(args, 'nifm_use_shared_embedding', bool)
    check_arg(args, 'nifm_use_project', bool)
    check_arg(args, 'nifm_project_size', int)
    check_arg(args, 'nifm_hidden_units', (list, tuple))
    check_arg(args, 'nifm_activation_fn', str)
    check_arg(args, 'nifm_dropout', float)
    check_arg(args, 'nifm_batch_norm', bool)
    check_arg(args, 'nifm_layer_norm', bool)
    check_arg(args, 'nifm_use_resnet', bool)
    check_arg(args, 'nifm_use_densenet', bool)

    check_arg(args, 'leaky_relu_alpha', float)
    check_arg(args, 'swish_beta', float)


def get_nifm_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('nifm'):
        _check_nifm_args(extra_options)
        use_shared_embedding = extra_options['nifm_use_shared_embedding']
        use_project = extra_options['nifm_use_project']
        project_size = extra_options['nifm_project_size']
        hidden_units = extra_options['nifm_hidden_units']
        activation_fn = extra_options['nifm_activation_fn']
        dropout = extra_options['nifm_dropout']
        batch_norm = extra_options['nifm_batch_norm']
        layer_norm = extra_options['nifm_layer_norm']
        use_resnet = extra_options['nifm_use_resnet']
        use_densenet = extra_options['nifm_use_densenet']

        leaky_relu_alpha = extra_options['leaky_relu_alpha']
        swish_beta = extra_options['swish_beta']

        activation_fn = get_activation_fn(activation_fn=activation_fn,
                                          leaky_relu_alpha=leaky_relu_alpha,
                                          swish_beta=swish_beta)

        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        if use_project:
            feature_vectors = project(feature_vectors, project_size)

        y = _nifm(feature_vectors=feature_vectors,
                  hidden_units=hidden_units,
                  activation_fn=activation_fn,
                  dropout=dropout,
                  is_training=is_training,
                  batch_norm=batch_norm,
                  layer_norm=layer_norm,
                  use_resnet=use_resnet,
                  use_densenet=use_densenet,
                  reduce_sum=True)  # [B, 1]
        with tf.variable_scope('logits') as logits_scope:
            logits = y
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _nifm(feature_vectors,
          hidden_units,
          activation_fn,
          dropout,
          is_training,
          batch_norm,
          layer_norm,
          use_resnet,
          use_densenet,
          reduce_sum=True):
    """Network in FM
      feature_vectors: List of shape [B, D] tensors, size N

    Return:
      Tensor of shape [B, 1] if reduce_sum is True, or shape of [B, N*(N-1)/2]
    """

    with tf.variable_scope('nifm'):
        outputs = []
        N = len(feature_vectors)
        for i in range(N-1):
            for j in range(i, N):
                scope_name = 'subnet_{}_{}'.format(i, j)
                vi = feature_vectors[i]
                vj = feature_vectors[j]
                v = tf.concat([vi, vj], axis=1)
                y = add_hidden_layers(v,
                                      hidden_units=hidden_units,
                                      activation_fn=tf.nn.relu,
                                      dropout=dropout,
                                      is_training=is_training,
                                      batch_norm=batch_norm,
                                      layer_norm=layer_norm,
                                      use_resnet=use_resnet,
                                      use_densenet=use_densenet,
                                      scope=scope_name)
                y = fc(y, 1)
                outputs.append(y)

        y = tf.concat(outputs, axis=1)  # [B, N*(N-1)/2]
        if reduce_sum:
            y = tf.reduce_sum(y, axis=-1, keepdims=True)  # [B, 1]

        tf.logging.info("nifm output = {}".format(y))
        return y
