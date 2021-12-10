#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from common.common import add_hidden_layer_summary
from common.common import add_hidden_layers
from common.common import fc
from common.common import check_arg
from common.common import get_activation_fn
from common.common import get_feature_vectors
from common.common import project


"""
Product-based Neural Networks for User Response Prediction over Multi-field Categorical Data
"""


def _check_pin_args(args):
    check_arg(args, 'pin_use_shared_embedding', bool)
    check_arg(args, 'pin_use_project', bool)
    check_arg(args, 'pin_project_size', int)
    check_arg(args, 'pin_hidden_units', list)
    check_arg(args, 'pin_activation_fn', str)
    check_arg(args, 'pin_dropout', float)
    check_arg(args, 'pin_batch_norm', bool)
    check_arg(args, 'pin_layer_norm', bool)
    check_arg(args, 'pin_use_resnet', bool)
    check_arg(args, 'pin_use_densenet', bool)
    check_arg(args, 'pin_use_concat', bool)
    check_arg(args, 'pin_concat_project', bool)
    check_arg(args, 'pin_subnet_hidden_units', list)

    check_arg(args, 'leaky_relu_alpha', float)
    check_arg(args, 'swish_beta', float)


def get_pin_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('pin'):
        _check_pin_args(extra_options)
        use_shared_embedding = extra_options['pin_use_shared_embedding']
        use_project = extra_options['pin_use_project']
        project_size = extra_options['pin_project_size']
        hidden_units = extra_options['pin_hidden_units']
        activation_fn = extra_options['pin_activation_fn']
        dropout = extra_options['pin_dropout']
        batch_norm = extra_options['pin_batch_norm']
        layer_norm = extra_options['pin_layer_norm']
        use_resnet = extra_options['pin_use_resnet']
        use_densenet = extra_options['pin_use_densenet']
        use_concat = extra_options['pin_use_concat']
        concat_project = extra_options['pin_concat_project']
        subnet_hidden_units = extra_options['pin_subnet_hidden_units']
        leaky_relu_alpha = extra_options['leaky_relu_alpha']
        swish_beta = extra_options['swish_beta']
        activation_fn = get_activation_fn(activation_fn=activation_fn,
                                          leaky_relu_alpha=leaky_relu_alpha,
                                          swish_beta=swish_beta)
        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        project_feature_vectors = None
        if use_project:
            project_feature_vectors = project(feature_vectors, project_size)

        y = _pin(feature_vectors=feature_vectors,
                 project_feature_vectors=project_feature_vectors,
                 use_project=use_project,
                 subnet_hidden_units=subnet_hidden_units,
                 units=units,
                 hidden_units=hidden_units,
                 activation_fn=activation_fn,
                 dropout=dropout,
                 batch_norm=batch_norm,
                 layer_norm=layer_norm,
                 use_resnet=use_resnet,
                 use_densenet=use_densenet,
                 is_training=is_training,
                 concat_project=concat_project,
                 use_concat=use_concat)

        with tf.variable_scope('logits') as logits_scope:
            logits = fc(y, units=units, name=logits_scope)
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _pin(feature_vectors,
         project_feature_vectors,
         use_project,
         subnet_hidden_units,
         units,
         hidden_units,
         activation_fn,
         dropout,
         batch_norm,
         layer_norm,
         use_resnet,
         use_densenet,
         is_training,
         concat_project,
         use_concat):
    """
      feature_vectors: List of shape [B, ?] tensors, size N
      project_feature_vectors: None or list of shape [B, D] tensors, size N

    Return:
      2D Tensor of shape [B, ?]
    """

    with tf.variable_scope("pin"):
        if use_project:
            x = project_feature_vectors
        else:
            x = feature_vectors

        y = _pin_layer(x, subnet_hidden_units, is_training)  # [B, ?]

        inputs = []
        inputs.append(y)
        if use_concat:
            if use_project and concat_project:
                inputs.extend(project_feature_vectors)
            else:
                inputs.extend(feature_vectors)
        y = tf.concat(inputs, axis=1)   # [B, ?]
        y = add_hidden_layers(y,
                              hidden_units=hidden_units,
                              activation_fn=activation_fn,
                              dropout=dropout,
                              is_training=is_training,
                              batch_norm=batch_norm,
                              layer_norm=layer_norm,
                              use_resnet=use_resnet,
                              use_densenet=use_densenet,
                              scope='hidden_layers')  # [B, ?]
        return y


def _pin_layer(feature_vectors, subnet_hidden_units, is_training):
    """PIN Layer
      feature_vectors: List of shape [B, ?] tensors, size N

    Return:
     2D Tensor of shape [B, ?]
    """

    with tf.variable_scope('pin_layer'):
        N = len(feature_vectors)
        outputs = []
        for i in range(N):
            for j in range(N):
                vi = feature_vectors[i]
                vj = feature_vectors[j]
                di = vi.shape[1].value
                dj = vj.shape[1].value
                kernel_name = 'pairwise_kernel_dot_{}_{}'.format(i, j)
                U = tf.get_variable(kernel_name, [di, dj])
                y = tf.matmul(feature_vectors[i], U)  # [B, dj]
                y = tf.concat([vi, vj, y], axis=1)
                scope_name = 'subnet_{}_{}'.format(i, j)
                y = add_hidden_layers(y,
                                      hidden_units=subnet_hidden_units,
                                      activation_fn=tf.nn.relu,
                                      dropout=None,
                                      is_training=is_training,
                                      batch_norm=True,
                                      layer_norm=False,
                                      use_resnet=False,
                                      use_densenet=False,
                                      scope=scope_name,
                                      last_layer_direct=True)
                outputs.append(y)
        y = tf.concat(outputs, axis=1)   # [B, ?]
        return y
