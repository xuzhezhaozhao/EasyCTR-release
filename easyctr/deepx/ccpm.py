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
from common.common import check_feature_dims
from common.common import k_max_pooling
from common.common import get_activation_fn
from common.common import get_feature_vectors
from common.common import project


def _check_ccpm_args(args):
    check_arg(args, 'ccpm_use_shared_embedding', bool)
    check_arg(args, 'ccpm_use_project', bool)
    check_arg(args, 'ccpm_project_size', int)
    check_arg(args, 'ccpm_hidden_units', list)
    check_arg(args, 'ccpm_activation_fn', str)
    check_arg(args, 'ccpm_dropout', float)
    check_arg(args, 'ccpm_batch_norm', bool)
    check_arg(args, 'ccpm_layer_norm', bool)
    check_arg(args, 'ccpm_use_resnet', bool)
    check_arg(args, 'ccpm_use_densenet', bool)
    check_arg(args, 'ccpm_kernel_sizes', list)
    check_arg(args, 'ccpm_filter_nums', list)

    check_arg(args, 'leaky_relu_alpha', float)
    check_arg(args, 'swish_beta', float)


def get_ccpm_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('ccpm'):
        _check_ccpm_args(extra_options)
        use_shared_embedding = extra_options['ccpm_use_shared_embedding']
        use_project = extra_options['ccpm_use_project']
        project_size = extra_options['ccpm_project_size']
        hidden_units = extra_options['ccpm_hidden_units']
        activation_fn = extra_options['ccpm_activation_fn']
        dropout = extra_options['ccpm_dropout']
        batch_norm = extra_options['ccpm_batch_norm']
        layer_norm = extra_options['ccpm_layer_norm']
        use_resnet = extra_options['ccpm_use_resnet']
        use_densenet = extra_options['ccpm_use_densenet']
        kernel_sizes = extra_options['ccpm_kernel_sizes']
        filter_nums = extra_options['ccpm_filter_nums']
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

        y = _build_ccpm_model(feature_vectors=feature_vectors,
                              kernel_sizes=kernel_sizes,
                              filter_nums=filter_nums,
                              hidden_units=hidden_units,
                              activation_fn=activation_fn,
                              dropout=dropout,
                              is_training=is_training,
                              batch_norm=batch_norm,
                              layer_norm=layer_norm,
                              use_resnet=use_resnet,
                              use_densenet=use_densenet)

        with tf.variable_scope('logits') as logits_scope:
            logits = fc(y, units=units, name=logits_scope)
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _build_ccpm_model(feature_vectors,
                      kernel_sizes,
                      filter_nums,
                      hidden_units,
                      activation_fn,
                      dropout,
                      is_training,
                      batch_norm,
                      layer_norm,
                      use_resnet,
                      use_densenet):
    """Build ccpm model
     feature_vectors: List of 2D tensors

    Return:
      Tensor of shape [B, ?]
    """
    assert len(kernel_sizes) == len(filter_nums)
    check_feature_dims(feature_vectors)

    with tf.variable_scope("ccpm"):
        with tf.variable_scope('cnn'):
            x = tf.stack(feature_vectors, axis=1)  # [B, N, D]
            L = len(kernel_sizes)
            N = x.shape[1].value
            D = x.shape[2].value
            y = tf.expand_dims(x, -1)   # [B, N, D, 1]
            for idx, (width, num) in enumerate(zip(kernel_sizes, filter_nums)):
                name = 'conv_{}_w{}_n{}'.format(idx, width, num)
                y = tf.layers.conv2d(y,
                                     filters=num,
                                     kernel_size=(width, D),
                                     strides=(1, 1),
                                     padding='SAME',
                                     activation=activation_fn,
                                     name=name)
                if idx == L - 1:
                    p = 3
                else:
                    p = (1 - (1.0 * idx / L)**(L - idx)) * N
                p = int(p)
                p = min(p, y.shape[1].value)
                name = 'k_max_pooling_{}_k{}'.format(idx, p)
                y = k_max_pooling(y, k=p, axis=1, name=name)

            y = tf.layers.flatten(y)  # [B, -1]

        # dnn
        y = add_hidden_layers(y,
                              hidden_units,
                              activation_fn=activation_fn,
                              dropout=dropout,
                              is_training=is_training,
                              batch_norm=batch_norm,
                              layer_norm=layer_norm,
                              use_resnet=use_resnet,
                              use_densenet=use_densenet,
                              scope='dnn')
        return y
