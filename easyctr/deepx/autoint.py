#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.common import check_arg
from common.common import check_feature_dims
from common.common import fc
from common.common import add_hidden_layer_summary
from common.common import get_feature_vectors
from common.common import project


"""
AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks

Borrow code from https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/featureRec/autoint/model.py
"""


def _check_autoint_args(args):
    check_arg(args, 'autoint_use_shared_embedding', bool)
    check_arg(args, 'autoint_use_project', bool)
    check_arg(args, 'autoint_project_size', int)
    check_arg(args, 'autoint_size_per_head', int)
    check_arg(args, 'autoint_num_heads', int)
    check_arg(args, 'autoint_num_blocks', int)
    check_arg(args, 'autoint_dropout', float)
    check_arg(args, 'autoint_has_residual', bool)


def get_autoint_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('autoint'):
        _check_autoint_args(extra_options)

        use_shared_embedding = extra_options['autoint_use_shared_embedding']
        use_project = extra_options['autoint_use_project']
        project_size = extra_options['autoint_project_size']
        size_per_head = extra_options['autoint_size_per_head']
        num_heads = extra_options['autoint_num_heads']
        num_blocks = extra_options['autoint_num_blocks']
        dropout = extra_options['autoint_dropout']
        has_residual = extra_options['autoint_has_residual']

        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        if use_project:
            feature_vectors = project(feature_vectors, project_size)

        check_feature_dims(feature_vectors)
        x = tf.stack(feature_vectors, axis=1)  # [B, N, D]
        y = _autoint(x,
                     num_blocks=num_blocks,
                     num_units=size_per_head*num_heads,
                     num_heads=num_heads,
                     dropout=dropout,
                     is_training=is_training,
                     has_residual=has_residual)

        tf.logging.info("autoint output = {}".format(y))
        with tf.variable_scope('logits') as logits_scope:
            logits = fc(y, units, name=logits_scope)
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _autoint(x,
             num_blocks,
             num_units,
             num_heads,
             dropout,
             is_training,
             has_residual):
    """AutoInt
      x: Tensor of shape [B, N, D]
      num_blocks: num of multihead attention layers
      num_units: num of hidden units, must be divisible by num_heads
      num_heads: num of attention heads
      dropout: dropout rate
      is_training: Is training mode
      has_residual: Use residual connection if True

    Return:
      Tensor of shape [B, N*num_units]
    """

    with tf.variable_scope('autoint'):
        y = x
        for block_id in range(num_blocks):
            y = _multihead_attention(queries=y,
                                     keys=y,
                                     values=y,
                                     num_units=num_units,
                                     num_heads=num_heads,
                                     dropout=dropout,
                                     is_training=is_training,
                                     has_residual=has_residual,
                                     name='block_{}'.format(block_id))
        y = tf.layers.flatten(y)  # [B, N*num_units]
        return y


def _layer_normalize(inputs, epsilon=1e-8, name='layer_norm'):
    '''
    Applies layer normalization
    Args:
        inputs: A tensor with 2 or more dimensions
        epsilon: A floating number to prevent Zero Division
    Returns:
        A tensor with the same shape and data dtype
    '''
    with tf.variable_scope(name):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

        return outputs


def _multihead_attention(queries,
                         keys,
                         values,
                         num_units=None,
                         num_heads=1,
                         dropout=0.0,
                         is_training=True,
                         has_residual=True,
                         name='multihead'):
    """
     queries: Tensor of shape [B, N, D]
     keys: Tensor of shape [B, N, D]
     values: Tensor of shape [B, N, D]
     num_units: Must be divisible by num_heads. If None, set to D.

    Return:
     Tensor of shape [B, N, num_units]
    """

    with tf.variable_scope(name):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        assert num_units % num_heads == 0, 'num_units must be divisible by num_heads'

        with tf.variable_scope('linear'):
            # Linear projections
            # [B, N, num_units]
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
            V = tf.layers.dense(values, num_units, activation=tf.nn.relu)
            if has_residual:
                V_res = tf.layers.dense(values, num_units, activation=tf.nn.relu)

        with tf.variable_scope('self_attention'):
            # Split and concat
            # [B*num_heads, N, num_units/num_heads]
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

            # Multiplication
            # [B*num_heads, N, N]
            weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

            # Scale
            # [B*num_heads, N, N]
            weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)

            # Activation
            # [B*num_heads, N, N]
            weights = tf.nn.softmax(weights)

            # Dropouts
            weights = tf.layers.dropout(weights,
                                        rate=dropout,
                                        training=is_training)

            # Weighted sum
            # [B*num_heads, N, num_units/num_heads]
            outputs = tf.matmul(weights, V_)

            # Restore shape
            # [B, N, num_units]
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

            # Residual connection
            if has_residual:
                outputs += V_res

            outputs = tf.nn.relu(outputs)

        # Layer Normalize
        outputs = _layer_normalize(outputs)

        return outputs  # [B, N, num_units]
