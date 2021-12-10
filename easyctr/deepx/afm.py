#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.common import check_arg
from common.common import add_hidden_layer_summary
from common.common import pairwise_dot
from common.common import get_feature_vectors
from common.common import project
from common.common import check_feature_dims


"""
Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks

https://github.com/hexiangnan/attentional_factorization_machine
"""


def _check_afm_args(args):
    check_arg(args, 'afm_use_shared_embedding', bool)
    check_arg(args, 'afm_use_project', bool)
    check_arg(args, 'afm_project_size', int)
    check_arg(args, 'afm_hidden_unit', int)


def get_afm_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('afm'):
        _check_afm_args(extra_options)
        use_shared_embedding = extra_options['afm_use_shared_embedding']
        use_project = extra_options['afm_use_project']
        project_size = extra_options['afm_project_size']
        hidden_unit = extra_options['afm_hidden_unit']

        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        if use_project:
            feature_vectors = project(feature_vectors, project_size)

        y = _afm(feature_vectors, hidden_unit, reduce_sum=True)  # [B, 1]
        with tf.variable_scope('logits') as logits_scope:
            logits = y
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _afm_attention_network(interactions, hidden_unit):
    """AFM attention network
      interactions: [B, N*(N-1)/2, D], N means number of features
      hidden_unit: attention network num of hidden units

    Return:
      attention weights, shape of [B, N*(N-1)/2, 1]
    """

    with tf.variable_scope('afm_att_net'):
        # [B, N*(N-1)/2, hidden_unit]
        y = tf.layers.dense(interactions,
                            units=hidden_unit,
                            activation=tf.nn.relu)

        h = tf.get_variable('afm_att_h', [hidden_unit])
        y = y * h  # [B, N*(N-1)/2, hidden_unit]
        y = tf.reduce_sum(y, axis=-1)  # [B, N*(N-1)/2]
        y = tf.nn.softmax(y, axis=-1)  # [B, N*(N-1)/2]
        y = tf.expand_dims(y, axis=2)  # [B, N*(N-1)/2, 1]

        return y  # [B, N*(N-1)/2, 1]


def _afm(feature_vectors, hidden_unit, reduce_sum=True):
    """AFM
      feature_vectors: List of shape [B, D] tensors, size N
      hidden_unit: see paper, attention layer units

    Return:
     Tensor of shape [B, 1] if reduce_sum is True, or shape of [B, num_iters]
    """
    with tf.variable_scope('afm'):
        check_feature_dims(feature_vectors)

        interactions = pairwise_dot(feature_vectors)  # [B, num_inters, D]
        tf.logging.info("interactions = {}".format(interactions))

        # [B, num_inters, 1]
        att_weights = _afm_attention_network(interactions, hidden_unit)
        tf.summary.histogram('afm_att_weights', att_weights)

        # attention pooling
        y = att_weights * interactions  # [B, num_inters, D]
        y = tf.reduce_sum(y, axis=-1)  # [B, num_inters]

        if reduce_sum:
            y = tf.reduce_sum(y, axis=1, keepdims=True)  # [B, 1]

        return y
