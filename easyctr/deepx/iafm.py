#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.common import check_arg
from common.common import add_hidden_layer_summary
from common.common import interaction_aware_pairwise_dot
from common.common import get_feature_vectors
from common.common import project
from common.common import check_feature_dims
from easyctr.deepx.afm import _afm_attention_network


"""
Interaction-aware Factorization Machines for Recommender Systems
"""


def _check_iafm_args(args):
    check_arg(args, 'iafm_use_shared_embedding', bool)
    check_arg(args, 'iafm_use_project', bool)
    check_arg(args, 'iafm_project_size', int)
    check_arg(args, 'iafm_hidden_unit', int)
    check_arg(args, 'iafm_field_dim', int)


def get_iafm_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('iafm'):
        _check_iafm_args(extra_options)
        use_shared_embedding = extra_options['iafm_use_shared_embedding']
        use_project = extra_options['iafm_use_project']
        project_size = extra_options['iafm_project_size']
        hidden_unit = extra_options['iafm_hidden_unit']
        field_dim = extra_options['iafm_field_dim']

        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        if use_project:
            feature_vectors = project(feature_vectors, project_size)

        y = _iafm(feature_vectors, hidden_unit, field_dim, reduce_sum=True)

        with tf.variable_scope('logits') as logits_scope:
            logits = y
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _iafm(feature_vectors, hidden_unit, field_dim, reduce_sum=True):
    """IAFM
      feature_vectors: List of shape [B, D] tensors, size N
      hidden_unit: see paper, attention layer units
      field_dim: see paper formula (6), hyper-param K_F

    Return:
     Tensor of shape [B, 1] if reduce_sum is True, or shape of [B, num_iters]
    """

    with tf.variable_scope('iafm'):
        check_feature_dims(feature_vectors)

        # [B, num_inters, D]
        interactions = interaction_aware_pairwise_dot(feature_vectors, field_dim)
        tf.logging.info("interactions = {}".format(interactions))

        # [B, num_inters, 1]
        att_weights = _afm_attention_network(interactions, hidden_unit)
        tf.summary.histogram('iafm_att_weights', att_weights)

        # attention pooling
        y = att_weights * interactions  # [B, num_inters, D]
        y = tf.reduce_sum(y, axis=-1)  # [B, num_inters]

        if reduce_sum:
            y = tf.reduce_sum(y, axis=1, keepdims=True)  # [B, 1]

        return y
