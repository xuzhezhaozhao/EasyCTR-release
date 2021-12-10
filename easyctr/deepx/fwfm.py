#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from common.common import pairwise_dot
from common.common import fc
from common.common import add_hidden_layer_summary
from common.common import check_arg
from common.common import get_feature_vectors
from common.common import project
from common.common import check_feature_dims


"""Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising

Only implement field-weighted FM, ignore the proposed method about linear terms
in the paper.
"""


def _check_fwfm_args(args):
    check_arg(args, 'fwfm_use_shared_embedding', bool)
    check_arg(args, 'fwfm_use_project', bool)
    check_arg(args, 'fwfm_project_size', int)


def get_fwfm_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('fwfm'):
        _check_fwfm_args(extra_options)
        use_shared_embedding = extra_options['fwfm_use_shared_embedding']
        use_project = extra_options['fwfm_use_project']
        project_size = extra_options['fwfm_project_size']

        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        if use_project:
            feature_vectors = project(feature_vectors, project_size)

        y = _fwfm(feature_vectors, units)  # [B, 1]
        with tf.variable_scope('logits') as logits_scope:
            logits = y
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _fwfm(feature_vectors, units):
    """FwFM
      feature_vectors: List of shape [B, D] tensors, size N
      units: logits units

    Return:
      Tensor of shape [B, 1]
    """
    with tf.variable_scope('fwfm'):
        check_feature_dims(feature_vectors)

        y = pairwise_dot(feature_vectors)  # [B, N*(N-1)/2, D]
        y = tf.reduce_sum(y, axis=-1)  # [B, N*(N-1)/2]
        y = fc(y, units)  # [B, 1]

        return y
