#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.common import check_feature_dims
from common.common import check_arg
from common.common import pairwise_dot
from common.common import pairwise_kernel_dot_aligned
from common.common import add_hidden_layers
from common.common import add_hidden_layer_summary
from common.common import fc
from common.common import get_activation_fn
from common.common import get_feature_vectors
from common.common import project


"""
FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
"""


def _check_fibinet_args(args):
    check_arg(args, 'fibinet_use_shared_embedding', bool)
    check_arg(args, 'fibinet_use_project', bool)
    check_arg(args, 'fibinet_project_size', int)
    check_arg(args, 'fibinet_hidden_units', list)
    check_arg(args, 'fibinet_activation_fn', str)
    check_arg(args, 'fibinet_dropout', float)
    check_arg(args, 'fibinet_batch_norm', bool)
    check_arg(args, 'fibinet_layer_norm', bool)
    check_arg(args, 'fibinet_use_resnet', bool)
    check_arg(args, 'fibinet_use_densenet', bool)

    check_arg(args, 'fibinet_use_se', bool)
    check_arg(args, 'fibinet_use_deep', bool)
    check_arg(args, 'fibinet_interaction_type', str)
    check_arg(args, 'fibinet_se_interaction_type', str)
    check_arg(args, 'fibinet_se_use_shared_embedding', bool)


def get_fibinet_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):
    with tf.variable_scope('fibinet'):
        _check_fibinet_args(extra_options)
        use_shared_embedding = extra_options['fibinet_use_shared_embedding']
        use_project = extra_options['fibinet_use_project']
        project_size = extra_options['fibinet_project_size']
        hidden_units = extra_options['fibinet_hidden_units']
        activation_fn = extra_options['fibinet_activation_fn']
        dropout = extra_options['fibinet_dropout']
        batch_norm = extra_options['fibinet_batch_norm']
        layer_norm = extra_options['fibinet_layer_norm']
        use_resnet = extra_options['fibinet_use_resnet']
        use_densenet = extra_options['fibinet_use_densenet']
        use_se = extra_options['fibinet_use_se']
        use_deep = extra_options['fibinet_use_deep']
        interaction_type = extra_options['fibinet_interaction_type']
        se_interaction_type = extra_options['fibinet_se_interaction_type']
        se_use_shared_embedding = extra_options['fibinet_se_use_shared_embedding']
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

        y = shallow_fibinet(features=features,
                            feature_columns=feature_columns,
                            shared_feature_vectors=feature_vectors,
                            se_use_shared_embedding=se_use_shared_embedding,
                            use_project=use_project,
                            project_size=project_size,
                            interaction_type=interaction_type,
                            se_interaction_type=se_interaction_type,
                            use_se=use_se)  # [B, -1]
        if use_deep:
            y = add_hidden_layers(y,
                                  hidden_units=hidden_units,
                                  activation_fn=activation_fn,
                                  dropout=dropout,
                                  is_training=is_training,
                                  batch_norm=batch_norm,
                                  layer_norm=layer_norm,
                                  use_resnet=use_resnet,
                                  use_densenet=use_densenet,
                                  scope='hidden_layers')
            with tf.variable_scope('logits') as logits_scope:
                logits = fc(y, units, name=logits_scope)
                add_hidden_layer_summary(logits, logits_scope.name)
        else:
            assert units == 1, "shallow_fibinet's units must be 1"
            with tf.variable_scope('logits') as logits_scope:
                logits = tf.reduce_sum(y, axis=-1, keepdims=True)  # [B, 1]
                add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def selayer(x, name='senet'):
    """Squeeze-Excitation network
     x: Tensor of shape [B, N, D], stacked featrue vectors

    Return:
     Tensor of shape [B, N, D]
    """

    with tf.variable_scope(name):
        N = x.shape[1].value

        # Squeeze
        y = tf.reduce_mean(x, axis=2)  # [B, N]

        # Excitation
        y = tf.layers.dense(y,
                            units=N,
                            activation=tf.nn.relu)
        y = tf.layers.dense(y,
                            units=N,
                            activation=None)  # [B, N]

        # Re-Weight
        y = tf.expand_dims(y, axis=2)  # [B, N, 1]
        y = y * x  # [B, N, D]

        return y


def bilinear(feature_vectors, interaction_type, name='bilinear'):
    """Bi-Linear Interaction
    For 'bilinear' interaction type, we choose Field-Interaction Type in the
    origin paper.

     feature_vectors: List of shape [B, D] tensors, size N

    Return:
      Tensor of shape [B, -1]
    """

    with tf.variable_scope(name):
        if interaction_type == 'inner':
            y = pairwise_dot(feature_vectors, name='inner')  # [B, N*(N-1)/2, D]
            y = tf.reduce_sum(y, axis=-1)   # [B, N*(N-1)/2]
        elif interaction_type == 'hadamard':
            # [B, N*(N-1)/2, D]
            y = pairwise_dot(feature_vectors, name='hadamard')
            y = tf.layers.flatten(y)   # [B, N*(N-1)/2*D]
        elif interaction_type == 'bilinear':
            # [B, N*(N-1)/2, D]
            y = pairwise_kernel_dot_aligned(feature_vectors, name='bilinear')
            y = tf.layers.flatten(y)  # [B, N(N-1)/2*D]
        else:
            raise ValueError("Unkonw interaction_type '{}'".format(interaction_type))

        return y


def shallow_fibinet(features,
                    feature_columns,
                    shared_feature_vectors,
                    se_use_shared_embedding,
                    use_project,
                    project_size,
                    interaction_type,
                    se_interaction_type,
                    use_se,
                    name='shallow_fibinet'):
    """Shallow part of FiBiNET
     feature_vectors: list of 2-D tensors of shape [B, D], size N.

    Return:
      Tensor of shape [B, -1]
    """

    with tf.variable_scope(name):
        check_feature_dims(shared_feature_vectors)

        y = bilinear(shared_feature_vectors, interaction_type)  # [B, -1]
        if use_se:
            if se_use_shared_embedding:
                se_feature_vectors = shared_feature_vectors
            else:
                se_feature_vectors = get_feature_vectors(features, feature_columns)
                if use_project:
                    se_feature_vectors = project(se_feature_vectors, project_size)
                check_feature_dims(se_feature_vectors)

            x = tf.stack(se_feature_vectors, axis=1)  # [B, N, D]
            se_x = selayer(x)  # [B, N, D]
            new_se_feature_vectors = tf.unstack(se_x, axis=1)  # N tensors of shape [B, D]
            se_y = bilinear(new_se_feature_vectors,
                            se_interaction_type,
                            name='se_bilinear')   # [B, -1]
            y = tf.concat([y, se_y], axis=1)  # [B, -1]

        return y
