#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from common.common import fc
from common.common import add_hidden_layer_summary
from common.common import check_arg
from common.common import check_feature_dims
from common.common import get_feature_vectors
from common.common import project


"""
Compressed Interaction Network used in xDeepFM.

https://github.com/Leavingseason/xDeepFM
https://github.com/shenweichen/DeepCTR
"""


def _check_cin_args(args):
    check_arg(args, 'cin_use_shared_embedding', bool)
    check_arg(args, 'cin_use_project', bool)
    check_arg(args, 'cin_project_size', int)
    check_arg(args, 'cin_hidden_feature_maps', list)
    check_arg(args, 'cin_split_half', bool)


def get_cin_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('cin'):
        _check_cin_args(extra_options)
        use_shared_embedding = extra_options['cin_use_shared_embedding']
        use_project = extra_options['cin_use_project']
        project_size = extra_options['cin_project_size']
        hidden_feature_maps = extra_options['cin_hidden_feature_maps']
        split_half = extra_options['cin_split_half']

        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        if use_project:
            feature_vectors = project(feature_vectors, project_size)

        check_feature_dims(feature_vectors)
        x = tf.stack(feature_vectors, axis=1)  # [B, N, D]
        y = _cin_layer(x, hidden_feature_maps, split_half, reduce_sum=False)  # [B, F]

        with tf.variable_scope('logits') as logits_scope:
            logits = fc(y, units=units, name=logits_scope)
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _cin_layer(x, hidden_feature_maps, split_half=True, reduce_sum=True):
    """CIN layer
      x: (B, N, D)

    Return:
      2D tensor with shape [B, 1] if reduce_sum is True, or shape [B, F],
      F means feature map num
    """

    with tf.variable_scope('cin'):

        dim = x.shape[-1].value
        field_size = x.shape[1].value
        field_nums = [field_size]
        filters_shape = []
        bias_shape = []

        for num in hidden_feature_maps:
            filters_shape.append([1, field_nums[-1] * field_nums[0], num])
            bias_shape.append([num])

            if split_half:
                assert num % 2 == 0, \
                    "num_feature_map must be even number if split_half is True"
                field_nums.append(num // 2)
            else:
                field_nums.append(num)

        hidden_nn_layers = [x]
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        for idx, num_feature_map in enumerate(hidden_feature_maps):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(
                dot_result_m, shape=[dim, -1, field_nums[0]*field_nums[idx]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            filters_name = 'filters_{}'.format(idx)
            bias_name = 'bias_{}'.format(idx)
            filters = tf.get_variable(filters_name, filters_shape[idx])
            bias = tf.get_variable(bias_name,
                                   bias_shape[idx],
                                   initializer=tf.initializers.zeros)
            curr_out = tf.nn.conv1d(dot_result,
                                    filters=filters,
                                    stride=1,
                                    padding='VALID')
            curr_out = tf.nn.bias_add(curr_out, bias)
            curr_out = tf.nn.relu(curr_out)
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if split_half:
                if idx != len(hidden_feature_maps) - 1:
                    next_hidden, direct_connect = tf.split(
                        curr_out, 2 * [num_feature_map // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        y = tf.concat(final_result, axis=1)
        y = tf.reduce_sum(y, axis=-1, keepdims=False)  # [B, F]

        if reduce_sum:
            y = tf.reduce_sum(y, axis=-1, keepdims=True)  # [B, 1]

        return y
