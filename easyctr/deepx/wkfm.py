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
Product-based Neural Networks for User Response Prediction over Multi-field Categorical Data

The name 'w*' means adding field weight (singe variable) for every pair of interactions.
"""


def _check_wkfm_args(args):
    check_arg(args, 'wkfm_use_shared_embedding', bool)
    check_arg(args, 'wkfm_use_project', bool)
    check_arg(args, 'wkfm_project_size', int)
    check_arg(args, 'wkfm_use_selector', bool)


def get_wkfm_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('wkfm'):
        _check_wkfm_args(extra_options)
        use_shared_embedding = extra_options['wkfm_use_shared_embedding']
        use_project = extra_options['wkfm_use_project']
        project_size = extra_options['wkfm_project_size']
        use_selector = extra_options['wkfm_use_selector']

        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns,
                                                  scope='wkfm_feature_vectors')
        else:
            feature_vectors = shared_feature_vectors

        if use_project:
            feature_vectors = project(feature_vectors, project_size)

        if use_selector:
            assert len(selector_feature_columns) == 1, \
                "selector_feature_columns size must be 1"
            selector_vector = tf.feature_column.input_layer(
                features, selector_feature_columns[0])
            y = _selector_wkfm_v2(feature_vectors, selector_vector, reduce_sum=True)  # [B, 1]
        else:
            y = _wkfm(feature_vectors, reduce_sum=True)  # [B, 1]

        with tf.variable_scope('logits') as logits_scope:
            # fc just for adding a bias
            logits = fc(y, units=units, name=logits_scope)
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _wkfm(feature_vectors, reduce_sum=True):
    """Kernel FM
      feature_vectors: List of shape [B, ?] tensors, size N

    Half-Optimized implementation

    Return:
      Tensor of shape [B, 1] if reduce_sum is True, or shape [B, T], T is the sum
      dimentions of all features.
    """

    with tf.variable_scope('wkfm'):
        outputs = []
        x = tf.concat(feature_vectors, axis=1)   # [B, T]
        T = x.shape[1].value
        N = len(feature_vectors)
        indices = []
        for j in range(N):
            vj = feature_vectors[j]
            dj = vj.shape[1].value
            indices.extend([j] * dj)

        for i in range(N):
            vi = feature_vectors[i]
            name = 'wkfm_{}'.format(i)
            di = vi.shape[1].value
            U = tf.get_variable(name, [T, di])  # [T, di]

            # 创建两两交叉特征的权重, 与 KFM 的主要区别就是这个权重
            name = 'wkfm_weightes_{}'.format(i)
            wkfm_weights = tf.get_variable(name, [N], initializer=tf.ones_initializer)
            weights = tf.gather(wkfm_weights, indices)

            y = tf.matmul(weights * x, U)   # [B, di]
            outputs.append(y)
        y = tf.concat(outputs, axis=1)   # [B, T]
        y = x * y  # [B, T]

        if reduce_sum:
            y = tf.reduce_sum(y, axis=1, keepdims=True)  # [B, 1]

        return y


def _selector_wkfm(feature_vectors, selector_vector, reduce_sum=True):
    """Kernel FM
      feature_vectors: List of shape [B, ?] tensors, size N
      selector_vector: shape of [B, selector_size], 样本集选择器，不同的样本集
      使用不同的U矩阵和W参数

    Half-Optimized implementation

    Return:
      Tensor of shape [B, 1] if reduce_sum is True, or shape [B, T], T is the sum
      dimentions of all features.
    """

    selector_size = selector_vector.shape[1].value
    tf.logging.info("wkfm selector size = {}".format(selector_size))

    with tf.variable_scope('selector_wkfm'):
        outputs = []
        x = tf.concat(feature_vectors, axis=1)   # [B, T]
        T = x.shape[1].value
        N = len(feature_vectors)
        indices = []
        for j in range(N):
            vj = feature_vectors[j]
            dj = vj.shape[1].value
            indices.extend([j] * dj)   # [T]

        for i in range(N):
            vi = feature_vectors[i]
            name = 'wkfm_{}'.format(i)
            di = vi.shape[1].value
            # 根据 selector 选出对应的 U 矩阵
            A = tf.get_variable(name, [selector_size, T, di])
            U = tf.tensordot(selector_vector, A, axes=1)  # [B, T, di]

            # 创建两两交叉特征的权重, 与 KFM 的主要区别就是这个权重
            # 权重同样需要根据 selector 选择
            name = 'wkfm_weightes_{}'.format(i)
            a_wkfm_weights = tf.get_variable(name, [selector_size, N], initializer=tf.ones_initializer)
            wkfm_weights = tf.matmul(selector_vector, a_wkfm_weights)  # [B, N]

            weights = tf.gather(wkfm_weights, indices, axis=1)   # [B, T]
            y = tf.matmul(tf.expand_dims(weights * x, axis=1), U)   # [B, 1, T]*[B, T, di] => [B, 1, di]
            y = tf.squeeze(y, axis=1)  # [B, di]
            outputs.append(y)
        y = tf.concat(outputs, axis=1)   # [B, T]
        y = x * y  # [B, T]

        if reduce_sum:
            y = tf.reduce_sum(y, axis=1, keepdims=True)  # [B, 1]

        return y


def _selector_wkfm_v2(feature_vectors, selector_vector, reduce_sum=True):
    """没有做计算优化的版本"""

    selector_size = selector_vector.shape[1].value
    tf.logging.info("wkfm selector size = {}".format(selector_size))

    with tf.variable_scope('selector_wkfm_v2'):
        interactions = []
        n = len(feature_vectors)
        for i in range(n):
            for j in range(n):
                di = feature_vectors[i].shape[1].value
                dj = feature_vectors[j].shape[1].value
                name = 'pairwise_kernel_dot_{}_{}'.format(i, j)
                A = tf.get_variable(name, [selector_size, di, dj]) # [S, di, dj]
                U = tf.tensordot(selector_vector, A, axes=1)  # [B, di, dj]
                y = tf.matmul(tf.expand_dims(feature_vectors[i], axis=1), U)  # [B, 1, di]*[B, di, dj] => [B, 1, dj]
                y = tf.squeeze(y, axis=1)  # [B, dj]
                y = y * feature_vectors[j]  # [B, dj]
                y = tf.reduce_sum(y, axis=-1, keepdims=True)  # [B, 1]
                name = 'wkfm_weightes_{}_{}'.format(i, j)
                a_wkfm_weights = tf.get_variable(name, [selector_size, 1], initializer=tf.ones_initializer) # [S, 1]
                wkfm_weights = tf.matmul(selector_vector, a_wkfm_weights)  # [B, S]*[S, 1] => [B, 1]
                y = y * wkfm_weights  # [B, 1]
                interactions.append(y)
        if reduce_sum:
            y = sum(interactions)  # [B, 1]
        else:
            y = tf.concat(interactions, axis=1)  # [B, N*N]

        return y
