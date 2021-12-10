#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.common import get_feature_vectors
from common.common import check_feature_dims
from common.common import check_arg
from common.common import project


"""
Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction
"""


def get_fgcnn_feature_vectors(features,
                              feature_columns,
                              feature_vectors,
                              options,
                              name='fgcnn'):
    """
    """

    with tf.variable_scope(name):
        _check_fgcnn_args(options)
        use_shared_embedding = options['fgcnn_use_shared_embedding']
        use_project = options['fgcnn_use_project']
        project_dim = options['fgcnn_project_dim']
        filter_nums = options['fgcnn_filter_nums']
        kernel_sizes = options['fgcnn_kernel_sizes']
        pooling_sizes = options['fgcnn_pooling_sizes']
        new_map_sizes = options['fgcnn_new_map_sizes']

        x = feature_vectors
        if not use_shared_embedding:
            x = get_feature_vectors(features,
                                    feature_columns,
                                    name + '_feature_vectors')
        if use_project:
            x = project(feature_vectors, project_dim)
        new_feature_vectors = _fgcnn(x,
                                     filter_nums=filter_nums,
                                     kernel_sizes=kernel_sizes,
                                     pooling_sizes=pooling_sizes,
                                     new_map_sizes=new_map_sizes)

        return new_feature_vectors


def _check_fgcnn_args(args):
    check_arg(args, 'fgcnn_use_shared_embedding', bool)
    check_arg(args, 'fgcnn_use_project', bool)
    check_arg(args, 'fgcnn_project_dim', int)
    check_arg(args, 'fgcnn_filter_nums', list)
    check_arg(args, 'fgcnn_kernel_sizes', list)
    check_arg(args, 'fgcnn_pooling_sizes', list)
    check_arg(args, 'fgcnn_new_map_sizes', list)


def _fgcnn(feature_vectors,
           filter_nums,
           kernel_sizes,
           pooling_sizes,
           new_map_sizes,
           name='fgcnn'):
    """FGCNN: Generate new features
    feature_vectors: List of N shape [B, D] tensors

    Return:
      New features: List of Tensors of shape [B, D]
    """

    with tf.variable_scope(name):
        check_feature_dims(feature_vectors)
        assert len(filter_nums) == len(kernel_sizes) == len(pooling_sizes) == len(new_map_sizes)

        x = tf.stack(feature_vectors, axis=1)   # [B, N, D]
        D = x.shape[2].value

        y = x
        y = tf.expand_dims(y, axis=-1)   # [B, N, D, 1]
        new_features = []
        for idx in range(len(filter_nums)):
            filters = filter_nums[idx]
            kernel_size = kernel_sizes[idx]
            pooling_size = pooling_sizes[idx]
            new_map = new_map_sizes[idx]
            name = 'conv{}'.format(idx)
            y = tf.layers.conv2d(y,
                                 filters=filters,
                                 kernel_size=(kernel_size, 1),
                                 strides=(1, 1),
                                 padding='SAME',
                                 activation=tf.nn.tanh,
                                 name=name)
            tf.logging.info('{} = {}'.format(name, y))
            name = 'pooling{}'.format(idx)
            y = tf.layers.max_pooling2d(y,
                                        pool_size=(pooling_size, 1),
                                        strides=(1, 1),
                                        name=name)
            tf.logging.info('{} = {}'.format(name, y))
            flatten = tf.layers.flatten(y)
            name = 'fc{}'.format(idx)
            num_feature = new_map * y.shape[1].value
            new_feature = tf.layers.dense(flatten,
                                          units=num_feature*D,
                                          activation=tf.nn.tanh,
                                          name=name)
            new_feature = tf.reshape(new_feature, [-1, num_feature, D])
            tf.logging.info('{} = {}'.format(name, new_feature))
            new_feature = tf.unstack(new_feature, axis=1)
            new_features.extend(new_feature)

    return new_features
