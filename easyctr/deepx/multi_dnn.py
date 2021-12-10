#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from common.common import add_hidden_layers
from common.common import add_hidden_layer_summary
from common.common import fc
from common.common import check_arg
from common.common import get_activation_fn
from common.common import get_feature_vectors
from common.common import project


def _check_multi_dnn_args(args):
    check_arg(args, 'multi_dnn_use_shared_embedding', bool)
    check_arg(args, 'multi_dnn_use_project', bool)
    check_arg(args, 'multi_dnn_project_size', int)
    check_arg(args, 'multi_dnn_shared_hidden_units', list)
    check_arg(args, 'multi_dnn_tower_hidden_units', list)
    check_arg(args, 'multi_dnn_tower_use_shared_embedding', bool)
    check_arg(args, 'multi_dnn_activation_fn', str)
    check_arg(args, 'multi_dnn_dropout', float)
    check_arg(args, 'multi_dnn_batch_norm', bool)
    check_arg(args, 'multi_dnn_layer_norm', bool)
    check_arg(args, 'multi_dnn_use_resnet', bool)
    check_arg(args, 'multi_dnn_use_densenet', bool)
    check_arg(args, 'multi_dnn_l2_regularizer', float)

    check_arg(args, 'leaky_relu_alpha', float)
    check_arg(args, 'swish_beta', float)


def get_multi_dnn_logits(
        features,
        feature_columns,
        selector_feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('multi_dnn'):
        _check_multi_dnn_args(extra_options)
        use_shared_embedding = extra_options['multi_dnn_use_shared_embedding']
        use_project = extra_options['multi_dnn_use_project']
        project_size = extra_options['multi_dnn_project_size']
        shared_hidden_units = extra_options['multi_dnn_shared_hidden_units']
        tower_hidden_units = extra_options['multi_dnn_tower_hidden_units']
        tower_use_shared_embedding = extra_options['multi_dnn_tower_use_shared_embedding']
        activation_fn = extra_options['multi_dnn_activation_fn']
        dropout = extra_options['multi_dnn_dropout']
        batch_norm = extra_options['multi_dnn_batch_norm']
        layer_norm = extra_options['multi_dnn_layer_norm']
        use_resnet = extra_options['multi_dnn_use_resnet']
        use_densenet = extra_options['multi_dnn_use_densenet']
        l2_regularizer = extra_options['multi_dnn_l2_regularizer']

        leaky_relu_alpha = extra_options['leaky_relu_alpha']
        swish_beta = extra_options['swish_beta']
        activation_fn = get_activation_fn(activation_fn=activation_fn,
                                          leaky_relu_alpha=leaky_relu_alpha,
                                          swish_beta=swish_beta)

        assert len(selector_feature_columns) == 1, \
            "selector_feature_columns size must be 1"
        selector_vector = tf.feature_column.input_layer(
            features, selector_feature_columns[0])   # [B, selector_size]
        selector_size = selector_vector.shape[1].value

        feature_vectors_list = None
        feature_vectors = None
        if not use_shared_embedding:
            if tower_use_shared_embedding:
                feature_vectors = get_feature_vectors(features, feature_columns)
            else:
                assert len(shared_hidden_units) == 0, \
                    '"shared_hidden_units" must be empty when "tower_use_shared_embedding" is False'
                feature_vectors_list = []
                for idx in range(selector_size):
                    v = get_feature_vectors(features, feature_columns, scope='feature_vectors_{}'.format(idx))
                    feature_vectors_list.append(v)
        else:
            assert tower_use_shared_embedding, \
                '"tower_use_shared_embedding" must be True when "use_shared_embedding" is True'
            feature_vectors = shared_feature_vectors

        if use_project:
            if feature_vectors is not None:
                feature_vectors = project(feature_vectors, project_size)
            if feature_vectors_list is not None:
                for idx in range(len(feature_vectors_list)):
                    feature_vectors_list[idx] = project(
                        feature_vectors_list[idx], project_size)

        y = _multi_dnn(feature_vectors,
                       feature_vectors_list,
                       selector_vector,
                       shared_hidden_units,
                       tower_hidden_units,
                       activation_fn,
                       dropout,
                       is_training,
                       batch_norm,
                       layer_norm,
                       use_resnet,
                       use_densenet,
                       l2_regularizer)

        with tf.variable_scope('logits') as logits_scope:
            logits = fc(y, units=units, name=logits_scope)
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _multi_dnn(feature_vectors,
               feature_vectors_list,
               selector_vector,
               shared_hidden_units,
               tower_hidden_units,
               activation_fn,
               dropout,
               is_training,
               batch_norm,
               layer_norm,
               use_resnet,
               use_densenet,
               l2_regularizer):
    """multi-dnn, 自研多塔模型，模型有多个塔，但每个样本只会过其中一个塔，
    根据selector_vector向量进行选择（selector_vector为one-hot向量），示例适用场
    景：回归任务，对于回归值小于N的样本过一个塔，回归值大于等于N的样本过另一个塔。

        feature_vectors 和 feature_vectors_list 只有一个非 None;
        feature_vectors_list 非 None 表示没有 shared_hidden_units，且 tower 使用
        单独的 embedding
    """
    selector_size = selector_vector.shape[1].value
    tf.logging.info("multi-dnn selector size = {}".format(selector_size))
    with tf.variable_scope('multi_dnn'):
        if len(shared_hidden_units) > 0:
            assert feature_vectors is not None
            x = tf.concat(feature_vectors, axis=1)   # [B, T]
            shared_bottom = add_hidden_layers(
                x,
                hidden_units=shared_hidden_units,
                activation_fn=activation_fn,
                dropout=dropout,
                is_training=is_training,
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                use_resnet=use_resnet,
                use_densenet=use_densenet,
                l2_regularizer=l2_regularizer,
                scope='shared_hidden_layers')
        else:
            if feature_vectors is not None :
                x = tf.concat(feature_vectors, axis=1)   # [B, T]
                shared_bottom = x
            else:
                assert feature_vectors_list is not None
                shared_bottom_list = []
                for idx in range(selector_size):
                    x = tf.concat(feature_vectors_list[idx], axis=1)   # [B, T]
                    shared_bottom_list.append(x)

        # multi task
        tasks = []
        for tower_id in range(selector_size):
            if feature_vectors_list is not None:
                shared_bottom = shared_bottom_list[tower_id]
            y = add_hidden_layers(shared_bottom,
                                  hidden_units=tower_hidden_units,
                                  activation_fn=activation_fn,
                                  dropout=dropout,
                                  is_training=is_training,
                                  batch_norm=batch_norm,
                                  layer_norm=layer_norm,
                                  use_resnet=use_resnet,
                                  use_densenet=use_densenet,
                                  l2_regularizer=l2_regularizer,
                                  scope='tower_hidden_layers_{}'.format(tower_id))
            tasks.append(y)  # list of [B, H], size = selector_size
        tasks = tf.stack(tasks, axis=1)  # [B, selector_size, H]
        tf.logging.info("multi_dnn tasks = {}".format(tasks))

        # select one tower
        selector_vector = tf.expand_dims(selector_vector, axis=-1)  # [B, selector_vector, 1]
        y = tasks * selector_vector  # [B, selector_size, H]
        y = tf.reduce_sum(y, axis=1)  # [B, H]

        return y
