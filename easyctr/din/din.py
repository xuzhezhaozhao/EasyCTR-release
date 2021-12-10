#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from easyctr.din import attention

from common.common import add_hidden_layer_summary
from common.common import add_hidden_layers
from common.common import mask_padding_embedding_lookup
from common.common import create_embedding
from common.common import fc

from common.estimator import head as head_lib
from common.estimator import optimizers

# The default learning rate of 0.05 is a historical artifact of the initial
# implementation, but seems a reasonable choice.
_LEARNING_RATE = 0.05


def _attention(features, feature_list, query, dimension,
               attention_type, attention_args, shared_embedding, is_training):
    """以数组形式返回 attention output 和 query embedding tensor """

    outputs = []
    query_name = query[0]
    query_num_buckets = query[1]
    # embedding num buckets +1 是为了处理 input id 以 -1 作为 padding 的情况，
    # 因为 tf.gather 不能处理 id 为 -1 的情况，而 0 又不是 padding id，所以取
    # num_buckets 作为 padding id, 于是创建 embedding 变量时需要额外 +1
    query_embeddings = create_embedding(
        query_name, query_num_buckets + 1, dimension)
    tf.logging.info("query_embeddings: {}".format(query_embeddings))
    query_inputs = features[query_name]
    query_inputs = tf.reshape(query_inputs, [-1])

    # -1 表示 padding id, gather 不能直接处理，需要将 -1 替换为 num_buckets
    condition = tf.math.not_equal(query_inputs, -1)
    mask = tf.ones_like(query_inputs) * query_num_buckets
    query_inputs = tf.where(condition, query_inputs, mask)
    # query_emb = tf.gather(query_embeddings, query_inputs)  # [B, H]
    # do padding mask
    query_emb = mask_padding_embedding_lookup(
        query_embeddings,
        dimension,
        query_inputs,
        query_num_buckets)

    query_emb.set_shape([None, dimension])
    tf.logging.info("query_emb: {}".format(query_emb))

    outputs.append(query_emb)

    for feature in feature_list:
        key_name = feature[0]
        key_num_buckets = feature[1]
        key_embeddings = None
        if shared_embedding:
            assert query_num_buckets == key_num_buckets
            key_embeddings = query_embeddings
        else:
            key_embeddings = create_embedding(
                key_name, key_num_buckets + 1, dimension)
        tf.logging.info("key_embeddings: {}".format(key_embeddings))
        key_inputs = features[key_name]
        condition = tf.math.not_equal(key_inputs, -1)
        key_length = tf.reduce_sum(tf.to_int32(condition), axis=-1)
        tf.logging.info("key_length: {}".format(key_length))
        mask = tf.ones_like(key_inputs) * key_num_buckets
        key_inputs = tf.where(condition, key_inputs, mask)
        key_emb = tf.gather(key_embeddings, key_inputs)  # [B, T, H]
        key_emb.set_shape([None, None, dimension])
        tf.logging.info("key_emb: {}".format(key_emb))

        if attention_type == 'din':
            output = attention.din_attention(query_emb, key_emb, key_length)
            output = tf.layers.batch_normalization(
                output, training=is_training)
            output = tf.layers.dense(output, dimension)
        elif attention_type == 'mlp':
            output = attention.mlp_attention(
                query_emb, key_emb, key_length,
                attention_args['attention_hidden_units'])
        else:
            raise ValueError("Unknown attention_type '{}'".format(attention_type))

        outputs.append(output)

    return outputs


def _attention_layer(features, attention_columns, is_training):
    outputs = []
    with tf.variable_scope('attention_layer'):
        for column in attention_columns:
            # column 类型是 models/transform.py 中的 feature_column_attention_columns
            output = _attention(
                features, column.attention_feature_list,
                column.attention_query,
                column.dimension,
                column.attention_type,
                column.attention_args,
                column.shared_embedding,
                is_training)
            outputs.extend(output)

    return outputs


class DINEstimator(tf.estimator.Estimator):
    """An estimator for alibaba din model.
    """

    def __init__(
        self,
        hidden_units,
        feature_columns,
        attention_columns,
        model_dir=None,
        n_classes=2,
        weight_column=None,
        label_vocabulary=None,
        optimizer='Adagrad',
        activation_fn=tf.nn.relu,
        dropout=None,
        input_layer_partitioner=None,
        config=None,
        warm_start_from=None,
        loss_reduction=tf.losses.Reduction.SUM,
        batch_norm=False,
        loss_fn=None
    ):

        def _model_fn(features, labels, mode, config):

            head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
                n_classes, weight_column, label_vocabulary, loss_reduction, loss_fn)

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            net = tf.feature_column.input_layer(features, feature_columns)
            inputs = _attention_layer(features, attention_columns, is_training)
            tf.logging.info('attention outputs = {}'.format(inputs))

            inputs.append(net)
            net = tf.concat(inputs, axis=-1)
            tf.logging.info("inputs: {}".format(net))

            if batch_norm:
                net = tf.layers.batch_normalization(
                    net, training=is_training)

            net = add_hidden_layers(net, hidden_units, activation_fn, dropout,
                                    is_training, batch_norm, 'DNN')
            with tf.variable_scope('logits') as logits_scope:
                logits = fc(net, head.logits_dimension, name=logits_scope)
                add_hidden_layer_summary(logits, logits_scope.name)

            return head.create_estimator_spec(
                features=features,
                mode=mode,
                labels=labels,
                optimizer=optimizers.get_optimizer_instance(
                    optimizer,
                    learning_rate=_LEARNING_RATE),
                logits=logits)

        super(DINEstimator, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from)
