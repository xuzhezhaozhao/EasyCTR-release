#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


"""
https://github.com/zhougr1993/DeepInterestNetwork/blob/master/din/model.py
"""


def din_attention(queries, keys, keys_length):
    '''
      queries:     [B, H],    可以理解为待排序物品
      keys:        [B, T, H], 变长序列, 例如浏览历史物品， 购买历史
      keys_length: [B],       变长序列的长度

      Return:
        weighted sum [B, H]
    '''
    dim = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  # [B, T*H]
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], dim])  # [B, T, H]
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # [B, T, 4*H]
    d_layer_1_all = tf.layers.dense(
        din_all, 80, activation=tf.nn.sigmoid, name='f1_att')  # [B, T, 80]

    d_layer_2_all = tf.layers.dense(
        d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')  # [B, T, 40]
    d_layer_3_all = tf.layers.dense(
        d_layer_2_all, 1, activation=None, name='f3_att')  # [B, T, 1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])   # [B, 1, T]
    outputs = d_layer_3_all  # [B, 1, T]
    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)   # [B, 1, T]
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)    # H**0.5

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]

    outputs = tf.squeeze(outputs, axis=1)  # [B, H]

    return outputs


def din_attention_multi_items(queries, keys, keys_length):
    '''
      queries:     [B, N, H] N is the number of ads
      keys:        [B, T, H]
      keys_length: [B]
    '''
    dim = queries.get_shape().as_list()[-1]
    queries_nums = queries.get_shape().as_list()[1]
    queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]])
    # shape : [B, N, T, H]
    queries = tf.reshape(
        queries, [-1, queries_nums, tf.shape(keys)[1], dim])
    max_len = tf.shape(keys)[1]
    keys = tf.tile(keys, [1, queries_nums, 1])
    # shape : [B, N, T, H]
    keys = tf.reshape(keys, [-1, queries_nums, max_len, dim])
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    d_layer_1_all = tf.layers.dense(
        din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
    d_layer_2_all = tf.layers.dense(
        d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
    d_layer_3_all = tf.layers.dense(
        d_layer_2_all, 1, activation=None, name='f3_att')
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])
    outputs = d_layer_3_all
    # Mask
    key_masks = tf.sequence_mask(keys_length, max_len)   # [B, T]
    key_masks = tf.tile(key_masks, [1, queries_nums])
    # shape : [B, N, 1, T]
    key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len])
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, N, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
    outputs = tf.reshape(outputs, [-1, 1, max_len])
    keys = tf.reshape(keys, [-1, max_len, dim])

    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, N, 1, H]
    outputs = tf.reshape(
        outputs, [-1, queries_nums, dim])   # [B, N, H]
    return outputs


def mlp_attention(queries, keys, keys_length, attention_hidden_units=[126, 64, 32]):
    '''
      queries:     [B, H],    可以理解为待排序物品
      keys:        [B, T, H], 变长序列, 例如浏览历史物品， 购买历史
      keys_length: [B],       变长序列的长度

      Return:
        weighted sum [B, H]
    '''
    dim = queries.get_shape().as_list()[-1]
    max_seq_len = tf.shape(keys)[1]  # padded_dim
    u_emb = tf.reshape(keys, shape=[-1, dim])
    a_emb = tf.reshape(tf.tile(queries, [1, max_seq_len]), shape=[-1, dim])
    net = tf.concat([u_emb, u_emb - a_emb, a_emb], axis=1)
    for units in attention_hidden_units:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)
    att_wgt = tf.reshape(att_wgt, shape=[-1, max_seq_len, 1])
    wgt_emb = tf.multiply(keys, att_wgt)  # [B, T, H]
    masks = tf.sequence_mask(keys_length, max_seq_len, dtype=tf.float32)
    masks = tf.expand_dims(tf.cast(masks, tf.float32), axis=-1)
    att_emb = tf.reduce_sum(tf.multiply(wgt_emb, masks), 1)
    att_emb.set_shape([None, dim])
    return att_emb
