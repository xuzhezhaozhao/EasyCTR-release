#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import tensorflow as tf

"""
https://github.com/artemmavrin/focal-loss/blob/master/src/focal_loss/_focal_loss.py
"""


def binary_focal_loss(labels, logits, gamma, alpha=1.0):
    """This loss function generalizes binary cross-entropy by introducing a
    hyperparameter called the *focusing parameter* that allows hard-to-classify
    examples to be penalized more heavily relative to easy-to-classify examples.

      labels: A Tensor of the same type and shape as logits.
      logits:  A Tensor of type float32 or float64.
      gamma: Focal loss gamma parameter in the formula

    Return:
      A Tensor of the same shape as logits with the componentwise logistic losses.
    """

    with tf.variable_scope('binary_focal_loss'):
        logits = tf.convert_to_tensor(logits)
        probs = tf.math.sigmoid(logits)

        if alpha == 1.0:
            loss_func = tf.nn.sigmoid_cross_entropy_with_logits
        else:
            loss_func = partial(tf.nn.weighted_cross_entropy_with_logits,
                                pos_weight=alpha)
        loss = loss_func(labels=labels, logits=logits)
        modulation_pos = (1 - probs) ** gamma
        modulation_neg = probs ** gamma
        cond = tf.dtypes.cast(labels, dtype=tf.bool)
        modulation = tf.where(cond, modulation_pos, modulation_neg)
        loss = modulation * loss
        loss = tf.reduce_sum(loss, -1, keepdims=True)

        return loss
