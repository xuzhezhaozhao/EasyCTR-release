#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def mape_loss(labels, logits, delta=0.0):
    """This loss function generalizes mean absolute percentage error (MAPE),
       also known as mean absolute percentage deviation (MAPD),
       is a measure of prediction accuracy of a forecasting method in statistics,
       for example in trend estimation, also used as a loss function for
       regression problems in machine learning

      labels: A Tensor of the same type and shape as logits.
      logits:  A Tensor of type float32 or float64.

    Return:
      A Tensor of the same shape as logits with the componentwise logistic losses.
    """

    with tf.variable_scope('mape_loss'):
        logits = tf.convert_to_tensor(logits)
        labels = tf.convert_to_tensor(labels)
        loss = (2*tf.math.abs(logits-labels)) / (tf.math.abs(labels)+tf.math.abs(logits)+delta)

        return loss
