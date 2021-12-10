#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def swish(x, beta=1.0):
    """Swish activation function

    Searching for Activation Functions （Updated version of "Swish: a Self-Gated Activation Function"）

    """

    with tf.variable_scope('swish'):
        y = x*tf.nn.sigmoid(beta*x)

        return y
