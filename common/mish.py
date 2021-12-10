#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def mish(x):
    """Mish activation function

    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    """

    with tf.variable_scope('mish'):
        y = x*tf.math.tanh(tf.math.softplus(x))

        return y
