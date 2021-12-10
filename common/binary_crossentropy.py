
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def binary_crossentropy(target, output):
    """Binary crossentropy between an output tensor and a target tensor.
    Arguments:
      target: A tensor with the same shape as `output`.
      output: A tensor. A probability distribution.

    Returns:
      A tensor.
    """

    output = tf.to_double(output)
    target = tf.to_double(target)

    epsilon = tf.convert_to_tensor(1e-7, dtype=tf.float64)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)

    bce = target * tf.math.log(output + epsilon)
    bce += (1. - target) * tf.math.log(1. - output + epsilon)

    return -bce
