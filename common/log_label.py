#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def log_label(labels, inverse=False):
    if inverse:
        return tf.exp(labels) - 1.0
    else:
        return tf.log(labels + 1.0)
