#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

from models.focal_loss import binary_focal_loss


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def reverse_sigmoid(y):
    x = math.log(y/(1.0-y))
    return x


N = 100
probs = np.linspace(1e-6, 1.0, N)
logits = [reverse_sigmoid(y) for y in probs]

labels = [1.0] * N
gammas = (0, 0.5, 1, 2, 5)

logits = np.reshape(logits, [N, 1])
labels = np.reshape(labels, [N, 1])

sess = tf.Session()
plt.figure()
for gamma in gammas:
    loss = binary_focal_loss(labels, logits, gamma=gamma)
    loss = sess.run(loss)

    label = 'gamma={}'.format(gamma)
    if gamma == 0:
        label += ' (cross-entropy)'

    plt.plot(probs, loss, label=label)

plt.legend(loc='best', frameon=True, shadow=True)
plt.xlim(0, 1)
plt.ylim(0, 5)
plt.xlabel('Probability of positive class')
plt.ylabel('Loss')
plt.title('Plot of focal loss for different gamma',
          fontsize=14)
plt.show()
