#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.bi_tempered_loss import bi_tempered_binary_logistic_loss
from common.focal_loss import binary_focal_loss
from common.mape_loss import mape_loss


def configure_loss_fn(opts):
    if opts.loss_fn == 'default':
        loss_wrapper = None
    elif opts.loss_fn == 'bi_tempered':
        """Robust Bi-Tempered Logistic Loss Based on Bregman Divergences"""

        def loss_wrapper(labels, logits):
            return bi_tempered_binary_logistic_loss(
                activations=logits,  # 根据论文, activations 就是网络 logits
                labels=labels,
                t1=opts.bi_tempered_loss_t1,
                t2=opts.bi_tempered_loss_t2,
                label_smoothing=opts.bi_tempered_loss_label_smoothing,
                num_iters=opts.bi_tempered_loss_num_iters)
    elif opts.loss_fn == 'focal':
        """Focal Loss"""
        def loss_wrapper(labels, logits):
            return binary_focal_loss(
                labels=labels,
                logits=logits,
                gamma=opts.focal_loss_gamma)
    elif opts.loss_fn == 'mape':
        """Mean absolute precentage error"""
        def loss_wrapper(labels, logits):
            return mape_loss(
                labels=labels,
                logits=logits,
                delta=opts.mape_loss_delta)
    elif opts.loss_fn == 'huber':
        def loss_wrapper(labels, logits):
            return tf.losses.huber_loss(
                labels=labels,
                predictions=logits,
                delta=opts.huber_loss_delta,
                reduction='none')
    else:
        raise ValueError("Unknown loss_fn '{}'".format(opts.loss_fn))

    return loss_wrapper
