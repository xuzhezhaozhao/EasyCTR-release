#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.common import warmup_learning_rate
from common.yellowfin import YFOptimizer
from common.lazy_adam_optimizer import LazyAdamOptimizer


def configure_deep_learning_rate(opts):
    global_step = tf.train.get_global_step()
    if opts.deep_learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(
            opts.deep_learning_rate,
            global_step,
            opts.deep_decay_steps,
            opts.deep_learning_rate_decay_factor,
            staircase=True,
            name='deep_exponential_decay_learning_rate')
    elif opts.deep_learning_rate_decay_type == 'fixed':
        return tf.constant(opts.deep_learning_rate,
                           name='deep_fixed_learning_rate')
    elif opts.deep_learning_rate_decay_type == 'polynomial':
        decay_steps = opts.deep_decay_steps
        if opts.deep_decay_steps <= 0:
            decay_steps = opts.total_steps
        return tf.train.polynomial_decay(
            opts.deep_learning_rate,
            global_step,
            decay_steps,
            opts.deep_end_learning_rate,
            power=opts.deep_polynomial_decay_power,
            cycle=False,
            name='deep_polynomial_decay_learning_rate')
    elif opts.deep_learning_rate_decay_type == 'warmup':
        learning_rate = warmup_learning_rate(
            init_lr=opts.deep_learning_rate,
            warmup_rate=opts.deep_warmup_rate,
            global_step=global_step,
            total_steps=opts.total_steps)
        return learning_rate
    elif opts.deep_learning_rate_decay_type == 'cosine':
        decay_steps = opts.deep_decay_steps
        if opts.deep_decay_steps <= 0:
            decay_steps = opts.total_steps
        return tf.train.cosine_decay(
            learning_rate=opts.deep_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            alpha=opts.deep_cosine_decay_alpha)
    else:
        raise ValueError('deep_learning_rate_decay_type [{}] was not recognized'
                         .format(opts.deep_learning_rate_decay_type))


def configure_wide_learning_rate(opts):
    global_step = tf.train.get_global_step()
    if opts.wide_learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(
            opts.wide_learning_rate,
            global_step,
            opts.wide_decay_steps,
            opts.wide_learning_rate_decay_factor,
            staircase=True,
            name='wide_exponential_decay_learning_rate')
    elif opts.wide_learning_rate_decay_type == 'fixed':
        return tf.constant(opts.wide_learning_rate,
                           name='wide_fixed_learning_rate')
    elif opts.wide_learning_rate_decay_type == 'polynomial':
        decay_steps = opts.wide_decay_steps
        if opts.wide_decay_steps <= 0:
            decay_steps = opts.total_steps

        return tf.train.polynomial_decay(
            opts.wide_learning_rate,
            global_step,
            decay_steps,
            opts.wide_end_learning_rate,
            power=opts.wide_polynomial_decay_power,
            cycle=False,
            name='wide_polynomial_decay_learning_rate')
    else:
        raise ValueError('wide_learning_rate_decay_type [{}] was not recognized'
                         .format(opts.wide_learning_rate_decay_type))


def configure_deep_optimizer(opts):
    """Configures the deep optimizer used for training."""

    learning_rate = configure_deep_learning_rate(opts)
    tf.summary.scalar('deep_learning_rate', learning_rate)
    if opts.deep_optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=opts.deep_adadelta_rho,
            epsilon=opts.deep_opt_epsilon)
    elif opts.deep_optimizer == 'adagrad':
        init_value = opts.deep_adagrad_initial_accumulator_value
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=init_value)
    elif opts.deep_optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=opts.deep_adam_beta1,
            beta2=opts.deep_adam_beta2,
            epsilon=opts.deep_opt_epsilon)
    elif opts.deep_optimizer == 'lazy_adam':
        optimizer = LazyAdamOptimizer(
            learning_rate,
            beta1=opts.deep_lazy_adam_beta1,
            beta2=opts.deep_lazy_adam_beta2,
            epsilon=opts.deep_opt_epsilon)
    elif opts.deep_optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=opts.deep_ftrl_learning_rate_power,
            initial_accumulator_value=opts.deep_ftrl_initial_accumulator_value,
            l1_regularization_strength=opts.deep_ftrl_l1,
            l2_regularization_strength=opts.deep_ftrl_l2)
    elif opts.deep_optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=opts.deep_momentum,
            name='Momentum')
    elif opts.deep_optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=opts.deep_rmsprop_decay,
            momentum=opts.deep_rmsprop_momentum,
            epsilon=opts.deep_opt_epsilon)
    elif opts.deep_optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif opts.deep_optimizer == 'proximal_adagrad':
        optimizer = tf.train.ProximalAdagradOptimizer(
            learning_rate,
            initial_accumulator_value=opts.deep_proximal_adagrad_initial_accumulator_value,
            l1_regularization_strength=opts.deep_proximal_adagrad_l1,
            l2_regularization_strength=opts.deep_proximal_adagrad_l2)
    elif opts.deep_optimizer == 'yellowfin':
        optimizer = YFOptimizer(
            learning_rate,
            momentum=opts.deep_momentum)
    elif opts.deep_optimizer == 'adamw':
        if opts.num_gpus > 1:
            from common.optimization_multi_gpu import AdamWeightDecayOptimizer
        else:
            from common.optimization import AdamWeightDecayOptimizer

        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=opts.deep_adamw_weight_decay_rate,
            beta_1=opts.deep_adamw_beta1,
            beta_2=opts.deep_adamw_beta2,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias",
                                       "BatchNorm", "batch_norm"])
    else:
        raise ValueError('Optimizer [%s] was not recognized'
                         % opts.deep_optimizer)
    return optimizer


def configure_wide_optimizer(opts):
    """Configures the wide optimizer used for training."""

    learning_rate = configure_wide_learning_rate(opts)
    tf.summary.scalar('wide_learning_rate', learning_rate)
    if opts.wide_optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=opts.wide_adadelta_rho,
            epsilon=opts.wide_opt_epsilon)
    elif opts.wide_optimizer == 'adagrad':
        init_value = opts.deep_adagrad_initial_accumulator_value
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=init_value)
    elif opts.wide_optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=opts.wide_adam_beta1,
            beta2=opts.wide_adam_beta2,
            epsilon=opts.wide_opt_epsilon)
    elif opts.wide_optimizer == 'lazy_adam':
        optimizer = LazyAdamOptimizer(
            learning_rate,
            beta1=opts.wide_lazy_adam_beta1,
            beta2=opts.wide_lazy_adam_beta2,
            epsilon=opts.wide_opt_epsilon)
    elif opts.wide_optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=opts.wide_ftrl_learning_rate_power,
            initial_accumulator_value=opts.wide_ftrl_initial_accumulator_value,
            l1_regularization_strength=opts.wide_ftrl_l1,
            l2_regularization_strength=opts.wide_ftrl_l2)
    elif opts.wide_optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=opts.wide_momentum,
            name='Momentum')
    elif opts.wide_optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=opts.wide_rmsprop_decay,
            momentum=opts.wide_rmsprop_momentum,
            epsilon=opts.wide_opt_epsilon)
    elif opts.wide_optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif opts.wide_optimizer == 'proximal_adagrad':
        optimizer = tf.train.ProximalAdagradOptimizer(
            learning_rate,
            initial_accumulator_value=opts.wide_proximal_adagrad_initial_accumulator_value,
            l1_regularization_strength=opts.wide_proximal_adagrad_l1,
            l2_regularization_strength=opts.wide_proximal_adagrad_l2)
    elif opts.wide_optimizer == 'yellowfin':
        optimizer = YFOptimizer(
            learning_rate,
            momentum=opts.wide_momentum)
    else:
        raise ValueError('Optimizer [%s] was not recognized'
                         % opts.wide_optimizer)
    return optimizer


def get_optimizer(opts):
    if opts.wide_optimizer == 'default':
        wide_optimizer = 'Ftrl'
    else:
        def wide_optimizer():
            return configure_wide_optimizer(opts)

    if opts.deep_optimizer == 'default':
        deep_optimizer = 'Adagrad'
    else:
        def deep_optimizer():
            return configure_deep_optimizer(opts)

    return wide_optimizer, deep_optimizer
