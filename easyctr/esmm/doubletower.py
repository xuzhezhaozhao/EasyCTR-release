#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import six

from common.common import check_no_sync_replicas_optimizer
from common.common import check_arg
from common.common import add_hidden_layer_summary
from common.common import add_hidden_layers
from common.common import get_activation_fn
from common.binary_crossentropy import binary_crossentropy

from common.estimator import head as head_lib
from common.estimator import optimizers


_LEARNING_RATE = 0.05


def _check_double_tower_args(args):
    pass


def _build_dtower_logits(features,
                       group_feature_columns,
                       units,
                       is_training,
                       extra_options):
    _check_double_tower_args(extra_options)

    bottom_columns_1 = group_feature_columns.get('dtower_bottom_1', [])
    bottom_columns_2 = group_feature_columns.get('dtower_bottom_2', [])

    bottom_1 = tf.feature_column.input_layer(features, bottom_columns_1)
    bottom_2 = tf.feature_column.input_layer(features, bottom_columns_2)

    utower = add_hidden_layers(
        inputs=bottom_1,
        hidden_units=[512, 256, 128, 64],
        activation_fn=tf.nn.relu,
        dropout=0.0,
        is_training=is_training,
        batch_norm=False,
        layer_norm=False,
        use_resnet=False,
        use_densenet=False,
        scope='utower')
    itower = add_hidden_layers(
        inputs=bottom_2,
        hidden_units=[512, 256, 128, 64],
        activation_fn=tf.nn.relu,
        dropout=0.0,
        is_training=is_training,
        batch_norm=False,
        layer_norm=False,
        use_resnet=False,
        use_densenet=False,
        scope='itower')

    logits = tf.concat([utower, itower], axis=1)
    logits = tf.layers.dense(logits, units=1, activation=None)

    return logits, utower, itower

def _dtower_model_fn(
        features,
        labels,
        mode,
        head,
        group_feature_columns,
        optimizer,
        input_layer_partitioner,
        config,
        run_mode,
        extra_options):

    num_ps_replicas = config.num_ps_replicas if config else 0
    tf.logging.info("num_ps_replicas = {}".format(num_ps_replicas))
    input_layer_partitioner = input_layer_partitioner or (
        tf.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    dtwoer_parent_scope = 'dtower'
    optimizer = optimizers.get_optimizer_instance(
            optimizer, learning_rate=_LEARNING_RATE)
    check_no_sync_replicas_optimizer(optimizer)
    dtower_partitioner = tf.min_max_variable_partitioner(
        max_partitions=num_ps_replicas)
    with tf.variable_scope(
            dtwoer_parent_scope,
            values=tuple(six.itervalues(features)),
            partitioner=dtower_partitioner) as scope:
        dtower_absolute_scope = scope.name
        logits, utower, itower = _build_dtower_logits(
            features=features,
            group_feature_columns=group_feature_columns,
            units=head.logits_dimension,
            is_training=is_training,
            extra_options=extra_options)

        prob = tf.math.sigmoid(logits)

        if mode == tf.estimator.ModeKeys.PREDICT:
            prediction_outputs = {
                'utower': utower,
                'itower': itower,
            }
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(
                    outputs=prediction_outputs),
            }
            return tf.estimator.EstimatorSpec(
                mode,
                predictions=prediction_outputs,
                export_outputs=export_outputs)


        loss = tf.reduce_sum(binary_crossentropy(labels, prob))

        tf.summary.scalar('loss', loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_global_step()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for bn
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            auc = tf.metrics.auc(labels=labels, predictions=prob)
            metrics = {
                'auc': auc
            }
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

class DtowerEstimator(tf.estimator.Estimator):
    """An estimator for ESSM model.
    """

    def __init__(
            self,
            group_columns,
            model_dir=None,
            n_classes=2,
            weight_column=None,
            label_vocabulary=None,
            optimizer='Adagrad',
            input_layer_partitioner=None,
            config=None,
            warm_start_from=None,
            loss_reduction=tf.losses.Reduction.SUM,
            loss_fn=None,
            run_mode=None,
            extra_options=None):

        def _model_fn(features, labels, mode, config):

            head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
                n_classes, weight_column, label_vocabulary, loss_reduction, loss_fn)

            return _dtower_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                group_feature_columns=group_columns,
                optimizer=optimizer,
                input_layer_partitioner=input_layer_partitioner,
                config=config,
                run_mode=run_mode,
                extra_options=extra_options)

        super(DtowerEstimator, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from)


