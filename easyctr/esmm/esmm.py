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


def _check_esmm_args(args):
    pass


def _build_esmm_logits(features,
                       group_feature_columns,
                       units,
                       is_training,
                       extra_options):
    _check_esmm_args(extra_options)

    bottom_columns_1 = group_feature_columns.get('esmm_bottom_1', [])
    bottom_columns_2 = group_feature_columns.get('esmm_bottom_2', [])

    # shared_bottom_1 = tf.feature_column.input_layer(features, bottom_columns_1)
    # shared_bottom_2 = tf.feature_column.input_layer(features, bottom_columns_2)
    bottom_1 = tf.feature_column.input_layer(features, bottom_columns_1)
    bottom_2 = tf.feature_column.input_layer(features, bottom_columns_2)
    bottom_3 = tf.feature_column.input_layer(features, bottom_columns_1)
    bottom_4 = tf.feature_column.input_layer(features, bottom_columns_2)

    esmm1 = add_hidden_layers(
        inputs=bottom_1,
        hidden_units=[512, 256, 128],
        activation_fn=tf.nn.relu,
        dropout=0.0,
        is_training=is_training,
        batch_norm=False,
        layer_norm=False,
        use_resnet=False,
        use_densenet=False,
        scope='esmm1')
    esmm2 = add_hidden_layers(
        inputs=bottom_2,
        hidden_units=[512, 256, 128],
        activation_fn=tf.nn.relu,
        dropout=0.0,
        is_training=is_training,
        batch_norm=False,
        layer_norm=False,
        use_resnet=False,
        use_densenet=False,
        scope='esmm2')
    esmm3 = add_hidden_layers(
        inputs=bottom_3,
        hidden_units=[512, 256, 128],
        activation_fn=tf.nn.relu,
        dropout=0.0,
        is_training=is_training,
        batch_norm=False,
        layer_norm=False,
        use_resnet=False,
        use_densenet=False,
        scope='esmm3')
    esmm4 = add_hidden_layers(
        inputs=bottom_4,
        hidden_units=[512, 256, 128],
        activation_fn=tf.nn.relu,
        dropout=0.0,
        is_training=is_training,
        batch_norm=False,
        layer_norm=False,
        use_resnet=False,
        use_densenet=False,
        scope='esmm4')

    # logits1 = tf.reduce_sum(esmm1*esmm2, -1, keepdims=True)
    # logits2 = tf.reduce_sum(esmm3*esmm4, -1, keepdims=True)
    logits1 = tf.concat([esmm1, esmm2], axis=1)
    logits1 = tf.layers.dense(logits1, units=1, activation=None)
    logits2 = tf.concat([esmm3, esmm4], axis=1)
    logits2 = tf.layers.dense(logits2, units=1, activation=None)

    return logits1, logits2, esmm1, esmm2, esmm3, esmm4

def _esmm_model_fn(
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

    # Build esmm logits
    esmm_parent_scope = 'esmm'
    optimizer = optimizers.get_optimizer_instance(
            optimizer, learning_rate=_LEARNING_RATE)
    check_no_sync_replicas_optimizer(optimizer)
    esmm_partitioner = tf.min_max_variable_partitioner(
        max_partitions=num_ps_replicas)
    with tf.variable_scope(
            esmm_parent_scope,
            values=tuple(six.itervalues(features)),
            partitioner=esmm_partitioner) as scope:
        esmm_absolute_scope = scope.name
        logits1, logits2, esmm1, esmm2, esmm3, esmm4 = _build_esmm_logits(
            features=features,
            group_feature_columns=group_feature_columns,
            units=head.logits_dimension,
            is_training=is_training,
            extra_options=extra_options)

        prob1 = tf.math.sigmoid(logits1)
        prob2 = tf.math.sigmoid(logits2)
        prob3 = prob1 * prob2

        if mode == tf.estimator.ModeKeys.PREDICT:
            prediction_outputs = {
                'esmm1': esmm1,
                'esmm2': esmm2,
                'esmm3': esmm3,
                'esmm4': esmm4,
            }
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(
                    outputs=prediction_outputs),
            }
            return tf.estimator.EstimatorSpec(
                mode,
                predictions=prediction_outputs,
                export_outputs=export_outputs)

        esmm_second_target_column = group_feature_columns.get(
            'esmm_second_target', [])
        esmm_second_target = tf.feature_column.input_layer(
            features, esmm_second_target_column)

        loss1 = tf.reduce_sum(binary_crossentropy(labels, prob1))
        loss2 = tf.reduce_sum(binary_crossentropy(esmm_second_target, prob3))
        loss = loss1 + loss2

        tf.summary.scalar('loss1', loss1)
        tf.summary.scalar('loss2', loss2)
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
            auc1 = tf.metrics.auc(labels=labels, predictions=prob1)
            auc2 = tf.metrics.auc(labels=labels, predictions=prob3)
            metrics = {
                'auc1': auc1,
                'auc2': auc2,
            }
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)


class EsmmEstimator(tf.estimator.Estimator):
    """An estimator for esmm model.
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

            return _esmm_model_fn(
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

        super(esmmEstimator, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from)
