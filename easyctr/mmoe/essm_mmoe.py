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


def _check_essm_mmoe_args(args):
    pass


def _build_essm_mmoe_logits(features,
                       group_feature_columns,
                       units,
                       is_training,
                       extra_options):
    _check_essm_mmoe_args(extra_options)

    bottom_columns_1 = group_feature_columns.get('essm_bottom_1', [])
    bottom_columns_2 = group_feature_columns.get('essm_bottom_2', [])

    shared_bottom_1 = tf.feature_column.input_layer(features, bottom_columns_1)
    shared_bottom_2 = tf.feature_column.input_layer(features, bottom_columns_2)
    final_bottom_column = tf.concat([shared_bottom_1, shared_bottom_2], axis=1, name='dense_vector')  # concate_all_bottom_columns

    # experts
    experts_weight = tf.get_variable(name='experts_weight',
                                     dtype=tf.float32,
                                     shape=(final_bottom_column.get_shape()[1], 128, 3),
                                     initializer=tf.contrib.layers.xavier_initializer())  # params.experts_units, params.experts_num
    experts_bias = tf.get_variable(name='expert_bias',
                                   dtype=tf.float32,
                                   shape=(128, 3),
                                   initializer=tf.contrib.layers.xavier_initializer())

    # gates
    gate1_weight = tf.get_variable(name='gate1_weight',
                                   dtype=tf.float32,
                                   shape=(final_bottom_column.get_shape()[1], 3),
                                   initializer=tf.contrib.layers.xavier_initializer())
    gate1_bias = tf.get_variable(name='gate1_bias',
                                 dtype=tf.float32,
                                 shape=(3,),
                                 initializer=tf.contrib.layers.xavier_initializer())
    gate2_weight = tf.get_variable(name='gate2_weight',
                                   dtype=tf.float32,
                                   shape=(final_bottom_column.get_shape()[1], 3),
                                   initializer=tf.contrib.layers.xavier_initializer())
    gate2_bias = tf.get_variable(name='gate2_bias',
                                 dtype=tf.float32,
                                 shape=(3,),
                                 initializer=tf.contrib.layers.xavier_initializer())

    # # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
    experts_output = tf.tensordot(final_bottom_column, experts_weight, axes=1)
    use_experts_bias = True
    if use_experts_bias:
        experts_output = tf.add(experts_output, experts_bias)
    experts_output = tf.nn.relu(experts_output)

    # # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
    gate1_output = tf.matmul(final_bottom_column, gate1_weight)
    gate2_output = tf.matmul(final_bottom_column, gate2_weight)
    user_gate_bias = True
    if user_gate_bias:
        gate1_output = tf.add(gate1_output, gate1_bias)
        gate2_output = tf.add(gate2_output, gate2_bias)
    gate1_output = tf.nn.softmax(gate1_output)
    gate2_output = tf.nn.softmax(gate2_output)

    # # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
    label1_input = tf.multiply(experts_output, tf.expand_dims(gate1_output, axis=1))
    label1_input = tf.reduce_sum(label1_input, axis=2)
    label1_input = tf.reshape(label1_input, [-1, 128])
    label2_input = tf.multiply(experts_output, tf.expand_dims(gate2_output, axis=1))
    label2_input = tf.reduce_sum(label2_input, axis=2)
    label2_input = tf.reshape(label2_input, [-1, 128])


    essm1 = add_hidden_layers(
        inputs=label1_input,
        hidden_units=[128, 256, 128, 128, 100],
        activation_fn=tf.nn.relu,
        dropout=0.0,
        is_training=is_training,
        batch_norm=False,
        layer_norm=False,
        use_resnet=False,
        use_densenet=False,
        scope='essm1')
    essm2 = add_hidden_layers(
        inputs=label1_input,
        hidden_units=[128, 256, 128, 128, 100],
        activation_fn=tf.nn.relu,
        dropout=0.0,
        is_training=is_training,
        batch_norm=False,
        layer_norm=False,
        use_resnet=False,
        use_densenet=False,
        scope='essm2')
    essm3 = add_hidden_layers(
        inputs=label2_input,
        hidden_units=[128, 256, 128, 128, 100],
        activation_fn=tf.nn.relu,
        dropout=0.0,
        is_training=is_training,
        batch_norm=False,
        layer_norm=False,
        use_resnet=False,
        use_densenet=False,
        scope='essm3')
    essm4 = add_hidden_layers(
        inputs=label2_input,
        hidden_units=[128, 256, 128, 128, 100],
        activation_fn=tf.nn.relu,
        dropout=0.0,
        is_training=is_training,
        batch_norm=False,
        layer_norm=False,
        use_resnet=False,
        use_densenet=False,
        scope='essm4')

    logits1 = tf.concat([essm1, essm2], axis=1)
    logits1 = tf.layers.dense(logits1, units=1, activation=None)
    logits2 = tf.concat([essm3, essm4], axis=1)
    logits2 = tf.layers.dense(logits2, units=1, activation=None)

    return logits1, logits2, essm1, essm2, essm3, essm4

def _essm_mmoe_model_fn(
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

    # Build essm logits
    essm_parent_scope = 'essm'
    optimizer = optimizers.get_optimizer_instance(
            optimizer, learning_rate=_LEARNING_RATE)
    check_no_sync_replicas_optimizer(optimizer)
    essm_partitioner = tf.min_max_variable_partitioner(
        max_partitions=num_ps_replicas)
    with tf.variable_scope(
            essm_parent_scope,
            values=tuple(six.itervalues(features)),
            partitioner=essm_partitioner) as scope:
        essm_absolute_scope = scope.name
        logits1, logits2, essm1, essm2, essm3, essm4 = _build_essm_mmoe_logits(
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
                'essm1': essm1,
                'essm2': essm2,
                'essm3': essm3,
                'essm4': essm4,
            }
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(
                    outputs=prediction_outputs),
            }
            return tf.estimator.EstimatorSpec(
                mode,
                predictions=prediction_outputs,
                export_outputs=export_outputs)

        essm_second_target_column = group_feature_columns.get(
            'essm_second_target', [])
        essm_second_target = tf.feature_column.input_layer(
            features, essm_second_target_column)

        loss1 = tf.reduce_sum(binary_crossentropy(labels, prob1))
        loss2 = tf.reduce_sum(binary_crossentropy(essm_second_target, prob3))
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


class ESSMMMOEEstimator(tf.estimator.Estimator):
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

            return _essm_mmoe_model_fn(
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

        super(ESSMMMOEEstimator, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from)
