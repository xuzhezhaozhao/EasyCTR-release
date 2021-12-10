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

from common.estimator import head as head_lib
from common.estimator import optimizers


_LEARNING_RATE = 0.05


def _check_dssm_args(args):
    check_arg(args, 'leaky_relu_alpha', float)
    check_arg(args, 'swish_beta', float)
    check_arg(args, 'dssm_mode', str)
    check_arg(args, 'dssm1_hidden_units', list)
    check_arg(args, 'dssm1_activation_fn', str)
    check_arg(args, 'dssm1_dropout', float)
    check_arg(args, 'dssm1_gaussian_dropout', float)
    check_arg(args, 'dssm1_batch_norm', bool)
    check_arg(args, 'dssm1_layer_norm', bool)
    check_arg(args, 'dssm1_use_resnet', bool)
    check_arg(args, 'dssm1_use_densenet', bool)
    check_arg(args, 'dssm2_hidden_units', list)
    check_arg(args, 'dssm2_activation_fn', str)
    check_arg(args, 'dssm2_dropout', float)
    check_arg(args, 'dssm2_gaussian_dropout', float)
    check_arg(args, 'dssm2_batch_norm', bool)
    check_arg(args, 'dssm2_layer_norm', bool)
    check_arg(args, 'dssm2_use_resnet', bool)
    check_arg(args, 'dssm2_use_densenet', bool)
    check_arg(args, 'dssm_cosine_gamma', float)


def _build_dssm_logits(features,
                       dssm1_feature_columns,
                       dssm2_feature_columns,
                       units,
                       is_training,
                       extra_options):
    _check_dssm_args(extra_options)
    leaky_relu_alpha = extra_options['leaky_relu_alpha']
    swish_beta = extra_options['swish_beta']
    dssm_mode = extra_options['dssm_mode']
    hidden_units_1 = extra_options['dssm1_hidden_units']
    activation_fn_1 = extra_options['dssm1_activation_fn']
    dropout_1 = extra_options['dssm1_dropout']
    gaussian_dropout_1 = extra_options['dssm1_gaussian_dropout']
    batch_norm_1 = extra_options['dssm1_batch_norm']
    layer_norm_1 = extra_options['dssm1_layer_norm']
    use_resnet_1 = extra_options['dssm1_use_resnet']
    use_densenet_1 = extra_options['dssm1_use_densenet']
    hidden_units_2 = extra_options['dssm2_hidden_units']
    activation_fn_2 = extra_options['dssm2_activation_fn']
    dropout_2 = extra_options['dssm2_dropout']
    gaussian_dropout_2 = extra_options['dssm2_gaussian_dropout']
    batch_norm_2 = extra_options['dssm2_batch_norm']
    layer_norm_2 = extra_options['dssm2_layer_norm']
    use_resnet_2 = extra_options['dssm2_use_resnet']
    use_densenet_2 = extra_options['dssm2_use_densenet']
    cosine_gamma = extra_options['dssm_cosine_gamma']

    activation_fn_1 = get_activation_fn(
        activation_fn=activation_fn_1,
        leaky_relu_alpha=leaky_relu_alpha,
        swish_beta=swish_beta)
    activation_fn_2 = get_activation_fn(
        activation_fn=activation_fn_2,
        leaky_relu_alpha=leaky_relu_alpha,
        swish_beta=swish_beta)

    with tf.variable_scope('dssm1'):
        dssm1 = tf.feature_column.input_layer(features, dssm1_feature_columns)
        tf.logging.info("dssm1_input: {}".format(dssm1))
        dssm1 = add_hidden_layers(
            inputs=dssm1,
            hidden_units=hidden_units_1,
            activation_fn=activation_fn_1,
            dropout=dropout_1,
            is_training=is_training,
            batch_norm=batch_norm_1,
            layer_norm=layer_norm_1,
            use_resnet=use_resnet_1,
            use_densenet=use_densenet_1,
            gaussian_dropout=gaussian_dropout_1,
            scope='dnn')

    with tf.variable_scope('dssm2'):
        dssm2 = tf.feature_column.input_layer(features, dssm2_feature_columns)
        tf.logging.info("dssm2_input: {}".format(dssm2))
        dssm2 = add_hidden_layers(
            inputs=dssm2,
            hidden_units=hidden_units_2,
            activation_fn=activation_fn_2,
            dropout=dropout_2,
            is_training=is_training,
            batch_norm=batch_norm_2,
            layer_norm=layer_norm_2,
            use_resnet=use_resnet_2,
            use_densenet=use_densenet_2,
            gaussian_dropout=gaussian_dropout_2,
            scope='dnn')

    with tf.variable_scope('logits') as logits_scope:
        if dssm_mode == 'dot':
            logits = tf.reduce_sum(dssm1*dssm2, -1, keepdims=True)
        elif dssm_mode == 'concat':
            logits = tf.concat([dssm1, dssm2], axis=1)
            logits = tf.layers.dense(logits, units=1, activation=None)
        elif dssm_mode == 'cosine':
            # dssm1 = tf.nn.l2_normalize(dssm1)  # 该api可以防止norm为0的nan情况
            # dssm2 = tf.nn.l2_normalize(dssm2)
            # logits = tf.reduce_sum(dssm1*dssm2, -1, keepdims=True)
            # logits = tf.clip_by_value(logits, -1, 1.0)

            norm1 = tf.norm(dssm1, axis=1, keepdims=True)
            norm2 = tf.norm(dssm2, axis=1, keepdims=True)
            logits = (dssm1*dssm2)/(norm1*norm2+1e-8)
            logits = tf.reduce_sum(logits, -1, keepdims=True)
            logits = tf.clip_by_value(logits, -1, 1.0)
            logits = cosine_gamma * logits
        else:
            raise ValueError("unknown dssm mode '{}'".format(dssm_mode))
        add_hidden_layer_summary(dssm1, 'dssm1')
        add_hidden_layer_summary(dssm2, 'dssm2')
        add_hidden_layer_summary(logits, 'logits')

    tf.logging.info("logits = {}".format(logits))
    return logits, dssm1, dssm2


def _dssm_model_fn(
        features,
        labels,
        mode,
        head,
        dssm1_feature_columns,
        dssm2_feature_columns,
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

    # Build dssm logits
    dssm_parent_scope = 'dssm'
    optimizer = optimizers.get_optimizer_instance(
            optimizer, learning_rate=_LEARNING_RATE)
    check_no_sync_replicas_optimizer(optimizer)
    dssm_partitioner = tf.min_max_variable_partitioner(
        max_partitions=num_ps_replicas)
    with tf.variable_scope(
            dssm_parent_scope,
            values=tuple(six.itervalues(features)),
            partitioner=dssm_partitioner) as scope:
        dssm_absolute_scope = scope.name
        logits, dssm1, dssm2 = _build_dssm_logits(
            features=features,
            dssm1_feature_columns=dssm1_feature_columns,
            dssm2_feature_columns=dssm2_feature_columns,
            units=head.logits_dimension,
            is_training=is_training,
            extra_options=extra_options)

    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""
        train_ops = []
        global_step = tf.train.get_global_step()
        if logits is not None:
            train_ops.append(
                optimizer.minimize(
                    loss,
                    var_list=tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES,
                        scope=dssm_absolute_scope)))
        train_op = tf.group(*train_ops)
        with tf.control_dependencies([train_op]):
            return tf.assign_add(global_step, 1).op

    if run_mode == 'easyrecall' and mode == tf.estimator.ModeKeys.PREDICT:
        item_embeddings = extra_options['item_embeddings']
        item_keys = extra_options['item_keys']
        if item_embeddings is None:
            prediction_outputs = {
                'item_embeddings': dssm2,
            }
        else:
            item_embeddings = tf.convert_to_tensor(item_embeddings, dtype=tf.float32)
            # TODO(zhezhaoxu)
            recall_k = tf.math.minimum(100, len(item_keys))
            recall_k = tf.to_int32(recall_k)
            logits = tf.matmul(dssm1, item_embeddings)
            scores, ids = tf.nn.top_k(logits, recall_k)
            table = tf.contrib.lookup.index_to_string_table_from_tensor(
                mapping=item_keys,
                default_value='')
            prediction_outputs = {
                    'scores': scores,
                    'items': table.lookup(tf.cast(ids, tf.int64)),
                }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(outputs=prediction_outputs),
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=prediction_outputs, export_outputs=export_outputs)

    if True and mode == tf.estimator.ModeKeys.PREDICT:  # TODO(zhezhaoxu) 粗排
        dssm_mode = extra_options['dssm_mode']
        if dssm_mode not in ('dot', 'cosine'):
            raise ValueError(
                "dssm_mode '{}' error for coarse ranking, "
                "valid modes are 'dot' and 'cosine'")
        prediction_outputs = {
            'dssm1': dssm1,
            'dssm2': dssm2,
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(outputs=prediction_outputs),
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=prediction_outputs, export_outputs=export_outputs)

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)


class DSSMEstimator(tf.estimator.Estimator):
    """An estimator for DSSM model.
    """

    def __init__(
            self,
            dssm1_columns,
            dssm2_columns,
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

            return _dssm_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                dssm1_feature_columns=dssm1_columns,
                dssm2_feature_columns=dssm2_columns,
                optimizer=optimizer,
                input_layer_partitioner=input_layer_partitioner,
                config=config,
                run_mode=run_mode,
                extra_options=extra_options)

        super(DSSMEstimator, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from)
