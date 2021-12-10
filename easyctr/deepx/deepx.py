#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import six

from common.estimator import head as head_lib
from common.estimator import optimizers

from common.common import validate_feature_columns
from common.common import check_no_sync_replicas_optimizer
from common.common import get_feature_vectors
from common.common import fc
from easyctr.deepx.linear import get_linear_logits
from easyctr.deepx.fm import get_fm_logits
from easyctr.deepx.fwfm import get_fwfm_logits
from easyctr.deepx.afm import get_afm_logits
from easyctr.deepx.iafm import get_iafm_logits
from easyctr.deepx.ifm import get_ifm_logits
from easyctr.deepx.kfm import get_kfm_logits
from easyctr.deepx.wkfm import get_wkfm_logits
from easyctr.deepx.nifm import get_nifm_logits
from easyctr.deepx.cin import get_cin_logits
from easyctr.deepx.cross import get_cross_logits
from easyctr.deepx.autoint import get_autoint_logits
from easyctr.deepx.dnn import get_dnn_logits
from easyctr.deepx.multi_dnn import get_multi_dnn_logits
from easyctr.deepx.nfm import get_nfm_logits
from easyctr.deepx.nkfm import get_nkfm_logits
from easyctr.deepx.ccpm import get_ccpm_logits
from easyctr.deepx.ipnn import get_ipnn_logits
from easyctr.deepx.kpnn import get_kpnn_logits
from easyctr.deepx.pin import get_pin_logits
from easyctr.deepx.fibinet import get_fibinet_logits


_LEARNING_RATE = 0.05


def _check_model_input(features, linear_feature_columns, deep_feature_columns):
    if not isinstance(features, dict):
        raise ValueError('features should be a dictionary of `Tensor`s. '
                         'Given type: {}'.format(type(features)))
    if not linear_feature_columns and not deep_feature_columns:
        raise ValueError(
            'Either linear_feature_columns or deep_feature_columns must be'
            'defined.')


def _build_deepx_logits(features,
                        feature_columns,
                        selector_feature_columns,
                        model_slots,
                        units,
                        is_training,
                        extra_options):

    shared_feature_vectors = get_feature_vectors(
        features=features,
        feature_columns=feature_columns,
        scope='shared_feature_vectors')

    kwargs = {}
    kwargs['features'] = features
    kwargs['feature_columns'] = feature_columns
    kwargs['selector_feature_columns'] = selector_feature_columns
    kwargs['shared_feature_vectors'] = shared_feature_vectors
    kwargs['units'] = units
    kwargs['is_training'] = is_training
    kwargs['extra_options'] = extra_options

    logits_fn_map = {
        # shallow
        'fm': get_fm_logits,
        'fwfm': get_fwfm_logits,
        'afm': get_afm_logits,
        'iafm': get_iafm_logits,
        'ifm': get_ifm_logits,
        'kfm': get_kfm_logits,
        'wkfm': get_wkfm_logits,
        'nifm': get_nifm_logits,
        'cin': get_cin_logits,
        'cross': get_cross_logits,

        # maybe deep
        'autoint': get_autoint_logits,
        'dnn': get_dnn_logits,
        'multi_dnn': get_multi_dnn_logits,
        'nfm': get_nfm_logits,
        'nkfm': get_nkfm_logits,
        'ccpm': get_ccpm_logits,
        'ipnn': get_ipnn_logits,
        'kpnn': get_kpnn_logits,
        'pin': get_pin_logits,
        'fibinet': get_fibinet_logits,
    }

    logits_list = []
    for model in model_slots:
        if model == 'linear':   # linear model will use seperated optimizer
            continue
        if model not in logits_fn_map:
            raise ValueError("Unknown model ''".format(model))
        logits = logits_fn_map[model](**kwargs)
        if isinstance(logits, (list, tuple)):
            logits_list.extend(logits)
        else:
            logits_list.append(logits)

    return logits_list


def _deepx_model_fn(
        features,
        labels,
        mode,
        head,
        linear_feature_columns,
        linear_optimizer,
        deep_feature_columns,
        deep_optimizer,
        selector_feature_columns,
        model_slots,
        extend_feature_mode,
        input_layer_partitioner,
        config,
        linear_sparse_combiner,
        use_seperated_logits,
        use_weighted_logits,
        extra_options):

    _check_model_input(features, linear_feature_columns, deep_feature_columns)

    num_ps_replicas = config.num_ps_replicas if config else 0
    tf.logging.info("num_ps_replicas = {}".format(num_ps_replicas))
    input_layer_partitioner = input_layer_partitioner or (
        tf.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Build deepx logits
    deepx_parent_scope = 'deepx'
    if not deep_feature_columns:
        deepx_logits_list = []
    else:
        deep_optimizer = optimizers.get_optimizer_instance(
                deep_optimizer, learning_rate=_LEARNING_RATE)
        check_no_sync_replicas_optimizer(deep_optimizer)
        deep_partitioner = (
            tf.min_max_variable_partitioner(
                max_partitions=num_ps_replicas))

        with tf.variable_scope(
                deepx_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=deep_partitioner) as scope:
            deepx_absolute_scope = scope.name

            deepx_logits_list = _build_deepx_logits(
                features=features,
                feature_columns=deep_feature_columns,
                selector_feature_columns=selector_feature_columns,
                model_slots=model_slots,
                units=head.logits_dimension,
                is_training=is_training,
                extra_options=extra_options)

    linear_parent_scope = 'linear'
    use_linear = 'linear' in model_slots
    if not linear_feature_columns or not use_linear:
        linear_logits = None
    else:
        linear_optimizer = optimizers.get_optimizer_instance(
                linear_optimizer,
                learning_rate=_LEARNING_RATE)
        check_no_sync_replicas_optimizer(linear_optimizer)
        with tf.variable_scope(
                linear_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=input_layer_partitioner) as scope:
            linear_absolute_scope = scope.name
            linear_logits = get_linear_logits(
                features,
                linear_feature_columns,
                head.logits_dimension,
                linear_sparse_combiner,
                scope)

    # Combine logits and build full model.
    logits = []
    logits.extend(deepx_logits_list)
    if linear_logits is not None:
        logits.append(linear_logits)

    assert not (use_seperated_logits and use_weighted_logits), \
        "'use_seperated_logits' and 'use_weighted_logits' should not be set " \
        "at the same time."

    if not use_seperated_logits:
        logits = sum(logits)

    if use_weighted_logits and len(logits) > 1:
        logits = tf.concat(logits, axis=1)
        logits = fc(logits, head.logits_dimension)

    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""
        train_ops = []
        global_step = tf.train.get_global_step()
        if len(deepx_logits_list) > 0:
            train_ops.append(
                deep_optimizer.minimize(
                    loss,
                    var_list=tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES,
                        scope=deepx_absolute_scope)))
        if linear_logits is not None:
            train_ops.append(
                linear_optimizer.minimize(
                    loss,
                    var_list=tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES,
                        scope=linear_absolute_scope)))
        train_op = tf.group(*train_ops)
        with tf.control_dependencies([train_op]):
            return tf.assign_add(global_step, 1).op

    return head.create_estimator_spec(
            features=features,
            mode=mode,
            labels=labels,
            train_op_fn=_train_op_fn,
            logits=logits,
            regularization_losses=[tf.losses.get_regularization_loss()])


class DeepXClassifier(tf.estimator.Estimator):

    def __init__(
        self,
        model_dir=None,
        linear_feature_columns=None,
        linear_optimizer='Ftrl',
        deep_feature_columns=None,
        deep_optimizer='Adagrad',
        selector_feature_columns=None,
        model_slots=None,
        extend_feature_mode=None,
        n_classes=2,
        weight_column=None,
        label_vocabulary=None,
        input_layer_partitioner=None,
        config=None,
        warm_start_from=None,
        loss_reduction=tf.losses.Reduction.SUM,
        loss_fn=None,
        linear_sparse_combiner='sum',
        use_seperated_logits=False,
        use_weighted_logits=False,
        extra_options=None
    ):
        def _model_fn(features, labels, mode, config):
            head = head_lib._binary_logistic_or_multi_class_head(
                n_classes=n_classes,
                weight_column=weight_column,
                label_vocabulary=label_vocabulary,
                loss_reduction=loss_reduction,
                loss_fn=loss_fn)

            self._feature_columns = validate_feature_columns(
                linear_feature_columns=linear_feature_columns,
                dnn_feature_columns=deep_feature_columns)

            return _deepx_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                linear_feature_columns=linear_feature_columns,
                linear_optimizer=linear_optimizer,
                deep_feature_columns=deep_feature_columns,
                deep_optimizer=deep_optimizer,
                selector_feature_columns=selector_feature_columns,
                model_slots=model_slots,
                extend_feature_mode=extend_feature_mode,
                input_layer_partitioner=input_layer_partitioner,
                config=config,
                linear_sparse_combiner=linear_sparse_combiner,
                use_seperated_logits=use_seperated_logits,
                use_weighted_logits=use_weighted_logits,
                extra_options=extra_options)

        super(DeepXClassifier, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            warm_start_from=warm_start_from)


class DeepXRegressor(tf.estimator.Estimator):

    def __init__(
        self,
        model_dir=None,
        linear_feature_columns=None,
        linear_optimizer='Ftrl',
        deep_feature_columns=None,
        deep_optimizer='Adagrad',
        selector_feature_columns=None,
        model_slots=None,
        extend_feature_mode=None,
        weight_column=None,
        label_vocabulary=None,
        input_layer_partitioner=None,
        config=None,
        warm_start_from=None,
        loss_reduction=tf.losses.Reduction.SUM,
        loss_fn=None,
        label_fn=None,
        linear_sparse_combiner='sum',
        use_seperated_logits=False,
        use_weighted_logits=False,
        extra_options=None
    ):
        def _model_fn(features, labels, mode, config):
            inverse_link_fn = None
            if labels is not None and label_fn is not None:
                labels = label_fn(labels, inverse=False)
            if label_fn is not None:
                inverse_link_fn = lambda logits: label_fn(logits, inverse=True)

            if labels is not None:
                tf.summary.histogram('labels', labels)
            head = head_lib._regression_head(  # pylint: disable=protected-access
                label_dimension=1,
                weight_column=weight_column,
                loss_reduction=loss_reduction,
                loss_fn=loss_fn,
                inverse_link_fn=inverse_link_fn)
            self._feature_columns = validate_feature_columns(
                linear_feature_columns=linear_feature_columns,
                dnn_feature_columns=deep_feature_columns)

            return _deepx_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                linear_feature_columns=linear_feature_columns,
                linear_optimizer=linear_optimizer,
                deep_feature_columns=deep_feature_columns,
                deep_optimizer=deep_optimizer,
                selector_feature_columns=selector_feature_columns,
                model_slots=model_slots,
                extend_feature_mode=extend_feature_mode,
                input_layer_partitioner=input_layer_partitioner,
                config=config,
                linear_sparse_combiner=linear_sparse_combiner,
                use_seperated_logits=use_seperated_logits,
                use_weighted_logits=use_weighted_logits,
                extra_options=extra_options)

        super(DeepXRegressor, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            warm_start_from=warm_start_from)
