#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
import numpy as np
from math import sqrt

from common.dice import dice
from common.dice import parametric_relu
from common.swish import swish
from common.mish import mish
from common.gelu import gelu
from common import hook


def get_file_content(filename, use_spark_fuel, spark):
    if use_spark_fuel:
        lines = spark.sparkContext.textFile(filename).collect()
        data = '\n'.join(lines)
    else:
        with open(filename) as f:
            data = f.read()

    return data


def hdfs_remove_dir(spark, path):
    from sparkfuel.common.hdfs import Hdfs
    sc = spark.sparkContext
    hdfs = Hdfs(sc)
    if hdfs.exists(path):
        return hdfs.remove(path)


def hdfs_create_file(spark, path):
    from sparkfuel.common.hdfs import Hdfs
    sc = spark.sparkContext
    hdfs = Hdfs(sc)
    return hdfs.create_new_file(path)


def add_hidden_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag, tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)


def add_hidden_layers(inputs,
                      hidden_units,
                      activation_fn,
                      dropout,
                      is_training,
                      batch_norm,
                      layer_norm,
                      use_resnet,
                      use_densenet,
                      gaussian_dropout=None,
                      l2_regularizer=0.0,
                      scope='dnn',
                      last_layer_direct=False):
    assert not (batch_norm and layer_norm), \
        'batch_norm and layer_norm must not all be True'

    assert not (use_resnet and use_densenet), \
        'use_resnet and use_densenet must not all be True'

    assert not (use_resnet or use_densenet) or len(set(hidden_units)) <= 1, \
        'hidden_units should contains all the same number for resnet or densenet'

    with tf.variable_scope(scope):
        net = inputs
        highways = []

        if l2_regularizer > 0.0:
            l2_regularizer = tf.contrib.layers.l2_regularizer(
                scale=l2_regularizer)
        else:
            l2_regularizer = None

        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('h_%d' % layer_id) as hidden_layer_scope:
                if layer_id == len(hidden_units) - 1 and last_layer_direct:
                    activation_fn = None
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=None,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    kernel_regularizer=l2_regularizer,
                    bias_regularizer=l2_regularizer,
                    name=hidden_layer_scope)

                if use_resnet and len(highways) > 0:
                    net += highways[-1]

                if use_densenet and len(highways) > 0:
                    net += sum(highways)

                if activation_fn:
                    net = activation_fn(net)

                if dropout is not None and dropout > 0.0:
                    net = tf.layers.dropout(
                        net,
                        rate=dropout,
                        training=is_training)

                if gaussian_dropout is not None:
                    net = layer_gaussian_dropout(
                        net,
                        rate=gaussian_dropout,
                        training=is_training)

                if batch_norm:
                    # The default momentum 0.99 actually crashes on certain
                    # problem, so here we use 0.999, which is the default of
                    # tf.contrib.layers.batch_norm.
                    net = tf.layers.batch_normalization(
                        net,
                        momentum=0.999,
                        training=is_training,
                        trainable=True)

                if layer_norm:
                    net = tf.contrib.layers.layer_norm(net)

                highways.append(net)
                add_hidden_layer_summary(net, hidden_layer_scope.name)
    return net


def mask_padding_embedding_lookup(embeddings, dim, inputs, padding_id):
    """ mask padding tf.nn.embedding_lookup.

    ref(@ay27): https://github.com/tensorflow/tensorflow/issues/2373
    """

    mask_padding_zero_op = tf.scatter_update(
        embeddings, padding_id, tf.zeros([dim], dtype=tf.float32),
        name="mask_padding_zero_op")
    with tf.control_dependencies([mask_padding_zero_op]):
        output = tf.gather(embeddings, inputs)
    return output


def create_embedding(name, num_buckets, dimension):
    with tf.variable_scope('embeddings'):
        embeds = tf.get_variable(
            name,
            shape=[num_buckets, dimension],
            initializer=tf.initializers.truncated_normal(stddev=sqrt(dimension)))
    return embeds


def validate_feature_columns(linear_feature_columns, dnn_feature_columns):
    """Validates feature columns DNNLinearCombinedRegressor."""
    linear_feature_columns = linear_feature_columns or []
    dnn_feature_columns = dnn_feature_columns or []
    feature_columns = (list(linear_feature_columns) + list(dnn_feature_columns))
    if not feature_columns:
        raise ValueError('Either linear_feature_columns or dnn_feature_columns '
                         'must be defined.')
    return feature_columns


def check_no_sync_replicas_optimizer(optimizer):
    if isinstance(optimizer, tf.train.SyncReplicasOptimizer):
        raise ValueError(
            'SyncReplicasOptimizer does not support multi optimizers case. '
            'Therefore, it is not supported in DNNLinearCombined model. '
            'If you want to use this optimizer, please use either DNN or Linear '
            'model.')


def fc(x, units, activation=None, name=None):
    """fully connected layer
    """
    y = tf.layers.dense(x,
                        units=units,
                        activation=activation,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        name=name)
    return y


def check_arg(args, key, dtypes):
    if args is None or key not in args or not isinstance(args[key], dtypes):
        raise ValueError("args error, args = {}".format(args))


def project(feature_vectors, project_dim, name='project'):
    with tf.variable_scope(name):
        project_vectors = []
        for v in feature_vectors:
            v = fc(v, project_dim)
            project_vectors.append(v)

        return project_vectors


def check_feature_dims(feature_vectors):
    expected_dim = None
    for v in feature_vectors:
        dim = v.shape[1].value
        if expected_dim is None:
            expected_dim = dim
        if expected_dim != dim:
            raise ValueError('feature_vectors dims should all be the same')


def pairwise_dot(feature_vectors, name='pairwise_dot'):
    """Pairwise dot
     feature_vectors: list of N shape [B, D] tensors

    Return:
     Tensor of shape [B, N*(N-1)/2, D]
    """

    check_feature_dims(feature_vectors)

    with tf.variable_scope(name):
        rows = []
        cols = []
        N = len(feature_vectors)
        for i in range(N-1):
            for j in range(i+1, N):
                rows.append(i)
                cols.append(j)

        expand_vectors = []
        for v in feature_vectors:
            expand_vectors.append(tf.expand_dims(v, axis=1))

        p = tf.concat([expand_vectors[idx] for idx in rows], axis=1)
        q = tf.concat([expand_vectors[idx] for idx in cols], axis=1)
        y = p * q   # [B, Num_Interactions, D]
        return y


def pairwise_dot_unorderd(feature_vectors, name='pairwise_dot_unorderd'):
    """Pairwise dot
     feature_vectors: list of N shape [B, D] tensors

    Return:
     Tensor of shape [B, N*N, D]
    """
    with tf.variable_scope(name):
        N = len(feature_vectors)
        x = tf.stack(feature_vectors, axis=1)   # [B, N, D]
        assert x.shape[1].value == N
        D = x.shape[2].value

        p = tf.expand_dims(x, axis=1)   # [B, 1, N, D]
        q = tf.expand_dims(x, axis=2)   # [B, N, 1, D]
        y = p * q   # [B, N, N, D]
        y = tf.reshape(y, [-1, N*N, D])  # [B, N*N, D]

        return y


def pairwise_kernel_dot(feature_vectors, name='pairwise_kernel_dot'):
    """Pairwise Kernel dot
     Use different kernel matrix for each featrue pair.

     feature_vectors: list of 2-D tensors, size N, and the second dimentions can
     be different.

    Return:
     Tensor of shape [B, N*N]
    """
    with tf.variable_scope(name):
        interactions = []
        n = len(feature_vectors)
        # Use order-aware kernel dot
        for i in range(n):
            for j in range(n):
                di = feature_vectors[i].shape[1].value
                dj = feature_vectors[j].shape[1].value
                name = 'pairwise_kernel_dot_{}_{}'.format(i, j)
                U = tf.get_variable(name, [di, dj])
                y = tf.matmul(feature_vectors[i], U)  # [B, dj]
                y = y * feature_vectors[j]  # [B, dj]
                y = tf.reduce_sum(y, axis=-1, keepdims=True)  # [B, 1]
                interactions.append(y)
        y = tf.concat(interactions, axis=1)  # [B, N*N]
        return y


def pairwise_kernel_dot_aligned(feature_vectors, name='pairwise_kernel_dot_aligned'):
    """Pairwise Kernel dot
     feature_vectors: list of 2-D tensors of shape [B, D], size N.

     Use different kernel matrix for each featrue pair.

    TODO(zhezhao) Need test: We use a optimized implementation which utilizes
    best parallelization.

    Return:
     Tensor of shape [B, N*(N-1)/2, D]
    """

    with tf.variable_scope(name):
        L = len(feature_vectors)
        rows = []
        cols = []
        for i in range(L-1):
            for j in range(i+1, L):
                rows.append(i)
                cols.append(j)

        expanded_feature_vectors = []
        for v in feature_vectors:
            expanded_feature_vectors.append(tf.expand_dims(v, axis=1))

        # [B, N*(N-1)/2, D]
        p = tf.concat([expanded_feature_vectors[idx] for idx in rows], axis=1)
        p = tf.expand_dims(p, axis=2)  # [B, N*(N-1)/2, 1, D]

        # [B, N*(N-1)/2, D]
        q = tf.concat([expanded_feature_vectors[idx] for idx in cols], axis=1)

        M = p.shape[1].value
        D = p.shape[2].value
        kernels = tf.get_variable('kernels', [M, D, D])
        y = p * kernels   # [B, N*(N-1)/2, D, D]
        y = tf.reduce_sum(y, axis=-1)  # [B, N*(N-1)/2, D]
        y = y * q   # [B, N*(N-1)/2, D]

        return y


def interaction_aware_pairwise_dot(feature_vectors,
                                   kf,
                                   name='interaction_aware_pairwise_dot'):
    """Interaction-Aware Pairwise dot
    see formula (7) of "Interaction-aware Factorization Machines for Recommender Systems"

    Args
     feature_vectors: list of N shape [B, D] tensors
     kf: field dimention

    Return
     Tensor of shape [B, N*(N-1)/2, D]
    """

    check_feature_dims(feature_vectors)

    with tf.variable_scope(name):
        rows = []
        cols = []
        N = len(feature_vectors)
        for i in range(N-1):
            for j in range(i+1, N):
                rows.append(i)
                cols.append(j)

        expand_vectors = []
        for v in feature_vectors:
            expand_vectors.append(tf.expand_dims(v, axis=1))

        p = tf.concat([expand_vectors[idx] for idx in rows], axis=1)
        q = tf.concat([expand_vectors[idx] for idx in cols], axis=1)
        r = p * q   # [B, N*(N-1)/2, D]

        D = r.shape[-1].value   # feature dimention
        field_aspect = tf.get_variable('field_aspect', [N, kf])
        # maybe use [N*(N-1)/2, kf, D] is better?
        project_d = tf.get_variable('project_d', [kf, D])

        s = tf.gather(field_aspect, rows)   # [N*(N-1)/2, kf]
        t = tf.gather(field_aspect, cols)   # [N*(N-1)/2, kf]
        u = s * t   # [N*(N-1)/2, kf]
        u = tf.matmul(u, project_d)  # [N*(N-1)/2, D]

        y = r * u  # [B, N*(N-1)/2, D]

        return y


def get_feature_vectors(features, feature_columns, scope='feature_vectors'):

    with tf.variable_scope(scope):
        feature_vectors = []
        for c in feature_columns:
            with tf.variable_scope(c.name):
                v = tf.feature_column.input_layer(features, c)
                feature_vectors.append(v)
        return feature_vectors


def k_max_pooling(x, k=1, axis=-1, name='k_max_pooling'):
    """K Max pooling that selects the k biggest value along the specific axis.

    https://github.com/shenweichen/DeepCTR/blob/master/deepctr/layers/sequence.py
    https://github.com/keras-team/keras/issues/373
    """

    with tf.variable_scope(name):
        assert k >= 1, 'k must >= 1'

        ndim = len(x.shape)
        perm = range(ndim)
        perm[-1], perm[axis] = perm[axis], perm[-1]
        x = tf.transpose(x, perm)
        top_k = tf.nn.top_k(x, k=k, sorted=True)[0]
        y = tf.transpose(top_k, perm)

        return y


def get_activation_fn(activation_fn, leaky_relu_alpha, swish_beta):

    activation_fn_map = {
        'sigmoid': tf.nn.sigmoid,
        'tanh': tf.nn.tanh,
        'relu': tf.nn.relu,
        'relu6': tf.nn.relu6,
        'leaky_relu': lambda x: tf.nn.leaky_relu(x, alpha=leaky_relu_alpha),
        'elu': tf.nn.elu,
        'crelu': tf.nn.crelu,
        'selu': tf.nn.selu,   # 需要仔细研究下如何正确使用
        'gelu': gelu,
        'dice': dice,
        'prelu': parametric_relu,
        'identity': tf.identity,
        'swish': lambda x: swish(x, beta=swish_beta),
        'mish': mish,
    }

    activation_fn = activation_fn_map[activation_fn]
    return activation_fn


def warmup_learning_rate(init_lr, warmup_rate, global_step, total_steps):
    assert total_steps > 0, 'total_steps must be larger than 0'
    assert warmup_rate > 0, 'warmup_rate must be larger than 0'

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        total_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)
    global_steps_int = tf.cast(global_step, tf.int32)
    num_warmup_steps = int(warmup_rate * total_steps)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_lr = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_lr)

    return learning_rate


def create_profile_hooks(save_steps, model_dir):
    """Create profile hooks."""

    meta_hook = hook.MetadataHook(
        save_steps=save_steps, output_dir=model_dir)
    profile_hook = tf.train.ProfilerHook(
        save_steps=save_steps,
        output_dir=model_dir,
        show_dataflow=True,
        show_memory=True)
    return meta_hook, profile_hook


# scheme 是由 assembler op 生成的一个 json 对象
def parse_scheme(conf_path, ops_path, use_spark_fuel):
    assembler_ops = tf.load_op_library(ops_path)
    with tf.Session() as sess:
        output = assembler_ops.assembler_scheme(conf_path=conf_path)
        scheme = sess.run(output)
    scheme = json.loads(scheme)

    return scheme

def layer_gaussian_dropout(x, rate, training):
    """Gaussian dropout

      x: 输入tensor
      rate: dropout概率，标准差为 sqrt(rate / (1 - rate))
      training: 是否是训练阶段

     Return:
      x * N(1, sqrt(rate / (1 - rate)))
    """

    if training and 0 < rate < 1.0:
        stddev = np.sqrt(rate / (1.0 - rate))
        return tf.random.normal(tf.shape(x), 1, stddev) * x

    return x
