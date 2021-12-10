#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


WEIGHT_COL = '_weight_column'


# 解析 assembler op 输出的 tensor，返回 feature dict
# is_dssm: 是否是 DSSM 粗排模式
def _parse_feature_dict(scheme, features, is_batch, is_dssm=False):
    feature_dict = {}
    idx = 0

    for item in scheme['feature_columns']:
        name = item['alias']
        if name in feature_dict:
            raise ValueError("feature name '{}' duplicated".format(name))
        tt = item['type_str']
        cid = item['cid_str']
        width = item['width']
        if is_batch:
            if is_dssm and (cid == 'user' or cid == 'ctx'):
                f = features[0:1, idx:idx + width]
            else:
                f = features[:, idx:idx + width]
        else:
            f = features[idx:idx + width]

        if tt == 'int' or tt == 'string' or tt == 'string_list':
            f = tf.to_int32(f)
            feature_dict[name] = f
        elif tt == 'weighted_string_list':
            assert width % 2 == 0  # 必须是偶数
            w = int(width / 2)
            if is_batch:
                keys = tf.to_int32(f[:, :w])
                weights = tf.to_float(f[:, w:])
            else:
                keys = tf.to_int32(f[:w])
                weights = tf.to_float(f[w:])
            feature_dict[name] = keys
            feature_dict[name + '.weight'] = weights
        elif tt == 'triple_list':
            assert width % 2 == 0
            w = int(width / 2)
            width_pos = item['width_pos']
            width_neg = item['width_neg']
            assert w == width_pos + width_neg
            if is_batch:
                pos_keys = tf.to_int32(f[:, :width_pos])
                pos_weights = tf.to_float(f[:, w:w+width_pos])
                neg_keys = tf.to_int32(f[:, width_pos:w])
                neg_weights = tf.to_float(f[:, w+width_pos:])
            else:
                pos_keys = tf.to_int32(f[:width_pos])
                pos_weights = tf.to_float(f[w:w+width_pos])
                neg_keys = tf.to_int32(f[width_pos:w])
                neg_weights = tf.to_float(f[w+width_pos:])
            feature_dict[name + '.pos'] = pos_keys
            feature_dict[name + '.pos.weight'] = pos_weights
            feature_dict[name + '.neg'] = neg_keys
            feature_dict[name + '.neg.weight'] = neg_weights
        elif tt == 'float' or tt == 'float_list':
            feature_dict[name] = f
        else:
            raise ValueError("Unknown feature type '{}'".format(tt))
        idx += width
    tf.logging.info('feature_dict = {}'.format(feature_dict))
    return feature_dict


def _parse_line(line, opts, scheme, is_predict, use_negative_sampling=False):
    assembler_ops = tf.load_op_library(opts.assembler_ops_path)
    if use_negative_sampling:
        feature, label, weight = assembler_ops.assembler_with_negative_sampling(
            input=line,
            conf_path=opts.conf_path,
            nce_items_path=opts.nce_items_path,
            nce_count=opts.nce_count,
            nce_items_min_count=opts.nce_items_min_count,
            nce_items_top_k=opts.nce_items_top_k,
            use_lua_sampler=opts.use_lua_sampler,
            lua_sampler_script=opts.lua_sampler_script)
        feature_dict = _parse_feature_dict(scheme, feature, is_batch=True)
    else:
        feature, label, weight = assembler_ops.assembler(
            input=line,
            is_predict=is_predict,
            conf_path=opts.conf_path,
            use_lua_sampler=opts.use_lua_sampler,
            lua_sampler_script=opts.lua_sampler_script)
        feature_dict = _parse_feature_dict(scheme, feature, is_batch=True)

    feature_dict[WEIGHT_COL] = weight
    if is_predict:
        assert not use_negative_sampling
        return feature_dict
    else:
        return feature_dict, label


def input_fn(opts, filenames, is_eval, is_predict, scheme, epoch=1,
             use_negative_sampling=False):
    batch_size = opts.eval_batch_size if is_eval else opts.batch_size

    def build_input_fn():
        if opts.read_buffer_size_mb is None:
            buffer_size = None
        else:
            buffer_size = 1024*1024*opts.read_buffer_size_mb
            buffer_size = max(1, buffer_size)

        if opts.use_spark_fuel:
            import sparkfuel as sf
            num_workers = sf.get_tf_num_workers()
            job_type, task_id = sf.get_tf_identity()
            ds = tf.data.Dataset.from_tensor_slices(filenames)
            task_id = task_id if job_type == 'chief' else task_id + 1
            ds = ds.shard(num_workers, task_id)
            ds = ds.flat_map(
                lambda filename: tf.data.TextLineDataset(
                    filename,
                    buffer_size=buffer_size,
                    compression_type=opts.compression_type))
        else:
            ds = tf.data.TextLineDataset(
                filenames,
                buffer_size=buffer_size,
                compression_type=opts.compression_type,
                num_parallel_reads=opts.num_parallel_reads)

        ds = ds.map(
            lambda line: _parse_line(
                line, opts, scheme, is_predict,
                use_negative_sampling=use_negative_sampling),
            num_parallel_calls=opts.map_num_parallel_calls)

        if is_predict:
            ds = ds.flat_map(
                lambda feature_dict:
                tf.data.Dataset.from_tensor_slices(feature_dict))
        else:
            ds = ds.flat_map(
                lambda feature_dict, label:
                tf.data.Dataset.from_tensor_slices((feature_dict, label)))

        if opts.shuffle_batch and not is_eval and not is_predict:
            ds = ds.shuffle(buffer_size=opts.shuffle_size)
        if not is_eval and not is_predict:
            ds = ds.repeat(epoch)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(opts.prefetch_size)
        return ds

    return build_input_fn


def build_serving_input_fn(opts, scheme):
    assembler_ops = tf.load_op_library(opts.assembler_ops_path)
    serialized = assembler_ops.assembler_serialize(conf_path=opts.conf_path)
    with tf.Session() as sess:
        serialized = sess.run(serialized)

    def fixed_string_feature(shape):
        return tf.FixedLenFeature(shape=shape, dtype=tf.string)

    def var_string_feature():
        return tf.VarLenFeature(dtype=tf.string)

    def fixed_int64_feature(shape):
        return tf.FixedLenFeature(shape=shape, dtype=tf.int64)

    def var_int64_feature():
        return tf.VarLenFeature(dtype=tf.int64)

    def serving_input_receiver_fn():
        is_recall = (opts.run_mode == 'easyrecall')
        feature_spec = {
            'user_feature': fixed_string_feature([1]),
            'item_features': var_string_feature(),
        }
        serialized_tf_example = tf.placeholder(
            dtype=tf.string, shape=[None], name='input_example_tensor')
        receiver_tensors = {'inputs': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)
        if is_recall:
            output = assembler_ops.assembler_dssm_serving(
                user_feature=features['user_feature'],
                serialized=serialized)
        else:
            features['item_features'] = tf.sparse.to_dense(
                features['item_features'], default_value='')
            output = assembler_ops.assembler_serving(
                user_feature=features['user_feature'],
                item_features=features['item_features'],
                serialized=serialized)
        feature_dict = _parse_feature_dict(scheme, output, is_batch=True)

        return tf.estimator.export.ServingInputReceiver(
            feature_dict, receiver_tensors)

    return serving_input_receiver_fn
