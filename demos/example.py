#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import tensorflow as tf


if len(sys.argv) != 3:
    print("<Usage>: <assembler_ops_path> <conf>")
    sys.exit(-1)


assembler_ops_path = sys.argv[1]
conf_path = sys.argv[2]


def parse_scheme(conf_path, ops_path):
    assembler_ops = tf.load_op_library(ops_path)
    with tf.Session() as sess:
        output = assembler_ops.assembler_scheme(conf_path=conf_path)
        scheme = sess.run(output)
    scheme = json.loads(scheme)

    return scheme


def parse_feature_dict(scheme, features, is_batch, is_dssm=False):
    feature_dict = {}
    idx = 0

    for item in scheme['feature_columns']:
        name = item['iname']
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


sess = tf.Session()
assembler_ops = tf.load_op_library(assembler_ops_path)
serialized = assembler_ops.assembler_serialize(conf_path=conf_path)
serialized = sess.run(serialized)
# 101 i_kd_union_tags
# 102 u_kd_portrait_tags
# 103 u_kd_behavior_play_history_longv
user_feature = ["""101|tag1,tag10,tag2,tag10,tag3\t102|tag1:0.1,tag10:1.0,tag2:0.2,tag10:1.0,tag3:0.3\t103|r1:10:10,r2:1:10,r3:1:10"""]
user_feature = assembler_ops.assembler_dssm_serving(
    user_feature=user_feature,
    serialized=serialized)

scheme = parse_scheme(conf_path, assembler_ops_path)
feature_dict = parse_feature_dict(scheme, user_feature, is_batch=True)

print('scheme = ')
print(scheme)

print('feature_dict = ')
for key in sorted(feature_dict.keys()):
    print("{}: {}".format(key, feature_dict[key]))
    print(sess.run(feature_dict[key]))
