#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys

"""
根据 json 配置文件，生成 wide&deep 模型对应的 columns
"""

_STR_TYPE = (str, )
if sys.version_info.major == 2:
    _STR_TYPE = (str, unicode)


class feature_column_attention_columns(object):
    def __init__(self,
                 attention_feature_list,
                 attention_query,
                 dimension,
                 combiner,   # TODO(zhezhaoxu) use it
                 attention_type,
                 attention_args,
                 shared_embedding):
        """
          attention_feature_list: tuple 列表，每个 tuple 包含 2 个元素，(特征名, 特征大小), 这里放的是浏览历史一类的变长特征
          attention_query：tuple，(特征名，特征大小)，待排序的物品特征名
          dimension：embedding 唯度
          attention_type: attention类型，din
          attention_args: python dict 类型，attention参数
          shared_embedding：是否共享embedding, 要求所有特征词典是一致的
        """
        self.attention_feature_list = attention_feature_list
        self.attention_query = attention_query
        self.dimension = dimension
        self.attention_type = attention_type
        self.attention_args = attention_args
        self.shared_embedding = shared_embedding


class Transform(object):
    def __init__(self, config, scheme):
        """
          config: python dict
          scheme: python dict
        """

        # print("scheme = '{}'".format(scheme))

        self.config = config
        # 获取 column 的一些信息，例如 string column 的词典大小，oov size 等,
        self.scheme = scheme
        self.scheme_map = {}   # column名到对应 column scheme 的映射
        self.column_map = {}   # 保存已经生成的 column 名到 column 对象的映射
        for item in self.scheme['feature_columns']:
            if item['type_str'] == 'triple_list':
                name = item['alias'] + '.pos'
                self.column_map[name] = name
                self.scheme_map[name] = item

                name = item['alias'] + '.neg'
                self.column_map[name] = name
                self.scheme_map[name] = item
            else:
                name = item['alias']
                self.column_map[name] = name
                self.scheme_map[name] = item

        # 利用 tf feature_column api 执行特征转换
        self._transform()

    @property
    def group_columns(self):
        return self._group_columns

    def _transform(self):
        self._group_columns = {}

        # 遍历配置文件中的所有 transform 操作
        for transform_item in self.config['transform']:
            output = transform_item['output']  # 该操作的输出 column 的名字
            if output in self.column_map:
                # column 不允许重名
                raise ValueError("Duplicated feature name '{}'".format(output))

            # 定义支持的转换操作
            transform_fn_map = {
                'numeric': self._numeric_transform,
                'bucketized': self._bucketized_transform,
                'embedding': self._embedding_transform,
                'shared_embedding': self._shared_embedding_transform,
                'pretrained_embedding': self._pretrained_embedding_transform,
                'indicator': self._indicator_transform,
                'categorical_identity': self._categorical_identity_transform,
                'sequence_categorical_identity': self._sequence_categorical_identity_transform,
                'cross': self._cross_transform,
                'weighted_categorical': self._weighted_categorical_transform,
                'attention': self._attention_transform,
            }

            op = transform_item['op']
            if op not in transform_fn_map:
                raise ValueError("Unknown op '{}'".format(op))
            # 根据 op 调用对应的转换函数
            column = transform_fn_map[op](transform_item)
            # 输出 column 保存在 map 中，后面的操作可以用前面生成的 column
            self.column_map[output] = column
            places = transform_item['places']

            if isinstance(column, list):
                column_list = column
                places_list = places
                assert len(column_list) == len(places_list), \
                    "add_shared_embedding columns and places size must be the same"
                for item in places_list:
                    assert isinstance(item, list), \
                        "add_shared_embedding places' element must be list type"
            else:
                column_list = [column]
                places_list = [places]

            for column, places in zip(column_list, places_list):
                for place in places:
                    if place not in self._group_columns:
                        self._group_columns[place] = []
                    self._group_columns[place].append(column)

        for place in self._group_columns:
            tf.logging.info("group {} size = {}".format(
                place, len(self._group_columns[place])))

    # 从 json 的转换元素中获取输入特征列的名字，输入特征可能有多列，例如 cross
    # 或者 weighted_categorical 操作
    def _get_feature(self, transform_item):
        feature_name = transform_item['input']
        if isinstance(feature_name, list):
            # cross column, shared embedding column, weighted categorical column
            columns = []
            for name in feature_name:
                if name not in self.column_map:
                    raise ValueError(
                        "Unknown feature '{}'".format(name))
                columns.append(self.column_map[name])
            return columns
        else:
            if feature_name not in self.column_map:
                raise ValueError("Unknown feature '{}'".format(feature_name))
            return self.column_map[feature_name]

    # 支持归一化
    def _numeric_transform(self, transform_item):
        feature = self._get_feature(transform_item)
        assert isinstance(feature, _STR_TYPE)
        normalizer_fn = None

        def norm_normalizer_fn(substract, denominator, exp):
            return lambda x: ((tf.to_float(x) - substract) / denominator) ** exp

        def log_normalizer_fn(substract, denominator, exp):
            return lambda x: ((tf.log(tf.to_float(x) + 1.0) -
                              substract) / denominator) ** exp

        normalizer_fn_map = {
            "norm": norm_normalizer_fn,
            "log-norm": log_normalizer_fn
        }
        if "normalizer_fn" in transform_item:
            fn_name = transform_item['normalizer_fn']
            substract = transform_item['substract']
            denominator = transform_item['denominator']
            exp = transform_item['exp']
            fn_builder = normalizer_fn_map[fn_name]
            normalizer_fn = fn_builder(substract, denominator, exp)

        return tf.feature_column.numeric_column(
            feature, normalizer_fn=normalizer_fn)

    def _bucketized_transform(self, transform_item):
        feature = self._get_feature(transform_item)
        boundaries = transform_item['boundaries']
        return tf.feature_column.bucketized_column(feature, boundaries)

    def _embedding_transform(self, transform_item):
        feature = self._get_feature(transform_item)
        dimension = transform_item['dimension']
        combiner = transform_item['combiner']
        return tf.feature_column.embedding_column(
            feature, dimension, combiner=combiner)

    def _shared_embedding_transform(self, transform_item):
        columns = self._get_feature(transform_item)
        dimension = transform_item['dimension']
        combiner = transform_item['combiner']
        return tf.feature_column.shared_embedding_columns(
            columns, dimension, combiner=combiner)

    # 加载预训练的 embedding
    def _pretrained_embedding_transform(self, transform_item):
        feature = self._get_feature(transform_item)
        dimension = transform_item['dimension']
        combiner = transform_item['combiner']
        trainable = bool(transform_item['trainable'])
        pretrained_embedding_file = transform_item['pretrained_embedding_file']
        embed_initializer = self._embed_initializer_from_file(
            pretrained_embedding_file)
        if isinstance(feature, _STR_TYPE):
            size = self.scheme_map[feature]['size']
            tf.logging.info("pretrained embedding column '{}', size = {}"
                            .format(self.scheme_map[feature]['alias'], size))
            vocab = range(size)
            feature = tf.feature_column.categorical_column_with_vocabulary_list(
                feature,
                vocab,
                default_value=-1)
        return tf.feature_column.embedding_column(
            feature, dimension, initializer=embed_initializer,
            trainable=trainable,
            combiner=combiner)

    # 可以用于 deep 层的稠密 one-hot 向量
    def _indicator_transform(self, transform_item):
        feature = self._get_feature(transform_item)
        return tf.feature_column.indicator_column(feature)

    # 由于 assembler op 已经将所有 string 类型转成了 float 类型，所以只需要做
    # identity 操作，不需要再提供 dict
    def _categorical_identity_transform(self, transform_item):
        feature = self._get_feature(transform_item)
        tf.logging.info("categorical identity column '{}'"
                        .format(self.scheme_map[feature]['alias']))
        assert isinstance(feature, _STR_TYPE)
        num_buckets = transform_item['num_buckets']
        if num_buckets < 0:
            num_buckets = self.scheme_map[feature]['size']
            tf.logging.info("categorical identity column '{}', size = {}"
                            .format(self.scheme_map[feature]['alias'], num_buckets))
        return tf.feature_column.categorical_column_with_vocabulary_list(
            feature,
            range(num_buckets),
            default_value=-1)

    def _sequence_categorical_identity_transform(self, transform_item):
        feature = self._get_feature(transform_item)
        tf.logging.info("sequence categorical identity column '{}'"
                        .format(self.scheme_map[feature]['alias']))
        assert isinstance(feature, _STR_TYPE)
        num_buckets = transform_item['num_buckets']
        if num_buckets < 0:
            num_buckets = self.scheme_map[feature]['size']
            tf.logging.info("sequence categorical identity column '{}', size = {}"
                            .format(self.scheme_map[feature]['alias'], num_buckets))
        return tf.contrib.feature_column.sequence_categorical_column_with_vocabulary_list(
            feature,
            range(num_buckets),
            default_value=-1)

    def _cross_transform(self, transform_item):
        features = self._get_feature(transform_item)
        hash_bucket_size = transform_item['hash_bucket_size']
        return tf.feature_column.crossed_column(features, hash_bucket_size)

    def _weighted_categorical_transform(self, transform_item):
        feature = self._get_feature(transform_item)
        weight_column_name = transform_item['weight_column_name']
        assert isinstance(feature, _STR_TYPE)
        num_buckets = self.scheme_map[feature]['size']
        tf.logging.info("weighted categorical column '{}', size = {}"
                        .format(self.scheme_map[feature]['alias'], num_buckets))
        feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
            feature,
            range(num_buckets),
            default_value=-1)
        return tf.feature_column.weighted_categorical_column(
            feature_column, weight_column_name)

    def _attention_transform(self, transform_item):
        features = transform_item['input']
        query = transform_item['attention_query']
        dimension = transform_item['dimension']
        combiner = transform_item['combiner']
        attention_type = transform_item['attention_type']
        attention_args = transform_item['attention_args']
        shared_embedding = transform_item['shared_embedding']

        attention_feature_list = []
        for feature in features:
            attention_feature_list.append((feature, self.scheme_map[feature]['size']))
        attention_query = (query, self.scheme_map[query]['size'])

        if shared_embedding:
            # 检验词典是否一致, vocab
            check_vocab = True
            base_vocab = self.scheme_map[query]['vocab']
            for feature in features:
                test_vocab = self.scheme_map[feature]['vocab']
                if cmp(base_vocab, test_vocab) != 0:
                    check_vocab = False
                    break
            if not check_vocab:
                raise ValueError("Attention column error, use shared embedding"
                                 " but got different vocabs.")

        return feature_column_attention_columns(
            attention_feature_list,
            attention_query,
            dimension,
            combiner,
            attention_type,
            attention_args,
            shared_embedding)

    def _embed_initializer_from_file(self, filename):
        # TODO(zhezhaoxu) read from hdfs?

        with open(filename) as f:
            header = f.readline().strip().split()
            if len(header) != 2:
                raise ValueError("embedding file '{}' header format error.")
            cnt, dim = map(int, header)
        data = np.zeros([cnt, dim], dtype=np.float32)
        real_cnt = 0
        for index, line in enumerate(open(filename)):
            if index < 1:
                continue  # skip header
            if (index+1) % 100000 == 0:
                tf.logging.info("load {}w lines ...".format((index+1)/10000))
            line = line.strip()
            if not line:
                break
            tokens = line.split()
            if len(tokens) != dim + 1:
                raise ValueError("embedding file format error")
            features = map(float, tokens[1:])
            data[real_cnt, :] = features
            real_cnt += 1
        if cnt != real_cnt:
            raise ValueError('embedding file error, lines count not match')

        def initializer(shape=None, dtype=tf.float32, partition_info=None):
            return data

        return initializer
