#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json


class AssemblerType(object):
    INT = 'int'
    FLOAT = 'float'
    STRING = 'string'
    STRING_LIST = 'string_list'
    FLOAT_LIST = 'float_list'
    WEIGHTED_STRING_LIST = 'weighted_string_list'
    TRIPLE_LIST = 'triple_list'

    ALL = ('int', 'float', 'string', 'string_list', 'float_list',
           'weighted_string_list', 'triple_list')


class AssemblerBlock(object):
    def __init__(self):
        self._iname = None
        self._alias = None
        self._type = None
        self._default = None
        self._min_count = None
        self._top_k = None
        self._oov_buckets = None
        self._default_key = None
        self._min_weight = None
        self._dict_file = None
        self._width = None
        self._scan_from = None
        self._reverse = None
        self._width_pos = None
        self._width_neg = None
        self._filter_type = None
        self._params = None
        self._force_add = None

    @property
    def iname(self):
        if self._iname is None:
            raise ValueError("'iname' should not be None.")
        return self._iname

    @iname.setter
    def iname(self, value):
        if not isinstance(value, str):
            raise ValueError("'iname' must be of str type.")
        self._iname = value

    @property
    def alias(self):
        return self._alias

    @alias.setter
    def alias(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("'alias' must be of str type.")
        self._alias = value

    @property
    def type(self):
        if self._type is None:
            raise ValueError("'type' should not be None.")
        return self._type

    @type.setter
    def type(self, value):
        if not isinstance(value, str):
            raise ValueError("'type' must be of str type.")
        if value not in AssemblerType.ALL:
            raise ValueError("'type' must be one of 'AssemblerType.INT', "
                             "'AssemblerType.FLOAT', 'AssemblerType.STRING', "
                             "'AssemblerType.STRING_LIST', "
                             "'AssemblerType.FLOAT_LIST', "
                             "'AssemblerType.WEIGHTED_STRING_LIST'")
        self._type = value

    @property
    def default(self):
        if not self._is_string_type() and self._default is None:
            raise ValueError("'default' should not be None.")
        return self._default

    @default.setter
    def default(self, value):
        if self._is_string_type():
            raise ValueError("'default' should not be set.")
        if not isinstance(value, (int, float)):
            raise ValueError("'default' should be of int or float type.")
        self._default = value

    @property
    def min_count(self):
        if self._is_string_type() and self._min_count is None:
            raise ValueError("'min_count' should not be None.")
        return self._min_count

    @min_count.setter
    def min_count(self, value):
        if not self._is_string_type():
            raise ValueError("'min_count' should not be set in this type.")
        if not isinstance(value, int):
            raise ValueError("'min_count' must be of int type.")
        self._min_count = value

    @property
    def min_weight(self):
        if self.type == AssemblerType.WEIGHTED_STRING_LIST and self._min_weight is None:
            raise ValueError("'min_weight' should not be None.")
        return self._min_weight

    @min_weight.setter
    def min_weight(self, value):
        if self.type != AssemblerType.WEIGHTED_STRING_LIST:
            raise ValueError("'min_weight' should not be set in this type.")
        if not isinstance(value, float):
            raise ValueError("'min_weight' must be of float type.")
        self._min_weight = value

    @property
    def top_k(self):
        if self._is_string_type() and self._top_k is None:
            raise ValueError("'top_k' should not be None.")
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        if not self._is_string_type():
            raise ValueError("'top_k' should not be set in this type.")
        if not isinstance(value, int):
            raise ValueError("'top_k' must be of int type or None.")
        self._top_k = value

    @property
    def oov_buckets(self):
        if self._is_string_type() and self._oov_buckets is None:
            raise ValueError("'oov_buckets' should not be None.")
        return self._oov_buckets

    @oov_buckets.setter
    def oov_buckets(self, value):
        if not self._is_string_type():
            raise ValueError("'oov_buckets' should not be set in this type.")
        if not isinstance(value, int):
            raise ValueError("'oov_buckets' must be of int type.")
        self._oov_buckets = value

    @property
    def default_key(self):
        if self._is_string_type() and self._default_key is None:
            raise ValueError("'default_key' should not be None.")
        return self._default_key

    @default_key.setter
    def default_key(self, value):
        if not self._is_string_type():
            raise ValueError("'default_key' should not be set in this type.")
        if not isinstance(value, str):
            raise ValueError("'default_key' must be of string type.")
        self._default_key = value

    @property
    def dict_file(self):
        if self._is_string_type() and self._dict_file is None:
            raise ValueError("'dict_file' should not be None.")
        return self._dict_file

    @dict_file.setter
    def dict_file(self, value):
        if not self._is_string_type():
            raise ValueError("'dict_file' should not be set in this type.")
        if not isinstance(value, str):
            raise ValueError("'dict_file' must be of str type.")
        self._dict_file = value

    @property
    def width(self):
        if (self.type == AssemblerType.STRING_LIST
                or self.type == AssemblerType.FLOAT_LIST
                or self.type == AssemblerType.WEIGHTED_STRING_LIST) and self._width is None:
            raise ValueError("'width' should not be None.")
        return self._width

    @width.setter
    def width(self, value):
        if (self.type != AssemblerType.STRING_LIST
                and self.type != AssemblerType.FLOAT_LIST
                and self.type != AssemblerType.WEIGHTED_STRING_LIST):
            raise ValueError("'width' should not be set in this type.")
        if not isinstance(value, int):
            raise ValueError("'width' must be of int type.")
        self._width = value

    @property
    def scan_from(self):
        if (self.type == AssemblerType.STRING_LIST
                or self.type == AssemblerType.WEIGHTED_STRING_LIST
                or self.type == AssemblerType.FLOAT_LIST
                or self.type == AssemblerType.TRIPLE_LIST) \
                and self._scan_from is None:
            raise ValueError("'scan_from' should not be None.")
        return self._scan_from

    @scan_from.setter
    def scan_from(self, value):
        if (self.type != AssemblerType.STRING_LIST
                and self.type != AssemblerType.WEIGHTED_STRING_LIST
                and self.type != AssemblerType.FLOAT_LIST
                and self.type != AssemblerType.TRIPLE_LIST):
            raise ValueError("'scan_from' should not be set in this type.")
        if not isinstance(value, str):
            raise ValueError("'scan_from' must be of str type.")
        self._scan_from = value

    @property
    def reverse(self):
        if (self.type == AssemblerType.STRING_LIST
                or self.type == AssemblerType.WEIGHTED_STRING_LIST
                or self.type == AssemblerType.FLOAT_LIST
                or self.type == AssemblerType.TRIPLE_LIST) \
                and self._reverse is None:
            raise ValueError("'reverse' should not be None.")
        return self._reverse

    @reverse.setter
    def reverse(self, value):
        if (self.type != AssemblerType.STRING_LIST
                and self.type != AssemblerType.WEIGHTED_STRING_LIST
                and self.type != AssemblerType.FLOAT_LIST
                and self.type != AssemblerType.TRIPLE_LIST):
            raise ValueError("'reverse' should not be set in this type.")
        if not isinstance(value, bool):
            raise ValueError("'reverse' must be of bool type.")
        self._reverse = value

    @property
    def width_pos(self):
        if self.type == AssemblerType.TRIPLE_LIST and self._width_pos is None:
            raise ValueError("'width_pos' should not be None.")
        return self._width_pos

    @width_pos.setter
    def width_pos(self, value):
        if self.type != AssemblerType.TRIPLE_LIST:
            raise ValueError("'width_pos' should not be set in this type.")
        if not isinstance(value, int):
            raise ValueError("'width_pos' must be of int type.")
        self._width_pos = value

    @property
    def width_neg(self):
        if self.type == AssemblerType.TRIPLE_LIST and self._width_neg is None:
            raise ValueError("'width_neg' should not be None.")
        return self._width_neg

    @width_neg.setter
    def width_neg(self, value):
        if self.type != AssemblerType.TRIPLE_LIST:
            raise ValueError("'width_neg' should not be set in this type.")
        if not isinstance(value, int):
            raise ValueError("'width_neg' must be of int type.")
        self._width_neg = value

    @property
    def filter_type(self):
        if self.type == AssemblerType.TRIPLE_LIST and self._filter_type is None:
            raise ValueError("'filter_type' should not be None.")
        return self._filter_type

    @filter_type.setter
    def filter_type(self, value):
        if self.type != AssemblerType.TRIPLE_LIST:
            raise ValueError("'filter_type' should not be set in this type.")
        if not isinstance(value, str):
            raise ValueError("'filter_type' must be of int type.")
        self._filter_type = value

    @property
    def params(self):
        if self.type == AssemblerType.TRIPLE_LIST and self._params is None:
            raise ValueError("'params' should not be None.")
        return self._params

    @params.setter
    def params(self, value):
        if self.type != AssemblerType.TRIPLE_LIST:
            raise ValueError("'params' should not be set in this type.")
        if not isinstance(value, dict):
            raise ValueError("'params' must be of dict type.")
        self._params = value

    @property
    def force_add(self):
        if self._force_add is None:
            raise ValueError("'force_add' should not be None.")
        return self._force_add

    @force_add.setter
    def force_add(self, value):
        if not isinstance(value, bool):
            raise ValueError("'force_add' must be of bool type.")
        self._force_add = value

    def to_obj(self):
        obj = {}
        obj['iname'] = self.iname
        if self.alias is None:
            obj['alias'] = self.iname
        else:
            obj['alias'] = self.alias
        obj['type'] = self.type
        if self.default is not None:
            obj['default'] = self.default
        if self.min_count is not None:
            obj['min_count'] = self.min_count
        if self.top_k is not None:
            obj['top_k'] = self.top_k
        if self.oov_buckets is not None:
            obj['oov_buckets'] = self.oov_buckets
        if self.default_key is not None:
            obj['default'] = self.default_key
        if self.min_weight is not None:
            obj['min_weight'] = self.min_weight
        if self.dict_file is not None:
            obj['dict_file'] = self.dict_file
        if self.width is not None:
            obj['width'] = self.width
        if self.scan_from is not None:
            obj['scan_from'] = self.scan_from
        if self.reverse is not None:
            obj['reverse'] = self.reverse
        if self.width_pos is not None:
            obj['width_pos'] = self.width_pos
        if self.width_neg is not None:
            obj['width_neg'] = self.width_neg
        if self.filter_type is not None:
            obj['filter_type'] = self.filter_type
        if self.params is not None:
            obj['params'] = self.params
        if self.force_add is not None:
            obj['force_add'] = self.force_add

        return obj

    def _is_string_type(self):
        if self.type in (AssemblerType.STRING, AssemblerType.STRING_LIST,
                         AssemblerType.WEIGHTED_STRING_LIST,
                         AssemblerType.TRIPLE_LIST):
            return True
        return False


class Assembler(object):
    def __init__(self):
        self._blocks = []

    @property
    def blocks(self):
        return self._blocks

    def _add_block(self, obj, ftype):
        block = AssemblerBlock()
        block.iname = obj[0]
        block.type = ftype
        if block.type == AssemblerType.STRING:
            block.dict_file = obj[1]
            block.min_count = obj[2] if obj[2] is not None else 1
            block.top_k = obj[3] if obj[3] is not None else -1
            block.oov_buckets = obj[4] if obj[4] is not None else 0
            block.default_key = obj[5] if obj[5] is not None else ''
            block.force_add = obj[6]
            block.alias = obj[7]
        elif block.type == AssemblerType.STRING_LIST:
            block.dict_file = obj[1]
            block.min_count = obj[2] if obj[2] is not None else 1
            block.width = obj[3]
            block.top_k = obj[4] if obj[4] is not None else -1
            block.oov_buckets = obj[5] if obj[5] is not None else 0
            block.default_key = obj[6] if obj[6] is not None else ''
            block.scan_from = obj[7]
            block.reverse = obj[8]
            block.force_add = obj[9]
            block.alias = obj[10]
        elif block.type == AssemblerType.FLOAT_LIST:
            block.default = obj[1]
            block.width = obj[2]
            block.scan_from = obj[3]
            block.reverse = obj[4]
            block.force_add = obj[5]
            block.alias = obj[6]
        elif block.type == AssemblerType.WEIGHTED_STRING_LIST:
            block.dict_file = obj[1]
            block.min_count = obj[2] if obj[2] is not None else 1
            block.width = obj[3]
            block.top_k = obj[4] if obj[4] is not None else -1
            block.min_weight = obj[5]
            block.oov_buckets = obj[6] if obj[6] is not None else 0
            block.default_key = obj[7] if obj[7] is not None else ''
            block.scan_from = obj[8]
            block.reverse = obj[9]
            block.force_add = obj[10]
            block.alias = obj[11]
        elif block.type == AssemblerType.TRIPLE_LIST:
            block.dict_file = obj[1]
            block.min_count = obj[2] if obj[2] is not None else 1
            block.width_pos = obj[3]
            block.width_neg = obj[4]
            block.top_k = obj[5] if obj[5] is not None else -1
            block.oov_buckets = obj[6] if obj[6] is not None else 0
            block.default_key = obj[7] if obj[7] is not None else ''
            block.filter_type = obj[8]
            block.params = obj[9] if obj[9] is not None else {}
            block.scan_from = obj[10]
            block.reverse = obj[11]
            block.force_add = obj[12]
            block.alias = obj[13]
        elif block.type == AssemblerType.INT or block.type == AssemblerType.FLOAT:
            block.default = obj[1]
            block.force_add = obj[2]
            block.alias = obj[3]
        else:
            raise ValueError("type error")
        self._blocks.append(block.to_obj())

    def add_int(self, iname, default_value, force_add=False, alias=None):
        self._add_block((iname, default_value, force_add, alias), AssemblerType.INT)

    def add_float(self, iname, default_value, force_add=False, alias=None):
        self._add_block((iname, default_value, force_add, alias), AssemblerType.FLOAT)

    def add_string(self, iname, dict_file='', min_count=1, oov_buckets=None,
                   top_k=None, default_key=None, force_add=False, alias=None):
        self._add_block((iname, dict_file, min_count, top_k, oov_buckets,
                         default_key, force_add, alias), AssemblerType.STRING)

    def add_string_list(self, iname, dict_file='', min_count=1, width=None,
                        top_k=None, oov_buckets=None, default_key=None,
                        scan_from='tail', reverse=False, force_add=False, alias=None):
        self._add_block((iname, dict_file, min_count, width, top_k, oov_buckets,
                         default_key, scan_from, reverse, force_add, alias),
                        AssemblerType.STRING_LIST)

    def add_float_list(self, iname, default_value, width, scan_from='head',
                       reverse=False, force_add=False, alias=None):
        self._add_block((iname, default_value, width, scan_from,
                         reverse, force_add, alias),
                        AssemblerType.FLOAT_LIST)

    def add_weighted_string_list(self, iname, dict_file='', min_count=1,
                                 width=None, top_k=None, min_weight=0.0,
                                 oov_buckets=None, default_key=None,
                                 scan_from='head', reverse=False,
                                 force_add=False, alias=None):
        self._add_block((iname, dict_file, min_count, width, top_k, min_weight,
                         oov_buckets, default_key, scan_from,
                         reverse, force_add, alias),
                        AssemblerType.WEIGHTED_STRING_LIST)

    def add_triple_list(self, iname, filter_type='default_filter', params=None,
                        dict_file='', min_count=1,
                        width_pos=None, width_neg=None, top_k=None,
                        oov_buckets=None, default_key=None,
                        scan_from='head',
                        reverse=False, force_add=False, alias=None):
        self._add_block((iname, dict_file, min_count, width_pos, width_neg,
                         top_k, oov_buckets, default_key,
                         filter_type, params, scan_from, reverse, force_add, alias),
                        AssemblerType.TRIPLE_LIST)

    def to_obj(self, use_blocks=None):
        blocks = []
        for block in self.blocks:
            if use_blocks is not None and not block['force_add'] and (
                    block['alias'] not in use_blocks and
                    block['alias'] + '.pos' not in use_blocks and
                    block['alias'] + '.neg' not in use_blocks):
                continue
            blocks.append(block)
        return blocks


class TransformOpType(object):
    NUMERIC = 'numeric'
    BUCKETIZED = 'bucketized'
    EMBEDDING = 'embedding'
    SHARED_EMBEDDING = 'shared_embedding'
    PRETRAINED_EMBEDDING = 'pretrained_embedding'
    INDICATOR = 'indicator'
    CATEGORICAL_IDENTITY = 'categorical_identity'
    CROSS = 'cross'
    WEIGHTED_CATEGORICAL = 'weighted_categorical'
    ATTENTION = 'attention'
    SEQUENCE_CATEGORICAL_IDENTITY = 'sequence_categorical_identity'

    ALL = ('numeric', 'bucketized', 'embedding', 'shared_embedding',
           'pretrained_embedding', 'indicator', 'categorical_identity', 'cross',
           'weighted_categorical', 'attention', 'sequence_categorical_identity')


class TransformNormalizerType(object):
    NORM = 'norm'
    LOG_NORM = 'log-norm'

    ALL = ('norm', 'log-norm')


class TransformBlock(object):
    def __init__(self):
        self._input = None
        self._output = None
        self._op = None
        self._places = None
        self._boundaries = None
        self._dimension = None
        self._combiner = None
        self._trainable = None
        self._pretrained_embedding_file = None
        self._num_buckets = None
        self._hash_bucket_size = None
        self._normalizer_fn = None
        self._substract = None
        self._denominator = None
        self._attention_query = None
        self._attention_type = None
        self._attention_args = None
        self._shared_embedding = None
        self._weight_column_name = None
        self._slotid = None

    @property
    def input(self):
        if self._input is None:
            raise ValueError("'input' should not be None.")
        return self._input

    @input.setter
    def input(self, value):
        if not isinstance(value, (str, list, tuple, set)):
            raise ValueError("'input' must be of str or list-like type.")
        self._input = value

    @property
    def output(self):
        if self._output is None:
            raise ValueError("'output' should not be None.")
        return self._output

    @output.setter
    def output(self, value):
        if not isinstance(value, str):
            raise ValueError("'output' must be of str type.")
        self._output = value

    @property
    def op(self):
        if self._op is None:
            raise ValueError("'op' should not be None.")
        return self._op

    @op.setter
    def op(self, value):
        if not isinstance(value, str):
            raise ValueError("'op' must be of str type.")
        if value not in TransformOpType.ALL:
            raise ValueError("'op' must be one of TransformOpType.* element.")
        self._op = value

    @property
    def places(self):
        if self._places is None:
            raise ValueError("'places' should not be None.")
        return self._places

    @places.setter
    def places(self, value):
        if not isinstance(value, (list, set, tuple)):
            raise ValueError("'places' must be of list-like type.")
        self._places = []
        for item in value:
            self._places.append(item)

    @property
    def boundaries(self):
        if self.op == TransformOpType.BUCKETIZED and self._boundaries is None:
            raise ValueError("'boundaries' should not be None.")
        return self._boundaries

    @boundaries.setter
    def boundaries(self, value):
        if self.op != TransformOpType.BUCKETIZED:
            raise ValueError("'op' should not be set.")
        if not isinstance(value, (list, tuple)):
            raise ValueError("'boundaries' should be of list or tuple type.")
        self._boundaries = []
        for item in value:
            if not isinstance(item, (int, float)):
                raise ValueError(
                    "Element of 'boundaries' should be of int or float type.")
            self._boundaries.append(item)

    @property
    def dimension(self):
        if (self.op == TransformOpType.EMBEDDING
                or self.op == TransformOpType.PRETRAINED_EMBEDDING
                or self.op == TransformOpType.SHARED_EMBEDDING
                or self.op == TransformOpType.ATTENTION) \
                and self._dimension is None:
            raise ValueError("'dimension' should not be None.")
        return self._dimension

    @dimension.setter
    def dimension(self, value):
        if self.op != TransformOpType.EMBEDDING \
                and self.op != TransformOpType.PRETRAINED_EMBEDDING \
                and self.op != TransformOpType.SHARED_EMBEDDING \
                and self.op != TransformOpType.ATTENTION:
            raise ValueError("'dimension' should not be set.")
        if not isinstance(value, int):
            raise ValueError("'dimension' should be of int type.")
        self._dimension = value

    @property
    def combiner(self):
        if (self.op == TransformOpType.EMBEDDING
                or self.op == TransformOpType.PRETRAINED_EMBEDDING
                or self.op == TransformOpType.SHARED_EMBEDDING
                or self.op == TransformOpType.ATTENTION) \
                and self._combiner is None:
            raise ValueError("'combiner' should not be None.")
        return self._combiner

    @combiner.setter
    def combiner(self, value):
        if self.op != TransformOpType.EMBEDDING \
                and self.op != TransformOpType.PRETRAINED_EMBEDDING \
                and self.op != TransformOpType.SHARED_EMBEDDING \
                and self.op != TransformOpType.ATTENTION:
            raise ValueError("'combiner' should not be set.")

        if value not in ('mean', 'sqrtn', 'sum'):
            raise ValueError("'combiner' should be one of 'mean', 'sqrtn' or 'sum'.")
        self._combiner = value

    @property
    def trainable(self):
        if self.op == TransformOpType.PRETRAINED_EMBEDDING \
                and self._trainable is None:
            return False
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        if self.op != TransformOpType.PRETRAINED_EMBEDDING:
            raise ValueError("'trainable' should not be set.")
        if not isinstance(value, bool):
            raise ValueError("'trainable' should be of bool type.")
        self._trainable = value

    @property
    def pretrained_embedding_file(self):
        if self.op == TransformOpType.PRETRAINED_EMBEDDING \
                and self._pretrained_embedding_file is None:
            raise ValueError("'pretrained_embedding_file' should not be None.")
        return self._pretrained_embedding_file

    @pretrained_embedding_file.setter
    def pretrained_embedding_file(self, value):
        if self.op != TransformOpType.PRETRAINED_EMBEDDING:
            raise ValueError("'pretrained_embedding_file' should not be None.")
        if not isinstance(value, str):
            raise ValueError(
                "'pretrained_embedding_file' should be of str type.")
        self._pretrained_embedding_file = value

    @property
    def num_buckets(self):
        if (self.op == TransformOpType.CATEGORICAL_IDENTITY \
                or self.op == TransformOpType.SEQUENCE_CATEGORICAL_IDENTITY) \
                and self._num_buckets is None:
            raise ValueError("'num_buckets' should not be None.")
        return self._num_buckets

    @num_buckets.setter
    def num_buckets(self, value):
        if self.op != TransformOpType.CATEGORICAL_IDENTITY \
               and self.op != TransformOpType.SEQUENCE_CATEGORICAL_IDENTITY:
            raise ValueError("'num_buckets' should not be set.")
        if not isinstance(value, int):
            raise ValueError("'num_buckets' should be of int type.")
        self._num_buckets = value

    @property
    def hash_bucket_size(self):
        if self.op == TransformOpType.CROSS and self._hash_bucket_size is None:
            raise ValueError("'hash_bucket_size' should not be None.")
        return self._hash_bucket_size

    @hash_bucket_size.setter
    def hash_bucket_size(self, value):
        if self.op != TransformOpType.CROSS:
            raise ValueError("'hash_bucket_size' should not be set.")
        if not isinstance(value, int):
            raise ValueError("'hash_bucket_size' should be of int type.")
        self._hash_bucket_size = value

    @property
    def normalizer_fn(self):
        # This can be None
        return self._normalizer_fn

    @normalizer_fn.setter
    def normalizer_fn(self, value):
        if self.op != TransformOpType.NUMERIC:
            raise ValueError("'normalizer_fn' should not be set.")
        if not isinstance(value, str):
            raise ValueError("'normalizer_fn' should be of str type.")
        if value not in TransformNormalizerType.ALL:
            raise ValueError(
                "'normalizer_fn' should be one of TransformNormalizerType.*")
        self._normalizer_fn = value

    @property
    def substract(self):
        if self.normalizer_fn is not None:
            if self._substract is None:
                raise ValueError("'substract' should be not be None.")
        return self._substract

    @substract.setter
    def substract(self, value):
        if self.op != TransformOpType.NUMERIC:
            raise ValueError("'substract' should not be set.")
        if not isinstance(value, (int, float)):
            raise ValueError("'substract' should be of int or float type.")
        self._substract = value

    @property
    def denominator(self):
        if self.normalizer_fn is not None:
            if self._denominator is None:
                raise ValueError("'denominator' should be not be None.")
        return self._denominator

    @denominator.setter
    def denominator(self, value):
        if self.op != TransformOpType.NUMERIC:
            raise ValueError("'denominator' should not be set.")
        if not isinstance(value, (int, float)):
            raise ValueError("'denominator' should be of int or float type.")
        if value == 0.0:
            raise ValueError("'denominator' should not be set to 0.0.")
        self._denominator = value

    @property
    def exp(self):
        if self.normalizer_fn is not None:
            if self._exp is None:
                raise ValueError("'exp' should be not be None.")
        return self._exp

    @exp.setter
    def exp(self, value):
        if self.op != TransformOpType.NUMERIC:
            raise ValueError("'exp' should not be set.")
        if not isinstance(value, (int, float)):
            raise ValueError("'exp' should be of int or float type.")
        if value == 0.0:
            raise ValueError("'exp' should not be set to 0.0.")
        self._exp = value

    @property
    def attention_query(self):
        if self.op == TransformOpType.ATTENTION and self._attention_query is None:
            raise ValueError("'attention_query' should not be None.")
        return self._attention_query

    @attention_query.setter
    def attention_query(self, value):
        if self.op != TransformOpType.ATTENTION:
            raise ValueError("'attention_query' should not be set.")
        if not isinstance(value, (str, unicode)):
            raise ValueError("'attention_query' should be of str type.")
        self._attention_query = value

    @property
    def attention_type(self):
        if self.op == TransformOpType.ATTENTION and self._attention_type is None:
            raise ValueError("'attention_type' should not be None.")
        return self._attention_type

    @attention_type.setter
    def attention_type(self, value):
        if self.op != TransformOpType.ATTENTION:
            raise ValueError("'attention_type' should not be set.")
        if not isinstance(value, (str, unicode)):
            raise ValueError("'attention_type' should be of str type.")
        self._attention_type = value

    @property
    def attention_args(self):
        return self._attention_args

    @attention_args.setter
    def attention_args(self, value):
        if self.op != TransformOpType.ATTENTION:
            raise ValueError("'attention_args' should not be set.")
        if value is not None and not isinstance(value, dict):
            raise ValueError("'attention_args' should be of dict type.")
        self._attention_args = value if value is not None else {}

    @property
    def shared_embedding(self):
        if self.op == TransformOpType.ATTENTION and self._shared_embedding is None:
            raise ValueError("'shared_embedding' should not be None.")
        return self._shared_embedding

    @shared_embedding.setter
    def shared_embedding(self, value):
        if self.op != TransformOpType.ATTENTION:
            raise ValueError("'shared_embedding' should not be set.")
        if not isinstance(value, bool):
            raise ValueError("'shared_embedding' should be of bool type.")
        self._shared_embedding = value

    @property
    def weight_column_name(self):
        if self.op == TransformOpType.WEIGHTED_CATEGORICAL and self._weight_column_name is None:
            raise ValueError("'weight_column_name' should not be None.")
        return self._weight_column_name

    @weight_column_name.setter
    def weight_column_name(self, value):
        if self.op != TransformOpType.WEIGHTED_CATEGORICAL:
            raise ValueError("'weight_column_name' should not be set.")
        if not isinstance(value, (str, unicode)):
            raise ValueError("'weight_column_name' must be of str type.")
        self._weight_column_name = value

    @property
    def slotid(self):
        return self._slotid

    @slotid.setter
    def slotid(self, value):
        if value is not None and not isinstance(value, int):
            raise ValueError("'slotid' must be of type int")
        self._slotid = value

    def to_obj(self):
        obj = {}
        obj['input'] = self.input
        obj['output'] = self.output
        obj['op'] = self.op
        obj['places'] = self.places

        if self.slotid is not None:
            obj['slotid'] = self.slotid

        if self.op == TransformOpType.BUCKETIZED:
            obj['boundaries'] = self.boundaries
        if self.op == TransformOpType.EMBEDDING or \
                self.op == TransformOpType.PRETRAINED_EMBEDDING or \
                self.op == TransformOpType.SHARED_EMBEDDING or \
                self.op == TransformOpType.ATTENTION:
            obj['dimension'] = self.dimension
            obj['combiner'] = self.combiner
        if self.op == TransformOpType.PRETRAINED_EMBEDDING:
            obj['trainable'] = self.trainable
            obj['pretrained_embedding_file'] = self.pretrained_embedding_file
        if self.op == TransformOpType.CATEGORICAL_IDENTITY or \
                self.op == TransformOpType.SEQUENCE_CATEGORICAL_IDENTITY:
            obj['num_buckets'] = self.num_buckets
        if self.op == TransformOpType.CROSS:
            obj['hash_bucket_size'] = self.hash_bucket_size
        if self.op == TransformOpType.NUMERIC:
            if self.normalizer_fn is not None:
                obj['normalizer_fn'] = self.normalizer_fn
                obj['substract'] = self.substract
                obj['denominator'] = self.denominator
                obj['exp'] = self.exp
        if self.op == TransformOpType.ATTENTION:
            obj['attention_query'] = self.attention_query
            obj['attention_type'] = self.attention_type
            obj['attention_args'] = self.attention_args
            obj['shared_embedding'] = self.shared_embedding
        if self.op == TransformOpType.WEIGHTED_CATEGORICAL:
            obj['weight_column_name'] = self.weight_column_name

        return obj


class Transform(object):
    def __init__(self):
        self._blocks = []
        self._inames = set()
        self._onames = set()

    @property
    def blocks(self):
        return self._blocks

    @property
    def inames(self):
        return self._inames

    def _add_block(self, obj):
        if not isinstance(obj, TransformBlock):
            raise ValueError("'obj' should be of isinstance 'TransformBlock'.")
        if obj.output in self._onames:
            raise ValueError("TransformBlock output name '{}' conflict."
                             .format(obj.output))
        self._onames.add(obj.output)
        if isinstance(obj.input, str):
            self._inames.add(obj.input)
        else:
            self._inames = self._inames.union(obj.input)
        self._blocks.append(obj.to_obj())

    def to_obj(self):
        return self.blocks

    def add_numeric(self, iname, places=[],
                    normalizer_fn=None, substract=0, denominator=1.0, exp=1.0,
                    oname=None, slotid=None):
        block = TransformBlock()
        block.input = iname
        block.output = oname if oname is not None else iname + '.numeric'
        block.places = places
        block.op = TransformOpType.NUMERIC
        block.slotid = slotid
        if normalizer_fn is not None:
            block.normalizer_fn = normalizer_fn
            block.substract = substract
            block.denominator = denominator
            block.exp = exp
        self._add_block(block)
        return block.output

    def add_bucketized(self, iname, places, boundaries, oname=None, slotid=None):
        block = TransformBlock()
        block.input = iname
        block.output = oname if oname is not None else iname + '.bucketized'
        block.places = places
        block.op = TransformOpType.BUCKETIZED
        block.slotid = slotid

        pre = None
        for b in boundaries:
            if pre is not None and b < pre:
                raise ValueError(
                    "Boundaries must be sorted, iname = {}".format(iname))
            pre = b

        block.boundaries = boundaries
        self._add_block(block)
        return block.output

    def add_embedding(self, iname, places, dimension, combiner='mean',
                      oname=None, slotid=None):
        block = TransformBlock()
        block.input = iname
        block.output = oname if oname is not None else iname + '.embedding'
        block.places = places
        block.op = TransformOpType.EMBEDDING
        block.dimension = dimension
        block.combiner = combiner
        block.slotid = slotid
        self._add_block(block)
        return block.output

    def add_shared_embedding(self, inames, places, dimension, combiner='mean',
                             oname=None, slotid=None):
        block = TransformBlock()
        block.input = inames
        block.output = oname if oname is not None else '#'.join(inames) + '.embedding'
        if all(map(lambda x: isinstance(x, str), places)):
            block.places = [places for _ in range(len(inames))]
        elif len(inames) == len(places) and all(map(lambda x: isinstance(x, list), places)):
            block.places = places
        else:
            raise ValueError("'add_shared_embedding' error: 'inames' and 'places' not matched")
        block.op = TransformOpType.SHARED_EMBEDDING
        block.dimension = dimension
        block.combiner = combiner
        block.slotid = slotid
        self._add_block(block)
        return block.output

    def add_pretrained_embedding(self, iname, places, dimension, trainable,
                                 pretrained_embedding_file, combiner='mean',
                                 oname=None, slotid=None):
        block = TransformBlock()
        block.input = iname
        block.output = oname if oname is not None else iname + '.pretrained_embedding'
        block.places = places
        block.op = TransformOpType.PRETRAINED_EMBEDDING
        block.dimension = dimension
        block.combiner = combiner
        block.trainable = trainable
        block.pretrained_embedding_file = pretrained_embedding_file
        block.slotid = slotid
        self._add_block(block)
        return block.output

    def add_indicator(self, iname, places, oname=None, slotid=None):
        block = TransformBlock()
        block.input = iname
        block.output = oname if oname is not None else iname + '.indicator'
        block.places = places
        block.op = TransformOpType.INDICATOR
        block.slotid = slotid
        self._add_block(block)
        return block.output

    def add_categorical_identity(self, iname, places, num_buckets=-1, oname=None,
                                 slotid=None):
        block = TransformBlock()
        block.input = iname
        block.output = oname if oname is not None else iname + '.identity'
        block.places = places
        block.op = TransformOpType.CATEGORICAL_IDENTITY
        block.num_buckets = num_buckets
        block.slotid = slotid
        self._add_block(block)
        return block.output

    def add_sequence_categorical_identity(self, iname, places, num_buckets=-1,
                                          oname=None, slotid=None):
        block = TransformBlock()
        block.input = iname
        block.output = oname if oname is not None else iname + '.identity'
        block.places = places
        block.op = TransformOpType.SEQUENCE_CATEGORICAL_IDENTITY
        block.num_buckets = num_buckets
        block.slotid = slotid
        self._add_block(block)
        return block.output

    def add_cross(self, inames, hash_bucket_size, places=['wide'], oname=None,
                  slotid=None):
        block = TransformBlock()
        block.input = inames
        block.output = oname if oname is not None else '#'.join(inames) + '.cross'
        block.places = places
        block.op = TransformOpType.CROSS
        block.hash_bucket_size = hash_bucket_size
        block.slotid = slotid
        self._add_block(block)
        return block.output

    def add_weighted_categorical(self, iname, places, oname=None, slotid=None):
        block = TransformBlock()
        block.input = iname
        block.output = iname + '.weighted_categorical'
        block.places = places
        block.op = TransformOpType.WEIGHTED_CATEGORICAL
        block.weight_column_name = iname + '.weight'
        block.slotid = slotid
        self._add_block(block)
        return block.output

    def add_attention(self, inames, attention_query, dimension, combiner='mean',
                      attention_type='din', attention_args=None,
                      shared_embedding=False, oname=None, slotid=None):
        """inames 中每个特征都会与attention_query特征做attention处理
          attention_type
            din: alibaba din 论文中的attention方式
          attention_args: python dict 类型, attention 方式的自定义参数, 例如MLP层数
          shared_embedding: 是否共享 embedding

          Note: 会同时添加 attention 特征（如浏览历史）和 query 特征（如排序物品ID）
        """
        block = TransformBlock()
        block.input = inames
        block.output = oname if oname is not None else '#'.join(inames) + '.attention'
        block.places = ['attention']
        block.op = TransformOpType.ATTENTION
        block.dimension = dimension
        block.combiner = combiner
        block.attention_query = attention_query
        block.attention_type = attention_type
        block.attention_args = attention_args
        block.shared_embedding = shared_embedding
        block.slotid = slotid
        self._add_block(block)
        return block.output


class InputData(object):
    def __init__(self):
        self._meta_file = None

    @property
    def meta_file(self):
        if self._meta_file is None:
            raise ValueError("'meta_file' should not be None.")
        return self._meta_file

    @meta_file.setter
    def meta_file(self, value):
        if not isinstance(value, str):
            raise ValueError("'meta_file' must be of str type.")
        self._meta_file = value

    def to_obj(self):
        obj = {}
        obj['meta_file'] = self.meta_file
        return obj


class Configuration(object):
    def __init__(self, assembler, transform, input_data, model_name=None):
        self._assembler = assembler
        self._transform = transform
        self._input_data = input_data
        self._model_name = model_name

    @property
    def assembler(self):
        if self._assembler is None:
            raise ValueError("'assembler' should not be None.")
        return self._assembler

    @assembler.setter
    def assembler(self, value):
        if not isinstance(value, Assembler):
            raise ValueError("'assembler' should be of Assembler instance.")
        self._assembler = value

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        if value is not None and not isinstance(value, Transform):
            raise ValueError("'transform' should be of Transform instance.")
        self._transform = value

    @property
    def input_data(self):
        if self._input_data is None:
            raise ValueError("'input_data' should not be None.")
        return self._input_data

    @input_data.setter
    def input_data(self, value):
        if not isinstance(value, InputData):
            raise ValueError("'input_data' should be of InputData instance.")
        self._input_data = value

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("'model_name' should be of str type")
        self._model_name = value

    def to_json(self):
        obj = {}
        inames = None
        if self.transform is not None:
            inames = self.transform.inames
            obj['transform'] = self.transform.to_obj()
        obj['assembler'] = self.assembler.to_obj(use_blocks=inames)
        obj['input_data'] = self.input_data.to_obj()

        if self.model_name is not None:
            obj['model_name'] = self.model_name

        return json.dumps(obj, indent=2)
