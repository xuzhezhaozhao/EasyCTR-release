#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common.estimator import linear

from common.common import add_hidden_layer_summary


def get_linear_logits(features,
                      linear_feature_columns,
                      units,
                      linear_sparse_combiner,
                      scope):
    logit_fn = linear.linear_logit_fn_builder(
            units=units,
            feature_columns=linear_feature_columns,
            sparse_combiner=linear_sparse_combiner)
    logits = logit_fn(features=features)
    add_hidden_layer_summary(logits, scope.name)

    return logits
