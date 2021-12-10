#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common.log_label import log_label


def configure_label_fn(opts):
    if opts.label_fn == 'default':
        label_fn = None
    elif opts.label_fn == 'log':
        label_fn = log_label
    else:
        raise ValueError("Unknow label_fn '{}'".format(opts.label_fn))

    return label_fn
