#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def DEFINE_string(parser, flag, default, description):
    parser.add_argument(
        "--" + flag,
        type=str,
        default=default,
        help=description
    )


def DEFINE_integer(parser, flag, default, description):
    parser.add_argument(
        "--" + flag,
        type=int,
        default=default,
        help=description
    )


def DEFINE_float(parser, flag, default, description):
    parser.add_argument(
        "--" + flag,
        type=float,
        default=default,
        help=description
    )


def DEFINE_bool(parser, flag, default, description):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument(
        "--" + flag,
        type=str2bool,
        default=default,
        help=description
    )


def DEFINE_list(parser, flag, default, description):
    def str2list(v):
        return filter(lambda x: x != '', v.split(','))

    parser.add_argument(
        "--" + flag,
        type=str2list,
        default=default,
        help=description
    )
