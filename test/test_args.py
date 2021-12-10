#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse


parser = argparse.ArgumentParser()


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
    "--test_bool",
    type=str2bool,
    default=False,
    help=''
)


def str2list(v):
    return v.split(',')


parser.add_argument(
    "--test_list",
    type=str2list,
    default='1,2,3',
    help=''
)


if __name__ == '__main__':
    args = parser.parse_args()
    print('test_bool = {}'.format(args.test_bool))
    print('test_list = {}'.format(args.test_list))
