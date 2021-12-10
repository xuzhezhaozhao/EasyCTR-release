#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import time
import logging
import sys

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(process)d %(filename)s:%(lineno)d] %(message)s')
logger = logging.getLogger(__name__)


def call(cmd, exit_on_error=True):
    logger.info("execute '{}'".format(cmd))
    x = subprocess.call(cmd, shell=True)
    if x != 0:
        logger.error("cmd '{}' execute error".format(cmd))
        if exit_on_error:
            sys.exit(-1)
    time.sleep(0.01)

    return x


# 检查过期时间
def check_out_of_date(ts, out_of_date_days):
    now = time.time()
    ts = time.strptime(ts, "%Y-%m-%d")
    ts = time.mktime(ts)
    logger.info("Database ts = {}".format(ts))
    if (now - ts) > out_of_date_days * 24 * 60 * 60:
        return True
    return False


def total_split(line):
    first_tokens = line.split('|')
    tokens = [None for _ in range(len(first_tokens))]
    for first_idx, first_token in enumerate(first_tokens):
        second_tokens = first_token.split('$')
        tokens[first_idx] = [None for _ in range(len(second_tokens))]
        for second_idx, second_token in enumerate(second_tokens):
            third_tokens = second_token.split('\t')
            tokens[first_idx][second_idx] = third_tokens

    return tokens


def check_tokens(mapping, tokens):
    """检查数据是否正确
      mapping: Python dict. field_name -> (pos0, pos1, pos2, func)
      tokens: Python list. Total splited tokens.

    Return:
     True if data is ok, or False.
    """

    for field in mapping:
        pos0, pos1, pos2, func = mapping[field]
        try:
            _ = tokens[pos0][pos1][pos2]
        except Exception:
            return False

    return True
