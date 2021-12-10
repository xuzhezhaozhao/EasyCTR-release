#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(process)d %(filename)s:%(lineno)d] %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=str,
    default='',
    help=''
)

parser.add_argument(
    "--min_count",
    type=int,
    default=0,
    help=''
)

parser.add_argument(
    "--top_k",
    type=int,
    default=-1,
    help=''
)

parser.add_argument(
    "--output_item_features",
    type=str,
    default='',
    help=''
)

parser.add_argument(
    "--output_item_keys",
    type=str,
    default='',
    help=''
)


def parse(opts):
    dirty_cnt = 0
    total_cnt = 0
    selected = 0
    with open(opts.output_item_features, 'w') as ff, open(opts.output_item_keys, 'w') as fk:
        for lineindex, line in enumerate(open(opts.input)):
            total_cnt += 1
            line = line.strip()
            if line == '':
                continue
            tokens = line.split(';')
            if len(tokens) != 3:
                dirty_cnt += 1
                logger.error('candidate items input error, lineindex = {}'.format(lineindex))
                continue
            cnt = int(tokens[2])
            if cnt < opts.min_count:
                continue
            if tokens[0] == '':
                dirty_cnt += 1
                continue
            fk.write(tokens[0] + '\n')
            ff.write(tokens[1] + '\n')

            selected += 1
            if opts.top_k > 0 and selected >= opts.top_k:
                break

    logger.error("candidate items input dirty cnt = {}".format(dirty_cnt))
    logger.info("candidate items input valid cnt = {}".format(selected))
    if 1.0 * dirty_cnt / total_cnt > 0.05:
        raise ValueError("candidate items input contains too many errors(>5%)")


if __name__ == '__main__':
    opts = parser.parse_args()
    parse(opts)
