#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import sparkfuel as sf
from pyspark.sql import SparkSession


if len(sys.argv) != 2:
    print("<Usage>: <filename>")
    sys.exit(-1)

filename = sys.argv[1]


def run(_):
    ds = tf.data.TextLineDataset(filename)
    it = ds.make_one_shot_iterator()
    n = it.get_next()

    sess = tf.Session()
    print(sess.run(n))
    print(sess.run(n))


spark = SparkSession.builder.getOrCreate()
with sf.TFSparkSession(spark, num_ps=2) as sess:
    sess.run(run, None)
