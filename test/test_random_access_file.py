#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
import sparkfuel as sf
from pyspark.sql import SparkSession

if len(sys.argv) != 3:
    print("<Usage>: <op_path> <filename>")
    sys.exit(-1)

op_path = sys.argv[1]
filename = sys.argv[2]


def run(_):

    print(os.getcwd())

    for x in os.listdir(os.getcwd()):
        print(x)

    op = tf.load_op_library('./libtest_random_access_file_so')
    node = op.test_random_access_file(filename=filename)

    sess = tf.Session()
    sess.run(node)


spark = SparkSession.builder.getOrCreate()
with sf.TFSparkSession(spark, num_ps=2) as sess:
    sess.run(run, None)
