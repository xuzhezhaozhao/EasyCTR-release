#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sparkfuel as sf
from pyspark.sql import SparkSession


def run(_):
    pass


spark = SparkSession.builder.getOrCreate()
rdd = spark.sparkContext.textFile('hdfs://ss-sng-dc-v2/data/SPARK/SNG/g_sng_qqkd_sng_buluo_kd_video_group/easyctr/sparkfuel/test/data.meta')
meta = rdd.collect()
meta = '\n'.join(meta)
print(meta)

with sf.TFSparkSession(spark, num_ps=2) as sess:
    sess.run(run, None)
