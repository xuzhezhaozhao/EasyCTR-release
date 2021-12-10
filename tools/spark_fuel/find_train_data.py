#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.sql import SparkSession
import sys
import os.path


if len(sys.argv) != 2:
    raise ValueError("Usage: <root_dir>")

root_dir = sys.argv[1]
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
hadoop_conf = sc._jsc.hadoopConfiguration()
p = sc._gateway.jvm.org.apache.hadoop.fs.Path(root_dir)
fs = p.getFileSystem(hadoop_conf)
files = fs.listStatus(p)
files = [f.getPath().toString() for f in files]
files = [f for f in files if os.path.basename(f).startswith('1')]
files = sorted(files, reverse=True)[:5]

print(files)

src_path = None
for d in files:
    check_file = os.path.join(d, 'file.check')
    p = sc._gateway.jvm.org.apache.hadoop.fs.Path(check_file)
    if fs.exists(p):
        src_path = d
        break

if not src_path:
    raise ValueError("No valid train data dir")

dst_path = os.path.join(root_dir, 'data')

# copy
sp = sc._gateway.jvm.org.apache.hadoop.fs.Path(src_path)
dp = sc._gateway.jvm.org.apache.hadoop.fs.Path(dst_path)
if fs.exists(dp):
    if not fs.delete(dp, True):
        raise ValueError("Delete dst_path '{}' failed".format(dst_path))

src_fs = sp.getFileSystem(hadoop_conf)
dst_fs = dp.getFileSystem(hadoop_conf)
if not sc._gateway.jvm.org.apache.hadoop.fs.FileUtil.copy(
        src_fs, sp, dst_fs, dp, False, True, hadoop_conf):
    raise ValueError("Copy data dir failed")
