#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import commands


if len(sys.argv) != 4:
    print("Usgae: <hdfs_path> <max_hours> <output_dir>")
    sys.exit(-1)

hdfs_path = sys.argv[1]
max_hours = int(sys.argv[2])
output_dir = sys.argv[3]

hadoop_bin = "/usr/local/services/tdw_hdfs_client-1.0/bin/tdwdfsclient/bin/hadoop fs " + \
    "-Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_qqkd_sng_buluo_kd_video_group "
cmd = hadoop_bin + " -ls " + hdfs_path
print(cmd)
output = [line.strip().split()[-1] for line in os.popen(cmd)]
output = [p for p in output if os.path.basename(p).startswith('2')]
output = sorted(output)
output = output[-max_hours:]
print("list hdfs dirs = {}".format(output))

download_dir = None
for p in output:
    b = os.path.basename(p)
    # check task complete
    check_file = os.path.join(p, b + '.check')
    cmd = hadoop_bin + " -test -e " + check_file
    status = commands.getstatusoutput(cmd)
    if status[0] != 0:
        print("skip non-completed task " + b + " ...")
        continue
    download_dir = p
    break


if download_dir is None:
    print("No data")
    sys.exit(-1)

# download
print("*********** Begin downloading data *******************")
p = download_dir
basename = os.path.basename(p)
curdir = output_dir + "/" + basename + "/"

# download data
cmd = hadoop_bin + " -get " + p + " " + output_dir
print(cmd)
os.system(cmd)

# concat data
cmd = "cat " + curdir + "/in/attempt_* > " + output_dir + "/data.txt"
print(cmd)
os.system(cmd)

# remove data
cmd = "rm -rf " + curdir + "/in/attempt_*"
print(cmd)
os.system(cmd)
