#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import commands
import datetime
from multiprocessing import Process, Queue

from utils import call
from utils import logger


if len(sys.argv) != 10:
    logger.info("""Usgae: <hdfs_path> <max_hours> <database_dir> <nthreads>
                <preprocess_script> <exit_on_error> <num_extra_check_hours>
                <num_reserved_hours> <use_incremental_training>""")
    sys.exit(-1)

hdfs_path = sys.argv[1]
max_hours = int(sys.argv[2])
database_dir = sys.argv[3]
nthreads = int(sys.argv[4])
preprocess_script = sys.argv[5]
exit_on_error = True if sys.argv[6] == 'true' else False
num_extra_check_hours = int(sys.argv[7])
num_reserved_hours = int(sys.argv[8])
use_incremental_training = True if sys.argv[9] == 'true' else False

hadoop_bin = "/usr/local/services/tdw_hdfs_client-1.0/bin/tdwdfsclient/bin/hadoop fs " + \
    "-Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_qqkd_sng_buluo_kd_video_group "
cmd = hadoop_bin + " -ls " + hdfs_path
logger.info(cmd)
output = [line.strip().split()[-1] for line in os.popen(cmd)]
output = [p for p in output if os.path.basename(p).startswith('20')]
output = sorted(output, reverse=True)
output = output[:max_hours+num_extra_check_hours]  # 主要是用于跳过有问题的数据，所以多检查一些小时数据

if use_incremental_training:
    output.reverse()
logger.info("list hdfs dirs = {}".format(output))


def get_exist_dirs(database_dir):
    exist_dirs = [p for p in os.listdir(database_dir) if os.path.basename(p).startswith('20')]
    exist_dirs = sorted(exist_dirs, reverse=True)
    return exist_dirs


download_dir_queue = Queue()
exist_dirs = set(get_exist_dirs(database_dir))


def get_skip_paths(skip_path, begin_hour, num_bad_hour):
    delta = datetime.timedelta(hours=1)
    begin_hour = datetime.datetime.strptime(begin_hour, "%Y%m%d%H")
    bad_hour = begin_hour
    invalid = set()
    for _ in range(num_bad_hour):
        p = bad_hour.strftime("%Y%m%d%H")
        invalid.add(os.path.join(skip_path, p))
        bad_hour += delta
    return invalid


# 以下路径数据有问题
invalid_paths = set(
    [
        'hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_qqkd_sng_buluo_kd_video_group/kdfeeds_miniv_train_data_hourly/2020012713',
        'hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_qqkd_sng_buluo_kd_video_group/kdfeeds_miniv_train_data_hourly/2020021119/',
    ]
)
skip_path = 'hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_qqkd_sng_buluo_kd_video_group/kdfeeds_miniv_train_data_hourly/'
begin_hour = '2020010715'
num_bad_hour = 48

invalid_paths = invalid_paths.union(get_skip_paths(skip_path, begin_hour, num_bad_hour))

logger.info("invalid_paths = {}".format(invalid_paths))

last_train_data_file = os.path.join(database_dir, 'last_train_data.txt')
if not os.path.exists(last_train_data_file):
    with open(last_train_data_file, 'w') as f:
        f.write(os.path.basename(output[0]))

for p in output:
    b = os.path.basename(p)
    if b in exist_dirs:
        logger.info("skip existed " + b + " ...")
        continue
    # check task complete
    check_file = os.path.join(p, b + '.check')
    cmd = hadoop_bin + " -test -e " + check_file
    status = commands.getstatusoutput(cmd)
    if status[0] != 0:
        logger.info("skip non-completed task " + b + " ...")
        continue
    if p in invalid_paths:
        logger.info("skip invalid path '{}' ...".format(p))
        continue

    logger.info("add " + p + " to download")
    download_dir_queue.put(p)


# download
logger.info("*********** Begin downloading data *******************")
cmd = "mkdir -p " + database_dir + "/tmp"
call(cmd)
cmd = "rm -rf " + database_dir + "/tmp/*"
call(cmd)


def download(tid):
    should_stop = False

    while True:
        if should_stop:
            logger.info("download thread {} exit".format(tid))
            break
        try:
            logger.info("queue size = {}".format(download_dir_queue.qsize()))
            p = download_dir_queue.get_nowait()
            # 如果dababase目录下数据量已经足够，而且要下的数据不是最新的，那么终止
            basename = os.path.basename(p)
        except Exception:  # 队列为空
            should_stop = True
            continue

        logger.info("hdfs get {}".format(p))
        tmpdir = database_dir + "/tmp/" + basename + "/"

        # download data
        cmd = "bash " + preprocess_script + " " + database_dir + " " + p + " " + tmpdir
        x = call(cmd,  exit_on_error)
        if x != 0:
            continue
        call("sleep 1")

        cmd = "mv " + tmpdir + " " + database_dir
        x = call(cmd, exit_on_error)
        if x != 0:
            continue
        call("sleep 1")


nthreads = min(download_dir_queue.qsize(), nthreads)
logger.info("queue size = {}".format(download_dir_queue.qsize()))
workers = []
for tid in range(nthreads):
    worker = Process(target=download, args=(tid,))
    logger.info("start download thread {} ...".format(tid))
    worker.start()
    workers.append(worker)

for worker in workers:
    worker.join()

# remove old
exist_dirs = get_exist_dirs(database_dir)

remove_dirs = exist_dirs[max_hours+num_reserved_hours+num_extra_check_hours:]

for p in remove_dirs:
    call("rm -rf " + os.path.join(database_dir, p))

logger.info("Download data done.")
