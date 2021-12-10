#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import sys
import os
import random
import datetime

from utils import call
from utils import check_out_of_date
from utils import logger


if len(sys.argv) != 11:
    logger.info(
        "Usgae: <database_dir> <num> <output_dir> "
        "<out_of_date> <shuffle_data> <test_data_type> "
        "<num_test_data> <use_spark_fuel> <use_incremental_training> "
        "<skip_num_thr>")
    sys.exit(-1)

database_dir = sys.argv[1]
num_train_data = int(sys.argv[2])
output_dir = sys.argv[3]
out_of_date = int(sys.argv[4])
shuffle_data = True if str.lower(sys.argv[5]) in ('1', 'true', 'y', 'yes') else False
test_data_type = sys.argv[6]
num_test_data = int(sys.argv[7])
use_spark_fuel = True if str.lower(sys.argv[8]) in ('1', 'true', 'y', 'yes') else False
use_incremental_training = True if str.lower(sys.argv[9]) in ('1', 'true', 'y', 'yes') else False
skip_num_thr = int(sys.argv[10])

logger.info("num_train_data = {}".format(num_train_data))
logger.info("num_test_data = {}".format(num_test_data))
logger.info("num_train_data_expired = {}".format(out_of_date))

if test_data_type == 'last' and num_train_data <= num_test_data:
    raise ValueError("num_train_data less than num_test_data")

# 增量训练模式下最后训练的数据
last_train_data = None
last_train_data_file = os.path.join(output_dir, 'last_train_data.txt')
if not os.path.exists(last_train_data_file):
    last_train_data_file = os.path.join(database_dir, 'last_train_data.txt')

if use_incremental_training:
    # get last train data from file
    if os.path.exists(last_train_data_file):
        with open(last_train_data_file) as f:
            x = f.readline().strip()
            if x.startswith('20') and len(x) == 10:
                last_train_data = x
    if last_train_data is None:
        raise ValueError("Sorry, I don't know where to start training")


invalid_data_path = (
    '2020-11-26',
)
# 取出训练数据
exist_dirs = [d for d in os.listdir(database_dir)
              if d.startswith('20')
              and (last_train_data is None or d > last_train_data)
              and d not in invalid_data_path]
exist_dirs = sorted(exist_dirs)
if use_incremental_training:
    # 数据必须按照时间顺序，不能缺失
    # 可以跳过长时间没生成的数据
    exist_dirs_set = set(exist_dirs)
    exist_dirs = []
    delta = datetime.timedelta(days=1)
    current_date = datetime.datetime.strptime(last_train_data, "%Y-%m-%d")
    while True:
        current_date += delta
        s = current_date.strftime("%Y-%m-%d")
        if s in exist_dirs_set:
            exist_dirs.append(s)
        else:
            # 小时 s 数据缺失，判断是否需要跳过
            # 判断条件为: 是否产生了后面至少skip_num_thr的数据
            beyond_num = len([i for i in exist_dirs_set if i > s])
            if beyond_num > skip_num_thr:
                logger.error("Skip '{}' ...".format(s))
                skip_files_output = os.path.join(output_dir, 'skip_files.txt')
                cmd = "echo " + s + " >> " + skip_files_output
                call(cmd)
                continue
            else:
                break
    if len(exist_dirs) < 2:
        logger.error("Database no enough data (incremental training)")
        sys.exit(-1)
else:
    if len(exist_dirs) < num_train_data:
        logger.error("Database no enough data")
        sys.exit(-1)

# 检查训练数据是否过期, 增量模式下不检测
if not use_incremental_training and check_out_of_date(exist_dirs[-1], out_of_date):
    logger.error("Database out of date")
    sys.exit(-1)

if num_train_data <= 0:
    logger.error("no train data, num_train_data is 0")
    sys.exit(-1)

# tf 的输入是一个包含文件名列表的文件, 下面两个变量为训练和测试的输入文件名
train_files_output = os.path.join(output_dir, 'train_files.txt')
eval_files_output = os.path.join(output_dir, 'eval_files.txt')
# 增量模式下记录训练样本
history_train_files_output = os.path.join(output_dir, 'history_train_files.txt')

if use_incremental_training:
    x = len(exist_dirs)   # 增量模式下训练所有可用数据
    train_data_dirs = exist_dirs[-x:]
else:
    train_data_dirs = exist_dirs[-num_train_data:]
logger.info("train_data_dirs = {}".format(train_data_dirs))

last_train_data = train_data_dirs[-1]
incremental_training_check_file = os.path.join(output_dir, 'use_incremental_training.check')
if not os.path.exists(incremental_training_check_file):
    # 非增量模式或者首次进入增量模式下先清空数据目录
    cmd = "rm -rf " + output_dir + "/*"
    call(cmd)
    if use_incremental_training:
        # 写入增量模式 check 文件
        with open(incremental_training_check_file, 'w') as f:
            f.write('')


cmd = "mkdir -p " + output_dir + "/dict"
call(cmd)

train_files_list = []
eval_files_list = []
total_line_cnt = 0
for p in train_data_dirs:
    datadir = os.path.join(database_dir, p)

    if use_spark_fuel:
        filename = os.path.join(datadir, 'hdfs_train_files.txt')
        train_files = sorted(
            [line.strip() for line in open(filename)
             if line.strip() != ''])
        filename = os.path.join(datadir, 'hdfs_eval_files.txt')
        eval_file = []
        if os.path.exists(filename):
            eval_files = sorted(
                [line.strip() for line in open(filename)
                 if line.strip() != ''])
    else:
        train_files = sorted(
            [os.path.join(datadir, p) for p in os.listdir(datadir)
             if p.find('train.txt') >= 0])
        # test_data_type 为 average 时会有这个文件
        # 文件名可能为 eval.txt 或者 eval.txt.gz
        eval_files = sorted(
            [os.path.join(datadir, p) for p in os.listdir(datadir)
             if p.find('eval.txt') >= 0])

    train_files_list.extend(train_files)
    eval_files_list.extend(eval_files)

    if not use_incremental_training:
        # concat dict files
        dict_files = [p for p in os.listdir(os.path.join(datadir, 'dict')) if p.endswith('.dict')]
        stat_files = [p for p in os.listdir(os.path.join(datadir, 'dict')) if p.endswith('.stat')]
        for dict_file in dict_files:
            cmd = "cat " + os.path.join(datadir, 'dict', dict_file) + " >> " \
                + os.path.join(output_dir, 'dict', dict_file)
            call(cmd)
        for stat_file in stat_files:
            cmd = "cat " + os.path.join(datadir, 'dict', stat_file) + " >> " \
                + os.path.join(output_dir, 'dict', stat_file)
            call(cmd)
            cmd = "echo '' >> " + os.path.join(output_dir, 'dict', stat_file)
            call(cmd)

    # 统计训练总行数，方便模型使用
    line_cnt_file = os.path.join(datadir, 'train_lines.txt')
    with open(line_cnt_file) as f:
        contents = f.readlines()
        try:
            total_line_cnt += int(contents[0].strip())
        except Exception:
            total_line_cnt = 0


# 有两种取测试数据的方式
# average: 取每小时 5% 数据作为测试数据，
# last:    取最后几个小时数据作为测试数据
if test_data_type == 'last':
    eval_files_list = train_files_list[-num_test_data:]
    train_files_list = train_files_list[:-num_test_data]
elif test_data_type == 'average':
    # avaerage 使用 eval.txt 文件做测试, 这里不需要处理
    pass
else:
    raise ValueError('Unknown test_data_type "{}"'.format(test_data_type))

# 输出训练&测试数据文件
if shuffle_data:
    # 难以做到 100% shuffle 数据，这里采用 shuffle 文件的方式近似 shuffle 数据
    random.shuffle(train_files_list)

for train_file in train_files_list:
    cmd = "echo " + train_file + " >> " + train_files_output
    call(cmd)
    if use_incremental_training:
        cmd = "echo " + train_file + " >> " + history_train_files_output
        call(cmd)

for eval_file in eval_files_list:
    cmd = "echo " + eval_file + " >> " + eval_files_output
    call(cmd)
    if use_incremental_training and test_data_type == 'last':
        cmd = "echo " + eval_file + " >> " + history_train_files_output
        call(cmd)

# 输出训练数据总行数
total_line_cnt_file = os.path.join(output_dir, "total_train_lines.txt")
with open(total_line_cnt_file, 'w') as f:
    f.write(str(total_line_cnt))

if use_incremental_training:
    history_total_line_cnt_file = os.path.join(output_dir, "history_total_train_lines.txt")
    cmd = "echo " + str(total_line_cnt) + " >> " + history_total_line_cnt_file
    call(cmd)

if use_incremental_training:
    with open(last_train_data_file, 'w') as f:
        f.write(last_train_data)

# 增量训练模式不合并
if not use_incremental_training:
    # 合并词典
    logger.info("merge dict ...")
    files = [name for name in os.listdir(os.path.join(output_dir, 'dict')) if name.endswith('.dict')]
    for dict_file in files:
        logger.info("merge {} ...".format(dict_file))
        dict_file = os.path.join(output_dir, 'dict', dict_file)
        counter = Counter()
        # read
        for line in open(dict_file):
            line = line.strip()
            if line == "" or line == '0':
                continue
            tokens = line.split('|')
            if len(tokens) != 2:
                logger.error("dict tokens size is not 2, line = " + line)
                continue
            if tokens[0] == "":
                continue
            counter[tokens[0]] += int(tokens[1])

        # write
        with open(dict_file, 'w') as f:
            for key, cnt in counter.most_common():
                f.write(key)
                f.write('|')
                f.write(str(cnt))
                f.write('\n')

    logger.info("merge dict done")

    # 合并均值、方差
    logger.info("merge mean and stddev ...")
    files = [name for name in os.listdir(os.path.join(output_dir, 'dict')) if name.endswith('.stat')]
    for stat_file in files:
        logger.info("merge {} ...".format(stat_file))
        stat_file = os.path.join(output_dir, 'dict', stat_file)
        stat_mean = 0.0
        stat_std = 0.0
        stat_log_mean = 0.0
        stat_log_std = 0.0
        cnt = 0
        for line in open(stat_file):
            line = line.strip()
            if line == "":
                continue
            tokens = line.split(',')
            stat_mean += float(tokens[0])
            stat_std += float(tokens[1])
            stat_log_mean += float(tokens[2])
            stat_log_std += float(tokens[3])
            cnt += 1
        stat_mean /= cnt
        stat_std /= cnt
        stat_log_mean /= cnt
        stat_log_std /= cnt
        with open(stat_file, 'w') as f:
            f.write(str(stat_mean))
            f.write(',')
            f.write(str(stat_std))
            f.write(',')
            f.write(str(stat_log_mean))
            f.write(',')
            f.write(str(stat_log_std))

    logger.info("merge mean and stddev done")
