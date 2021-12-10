#!/usr/bin/bash

PATHON_PATH=/usr/local/services/kd_anaconda3_cpu-1.0/lib/anaconda3/bin
local_root=/data/tantanli/lgb
cur_sec_and_ns=`date '+%s-%N'`
cur_sec=${cur_sec_and_ns%-*}
run_dir=${local_root}/src

PATHON_PATH=/usr/local/services/kd_anaconda3_cpu-1.0/lib/anaconda3/bin
echo ${PATHON_PATH}/python $run_dir/train.py --root_dir=$local_root --version=$cur_sec
${PATHON_PATH}/python $run_dir/train.py --root_dir=$local_root --version=$cur_sec

