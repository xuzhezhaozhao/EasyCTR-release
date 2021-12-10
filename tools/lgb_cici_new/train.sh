#!/usr/bin/bash

PYTHON_PATH=/usr/local/services/kd_anaconda3_cpu-1.0/lib/anaconda3/bin
local_root=/data/cicixichen/lgb
cur_sec_and_ns=`date '+%s-%N'`
cur_sec=${cur_sec_and_ns%-*}
run_dir=${local_root}/src
feature_dir=${local_root}/data/feature
data_dir=${local_root}/existing_data

PYTHON_PATH=/usr/local/services/kd_anaconda3_cpu-1.0/lib/anaconda3/bin
echo ${PYTHON_PATH}/python $run_dir/train.py --root_dir=$local_root --version=$cur_sec
${PYTHON_PATH}/python $run_dir/train.py --root_dir=$local_root --version=$cur_sec

echo ${PYTHON_PATH}/python $run_dir/trans_dec.py ${feature_dir}/feature_importance_080222_withsl_f.csv ${data_dir}/feature_importance_080222_withsl_f.csv --root_dir=$local_root --version=$cur_sec ${PYTHON_PATH}/python $run_dir/trans_dec.py ${feature_dir}/feature_importance_080222_withsl_f.csv ${data_dir}/feature_importance_080222_withsl_f.csv --root_dir=$local_root --version=$cur_sec

cat ${data_dir}/feature_importance_080222_withsl_f.csv | sort -t ',' -k2rn > ${data_dir}/feature_importance_080222_withsl_sorted.csv 
