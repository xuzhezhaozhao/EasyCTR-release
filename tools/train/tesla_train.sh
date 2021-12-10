#! /usr/bin/env bash

set -e

begin_ts=`date +%s`

echo "************** tesla_train.sh ********************"
cat ${BASH_SOURCE[0]}
echo "**********************************************"

hadoop_bin="hadoop fs "

easyctr_dir=./kd_tools_easy_ctr-1.0
assembler_ops_path=${easyctr_dir}/ops/libassembler_ops.so
serving_warmup_file=${easyctr_dir}/models/tf_serving_warmup_requests

source ${easyctr_dir}/tools/train/model.conf.default
source model.conf
echo "************** model.conf ********************"
cat model.conf
echo "**********************************************"

if [[ ${sf_hdfs_root_dir} == '' ]]; then
    echo 'sf_hdfs_root_dir should not be empty'
    exit -1
fi
set +e
data_dir=`${hadoop_bin} -ls ${sf_hdfs_root_dir}/data/1* | sort | tail -n1 | awk '{print $8}'`
set -e
if [[ ${data_dir} == '' ]]; then
    echo 'data_dir error, run hadoop cmd failed'
    exit -1
fi
train_data_path=${data_dir}/train_files.txt
eval_data_path=${data_dir}/eval_files.txt
conf_path=${data_dir}/conf.json

model_dir=${data_dir}/model_dir
export_model_dir=${data_dir}/export_model_dir

do_train=true
do_eval=true
do_export=true
source ${easyctr_dir}/tools/train/train_params.sh
train_begin_ts=`date +%s`
python ${easyctr_dir}/models/main.py ${params_str} \
    --do_train=${do_train} \
    --do_eval=${do_eval} \
    --do_export=${do_export} \
    --train_data_path=${train_data_path} \
    --eval_data_path=${eval_data_path}
train_end_ts=`date +%s`
train_run_hours=`echo "scale=2;(${train_end_ts}-${train_begin_ts})/3600.0"|bc`
