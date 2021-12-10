#! /usr/bin/env bash

source /usr/local/services/kd_anaconda2-1.0/anaconda2_profile >/dev/null 2>&1
source /usr/local/services/kd_anaconda2_gpu-1.0/anaconda2_profile >/dev/null 2>&1

if [ $# != 2 ] ; then
    echo "Usage: <model_dir> <param_name>"
    exit -1
fi

model_dir=$1
param_name=$2

python /usr/local/services/kd_tools_easy_ctr-1.0/tools/show_param.py ${model_dir} ${param_name}
