#! /usr/bin/env bash

source /usr/local/services/kd_anaconda2-1.0/anaconda2_profile >/dev/null 2>&1
source /usr/local/services/kd_anaconda2_gpu-1.0/anaconda2_profile >/dev/null 2>&1

if [ $# != 1 ] ; then
    echo "Usage: <model_dir>"
    exit -1
fi

model_dir=$1

process_name=tensorboard
pid=$(ps aux | grep ${process_name} | grep python | awk '{print $2}')

if [[ ${pid} != '' ]]; then
    echo 'kill old tensorboard ...'
    kill -9 ${pid}
    echo 'kill old tensorboard done'
fi

sleep 1
tensorboard --logdir ${model_dir} --port 8080 > tensorboard.log 2>&1 &
