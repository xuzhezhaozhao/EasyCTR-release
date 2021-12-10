#! /usr/bin/env bash

process_name=/usr/local/services/kd_tools_easy_ctr-1.0/tools/train/train.sh
ps aux | grep ${process_name} | grep -v grep
pid=$(ps aux | grep ${process_name} | grep -v grep | awk '{print $2}')
if [[ $pid != '' ]]; then
    echo "kill training process group [${pid}] ..."
    kill $(pstree ${pid} -p -a -l | cut -d, -f2 | cut -d' ' -f1)
fi
sleep 5


process_name=/usr/local/services/kd_tools_easy_ctr-1.0/tools/train/crontab_download_data.sh
ps aux | grep ${process_name} | grep -v grep | grep -v flock
pid=$(ps aux | grep ${process_name} | grep -v grep | grep -v flock | awk '{print $2}')
if [[ $pid != '' ]]; then
    echo "kill download process group [${pid}] ..."
    kill $(pstree ${pid} -p -a -l | cut -d, -f2 | cut -d' ' -f1)
fi
sleep 5

rm -rf data/database/*
rm -rf data/data/*
rm -rf data/model_dir.bak
if [[ -d 'data/model_dir' ]]; then
    mv data/model_dir data/model_dir.bak
fi
