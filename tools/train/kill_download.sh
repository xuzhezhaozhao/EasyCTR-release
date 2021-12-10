#! /usr/bin/env bash

process_name=/usr/local/services/kd_tools_easy_ctr-1.0/tools/train/crontab_download_data.sh
ps aux | grep ${process_name} | grep -v grep | grep -v flock
pid=$(ps aux | grep ${process_name} | grep -v grep | grep -v flock | awk '{print $2}')
if [[ $pid != '' ]]; then
    echo "kill download process group [${pid}] ..."
    kill $(pstree ${pid} -p -a -l | cut -d, -f2 | cut -d' ' -f1)
fi
