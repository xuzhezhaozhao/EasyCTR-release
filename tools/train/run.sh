#! /usr/bin/env bash

log_dir=data/log
mkdir -p ${log_dir}

# 清理日志
find ${log_dir}/* -mtime +14 -exec rm -rf {} \; >/dev/null 2>&1

ts=`date +%Y_%m_%d_%H_%M`
logfile=`pwd -P`/${log_dir}/process.log.${ts}
ln -sfT ${logfile} ${log_dir}/process.log

date >> ${logfile} 2>&1

bash /data1/11090252/common/easyctr/tools/train/train.sh >> ${logfile} 2>&1

if [[ $? -ne 0 ]]; then
  # 推送消息到vchat
  echo "check train and eval finished"
    python /data1/11090252/common/easyctr/tools/train/v_chat_tools.py "'${ts}':`hostname --fqdn`:`pwd -P`:model train or eval failed"
  echo "check train and eval finished done"
else
    echo "train succeed"
fi

date >> ${logfile} 2>&1
