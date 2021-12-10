#! /usr/bin/env bash

if [[ $# -ne 1 ]]; then
    echo 'Usage: <day>'
    exit -1
fi

#日期参数
cur_day=$1

source /data1/11090252/common/anaconda2/anaconda2_profile >/dev/null 2>&1
source /data1/11090252/common/anaconda2_gpu/anaconda2_profile >/dev/null 2>&1

source /data1/11090252/common/easyctr/tools/train/model.conf.default
source model.conf

log_dir=data/log
mkdir -p ${log_dir}

# 清理日志
find ${log_dir}* -mtime +14 -exec rm -rf {} \; >/dev/null 2>&1

database_dir=data/database
mkdir -p ${database_dir}

ts=`date +%s%6N`
logfile=`pwd -P`/${log_dir}/crontab_download_data.log.${ts}
ln -sfT ${logfile} ${log_dir}/crontab_download_data.log

# download and process
/data1/11090252/common/easyctr/tools/train/get_reflow_offline_sample.sh \
    ${cur_day} >> ${logfile} 2>&1

# clean data
clean_dir=${database_dir}
keep_max=$((num_train_data+num_test_data+num_reserved))
keep=0
files=$(ls -d ${clean_dir}/2* | sort -r)
for file in ${files}
do
    keep=$((keep+1))
    echo "check ${file} ..."
    if [[ ${keep} -gt ${keep_max} ]]; then
        echo "rm -rf ${file} ..."
        rm -rf ${file}
    fi
done
