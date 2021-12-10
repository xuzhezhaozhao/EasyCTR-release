#! /usr/bin/env bash

source /usr/local/services/kd_anaconda2-1.0/anaconda2_profile >/dev/null 2>&1
source /usr/local/services/kd_anaconda2_gpu-1.0/anaconda2_profile >/dev/null 2>&1

set -e

source /usr/local/services/kd_tools_easy_ctr-1.0/tools/train/model.conf.default
source model.conf

hadoop_bin="/usr/local/services/tdw_hdfs_client-1.0/bin/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_qqkd_sng_buluo_kd_video_group "
string_index_bin='/usr/local/services/kd_tools_easy_ctr-1.0/tools/string_indexer '
pigz_bin=/usr/local/services/kd_tools_easy_ctr-1.0/tools/pigz

database_dir=$1
p=$2
curdir=$3

# grep -E '^1\||^0\|' ${curdir}/data.txt.raw > ${curdir}/data.txt
# 添加 python import 路径
export PYTHONPATH=/usr/local/services/kd_tools_easy_ctr-1.0/tools/train/:$PYTHONPATH

# download data
mkdir -p ${curdir}/in
bash /usr/local/services/kd_tools_easy_ctr-1.0/tools/train/pull_data_from_hdfs.sh \
    ${p}/in/ \
    ${curdir}/in \
    ${hdfs_download_thread}

if [[ ${generate_samples_script} != '' ]]; then
    echo "'generate_samples_script' must be empty"
    exit -1
fi

if [[ ${shuffle_data_in_hour} == 'true' ]]; then
    echo "'shuffle_data_in_hour' must be 'false'"
    exit -1
fi

# split train and eval data
train_lines=0
if [[ ${test_data_type} == 'average' ]]; then
    echo "'test_data_type' must be 'last'"
    exit -1
fi
echo ${train_lines} > ${curdir}/train_lines.txt

# string index
if [[ ${use_incremental_training} == 'false' && ${do_string_index} == 'true' ]]; then
    echo "'do_string_index' must be 'false'"
    exit -1
else
    echo "Do not do string index"
fi

# shuffle 或者压缩数据
if [[ ${shuffle_data} == 'true' || ${use_spark_fuel} == 'true' ]]; then
    echo "'shuffle_data' must be 'false'"
else
    if [[ ${compression_type} == 'GZIP' ]]; then
        echo 'pigz train.txt ...'
        time (cat ${curdir}/in/attempt_* | ${pigz_bin} -1 -p ${num_compression_thread} > ${curdir}/train.txt.000.gz)
        let sz=`du -s -BM ${curdir}/train.txt.000.gz | awk -F 'M' '{print $1}'`
        if [ $sz -le 10 ]; then
           echo "[warning] download data too small, size = ${sz}M"
           # exit -1
        fi
        rm -rf ${curdir}/in/attempt_*
        echo 'pigz train.txt done'
    else
        echo 'cat data ...'
        time (cat ${curdir}/in/attempt_* > ${curdir}/train.txt.000)
        echo 'cat data done'
    fi
fi
