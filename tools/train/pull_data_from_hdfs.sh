#! /usr/bin/env bash

set -e

hdfs_path=$1
output_path=$2
hdfs_download_thread=$3
hadoop_bin="/usr/local/services/tdw_hdfs_client-1.0/bin/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_qqkd_sng_buluo_kd_video_group "

parallel_download() {
${hadoop_bin} -ls $1 > ${output_path}/ls_cmd.tmp
sed 's/.*hdfs:/hdfs:/g' ${output_path}/ls_cmd.tmp > ${output_path}/file_list.tmp

cnt=0
for line in `cat ${output_path}/file_list.tmp`
do
	nohup ${hadoop_bin} -get $line ${output_path} > ${output_path}/hdfs.log 2>&1 &
    let cnt+=1
    stop=$((cnt % ${hdfs_download_thread}))
    if [[ $stop == 0 ]]; then
        wait
    fi
done
wait
}

parallel_download ${hdfs_path}
