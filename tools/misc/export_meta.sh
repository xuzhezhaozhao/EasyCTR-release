#! /usr/bin/env bash

set -e

cd "$( dirname "${BASH_SOURCE[0]}" )"

if [ $# != 4 ] ; then
    echo "Usage: <ip> <port> <user> <password>"
    exit -1
fi

ip=$1
port=$2
user=$3
password=$4

hadoop_bin="/usr/local/services/tdw_hdfs_client-1.0/bin/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_qqkd_sng_buluo_kd_video_group "
target_dir=hdfs://ss-sng-dc-v2/data/SPARK/SNG/g_sng_qqkd_sng_buluo_kd_video_group/feature_center/meta/

ts=`date +%s`
meta_file=feature.meta.${ts}
rm -rf feature.meta.*
mysql -h${ip} -P${port} -u${user} -p${password} -D kd_video_kuaibao --default-character-set=utf8 -e "select id,name,data_type,field_type,created_time,available_time,modified_time,status,belong_group,original_name,business_type from kb_feature_center_field_config where status=1" > ${meta_file}

${hadoop_bin} -mkdir -p ${target_dir}/tmp
${hadoop_bin} -put ${meta_file} ${target_dir}/tmp
${hadoop_bin} -mv ${target_dir}/tmp/${meta_file} ${target_dir}/${meta_file}

bash ./hdfs_remove_outdate.sh ${target_dir} 14
