#! /usr/bin/env bash

set -e

if [ $# != 5 ] ; then
    echo "Usage: <file_pattern> <dict> <min_count> <top_k> <output>"
    exit -1
fi

file_pattern=$1
dict=$2
min_count=$3
top_k=$4
output=$5

hadoop_bin="/usr/local/services/tdw_hdfs_client-1.0/bin/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_qqkd_sng_buluo_kd_video_group "

# 获取最新的 embedding 文件路径
embedding_file=`${hadoop_bin} -ls ${file_pattern}  2>/dev/null | awk '{print $8}' | sort | tail -n1`

###############################################################################
# 检查文件是否过期 (3 天)
echo "check data timestampe ..."
array_check=($embedding_file)
today_timestamp=$(date -d "$(date +"%Y-%m-%d %H:%M")" +%s)
out_of_date_hours=72
checkOutDate() {
    ${hadoop_bin} -ls $1 > .temp.txt
    cat .temp.txt | while read quanxian temp user group size day hour filepath
    do
        current_file_time="$day $hour"
        current_file_timestamp=$(date -d "$current_file_time" +%s)
        if [ $(($today_timestamp-$current_file_timestamp)) -ge $((${out_of_date_hours}*60*60)) ];then
            echo "$(date +'%Y-%m-%d %H:%M:%S') $1 out of date"
            exit -1
        fi
    done
}
for filename in ${array_check[@]}
do
    echo "$(date +'%Y-%m-%d %H:%M:%S') processing filepath: $filename"
    checkOutDate $filename
    echo -e "\n"
done
###############################################################################

###############################################################################
# 下载数据
data_dir=./data/data
mkdir -p ${data_dir}
filename=`basename ${embedding_file}`
rm -rf ${data_dir}/${filename}
echo "Downloading ${data_dir}/${filename} ..."
${hadoop_bin} -get ${embedding_file} ${data_dir}/
echo "Download done"
###############################################################################

##############################################################################
# 处理 embedding 数据，匹配实际词典
crop_embedding_bin='/usr/local/services/kd_tools_easy_ctr-1.0/tools/crop_embedding '
echo "crop embeddings ${data_dir}/${filename} 1 ${dict} ${min_count} ${top_k} ${output}.tmp ..."
${crop_embedding_bin} ${data_dir}/${filename} 1 ${dict} ${min_count} ${top_k} ${output}.tmp
echo "crop embeddings done"

echo "fill default value ..."
python /usr/local/services/kd_tools_easy_ctr-1.0/tools/fill_default_values.py ${output}.tmp ${output}
echo "fill default value done"
##############################################################################
