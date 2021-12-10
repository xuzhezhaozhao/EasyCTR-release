#!/bin/bash

set -e

if [[ $# -ne 1 ]]; then
    echo 'Usage: <day>'
    exit -1
fi

#日期参数
cur_day=$1

source /data1/11090252/common/easyctr/tools/train/model.conf.default
source model.conf

#下载数据日期参数，次留4，付费5，其它3
if [ $hive_cv_type == 8 ];then
	download_data_day=`date -d "4 days ago ${cur_day}" +%Y-%m-%d`
elif [ $hive_cv_type == 9 ]; then
	download_data_day=`date -d "5 days ago ${cur_day}" +%Y-%m-%d`
else
	download_data_day=`date -d "3 days ago ${cur_day}" +%Y-%m-%d`
fi

echo "${download_data_day}"

database_dir=`pwd`/data/database
database_tmp_dir=${database_dir}/tmp
tmp_data_dir=${database_tmp_dir}/${download_data_day}
tmp_dict_dir=${database_tmp_dir}/${download_data_day}/dict
final_data_dir=${database_dir}/${download_data_day}

rm -rf ${tmp_data_dir}

mkdir -p ${database_dir}
mkdir -p ${database_tmp_dir}
mkdir -p ${tmp_data_dir}
mkdir -p ${tmp_dict_dir}

# 下载训练样本数据
hive_sql=`echo $hive_sql | sed "s/@download_data_day@/${download_data_day}/g"`
hive_sql=`echo $hive_sql | sed "s/@hive_cv_type@/${hive_cv_type}/g"`
hive_sql=`echo $hive_sql | sed "s/@hive_media_type@/${hive_media_type}/g"`
hive_sql=`echo $hive_sql | sed "s/@hive_cv_label_thr@/${hive_cv_label_thr}/g"`
hive_sql=`echo $hive_sql | sed "s/@hive_page_type@/${hive_page_type}/g"`
echo "hive_sql = ${hive_sql}"
hive -e "${hive_sql}" | awk -F '\t' '{print $1}' > ${tmp_data_dir}/train.txt

train_lines=$(wc -l ${tmp_data_dir}/train.txt | awk '{print $1}')
echo ${train_lines} > ${tmp_data_dir}/train_lines.txt

# add data check
if [[ ${train_lines} -lt ${min_line_count} ]]; then
    echo "'${cur_day}' data too small, train_lines = ${train_lines}, failed"
    python /data1/11090252/common/easyctr/tools/train/v_chat_tools.py "'${download_data_day}':`hostname --fqdn`:`pwd -P`:data too small, train_lines = ${train_lines}, failed"
    exit -1
fi

#生成字典，并将数据从tmp目录移动到正式目录
echo "successed to get last 3 day trainnning data"

echo "generating data dict file ..."
/data1/11090252/common/easyctr/tools/string_indexer/string_indexer ./data.meta ${tmp_data_dir}/train.txt ${tmp_dict_dir}
echo "generate data dict file done"

echo "generating final data file ..."
rm -rf ${final_data_dir}
mkdir -p ${final_data_dir}
echo "generate final data file done"

echo "moving tmp data file to final data file ..."
mv ${tmp_data_dir}/* ${final_data_dir}
echo "move tmp data file to final data file done"

echo "removing tmp data file ..."
rm -rf ${tmp_data_dir}

echo "remove tmp data file "${tmp_data_dir}" done"

echo "check task done"
if [[ ! -d "${final_data_dir}"]]; then
    python /data1/11090252/common/easyctr/tools/train/v_chat_tools.py "'${download_data_day}':`hostname --fqdn`:`pwd -P`:回流模型下载数据失败"
fi
echo "check task done finish"
