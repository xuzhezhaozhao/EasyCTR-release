#! /usr/bin/env bash

source /usr/local/services/kd_anaconda2-1.0/anaconda2_profile >/dev/null 2>&1
source /usr/local/services/kd_anaconda2_gpu-1.0/anaconda2_profile >/dev/null 2>&1

source /usr/local/services/kd_tools_easy_ctr-1.0/tools/train/model.conf.default
source model.conf

set -e

data_dir=`pwd -P`/data/data

echo "download candidate items from ${hdfs_easyrecall_candidate_items_path} ..."
mkdir -p ${data_dir}/candidate_items/
rm -rf ${data_dir}/candidate_items/*
python /usr/local/services/kd_tools_easy_ctr-1.0/tools/train/easyrecall/download_candidate_items.py \
    ${hdfs_easyrecall_candidate_items_path} \
    8 \
    ${data_dir}/candidate_items/
echo "download candidate items done"

sort -t ';' -k 3 -n -r ${data_dir}/candidate_items/data.txt > ${data_dir}/candidate_items/data.txt.sorted
cp ${data_dir}/candidate_items/data.txt.sorted ${data_dir}/candidate_items/data.txt

echo "parse candidate items ..."
python /usr/local/services/kd_tools_easy_ctr-1.0/tools/train/easyrecall/parse_candidate_items.py \
    --input ${data_dir}/candidate_items/data.txt.sorted \
    --output_item_features ${data_dir}/candidate_items_features.txt \
    --output_item_keys ${data_dir}/candidate_items.txt \
    --min_count ${easyrecall_candidate_items_min_count} \
    --top_k ${easyrecall_candidate_items_top_k}
echo "parse candidate items done"

pigz_bin=/usr/local/services/kd_tools_easy_ctr-1.0/tools/pigz
if [[ ${compression_type} == 'GZIP' ]]; then
    echo "gzip candidate items ..."
    rm -rf ${data_dir}/candidate_items_features.txt.gz
    time ${pigz_bin} -1 -p ${num_compression_thread} ${data_dir}/candidate_items_features.txt
    echo "${data_dir}/candidate_items_features.txt.gz" > ${data_dir}/predict_files.txt
    echo "gzip candidate items done"
else
    echo "${data_dir}/candidate_items_features.txt" > ${data_dir}/predict_files.txt
fi
