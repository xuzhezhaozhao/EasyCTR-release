#! /usr/bin/env bash

folder=/usr/local/services/kb_easy_ctr_offline_v5-1.0/bin/data/
new_folder=/data/cicixichen/lgb/data
local_root=/data/cicixichen/lgb
run_dir=${local_root}/src
cur_sec_and_ns=`date '+%s-%N'`
cur_sec=${cur_sec_and_ns%-*}
[ ! -d ${folder}/database/ ] && echo "folder not found" && exit 1
[ -d ${new_folder}/dict ] && rm -rf ${new_folder}/dict
mkdir ${new_folder}/dict/
[ -d ${new_folder}/data ] && rm -rf ${new_folder}/data
mkdir ${new_folder}/data/
[ -d ${new_folder}/feature ] && rm -rf ${new_folder}/feature
mkdir ${new_folder}/feature/
[ -d ${new_folder}/model ] && rm -rf ${new_folder}/model
mkdir ${new_folder}/model/

cat ${folder}/database/train_0721.txt |sed 's/\$/\t/g' | sed 's/|/\t/g' > ${new_folder}/data/train_0721.csv
cat ${folder}/database/eval_0721.txt | sed 's/\$/\t/g' | sed 's/|/\t/g' >${new_folder}/data/test_0721.csv
#扔掉没有对齐的字段
cat ${new_folder}/data/test_0721.csv | awk -F'\t' 'NF==142{print $0}' > ${new_folder}/data/test_0721.csv.clean
cat ${new_folder}/data/train_0721.csv | awk -F'\t' 'NF==142{print $0}' > ${new_folder}/data/train_0721.csv.clean
#PYTHON_PATH=/usr/local/services/kd_anaconda3_cpu-1.0/lib/anaconda3/bin
#echo ${PYTHON_PATH}/python ${run_dir}/duiqi.py --root_dir=${local_root} --version=${cur_sec} 
#${PYTHON_PATH}/python ${run_dir}/duiqi.py --root_dir=${local_root} --version=${cur_sec} 
cp ${folder}/data/*.dict ${new_folder}/dict/
