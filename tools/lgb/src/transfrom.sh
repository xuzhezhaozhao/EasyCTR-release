#! /usr/bin/env bash

folder=/usr/local/services/kb_easy_ctr_offline_v2-1.0/bin/data/
new_folder=/data/tantanli/lgb/data
[ ! -d ${folder}/data ] && echo "folder not found" && exit 1

[ -d ${new_folder}/dict ] && rm -rf ${new_folder}/dict
mkdir ${new_folder}/dict/
[ -d ${new_folder}/data ] && rm -rf ${new_folder}/data
mkdir ${new_folder}/data/
[ -d ${new_folder}/feature ] && rm -rf ${new_folder}/feature
mkdir ${new_folder}/feature/
[ -d ${new_folder}/model ] && rm -rf ${new_folder}/model
mkdir ${new_folder}/model/

cat ${folder}/data.other/train.txt | sed 's/\$/\t/g' | sed 's/|/\t/g' > ${new_folder}/data/train.csv
cat ${folder}/data.other/eval.txt | sed 's/\$/\t/g' | sed 's/|/\t/g' >${new_folder}/data/test.csv

cp ${folder}/data/*.dict ${new_folder}/dict/
