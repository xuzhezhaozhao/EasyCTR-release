#! /usr/bin/env bash

cd $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

source /usr/local/services/kd_anaconda2-1.0/anaconda2_profile >/dev/null 2>&1
source /usr/local/services/kd_anaconda2_gpu-1.0/anaconda2_profile >/dev/null 2>&1

data=data.txt
nrows=''
if [ $# == 1 ] ; then
    data=$1
elif [ $# == 2 ] ; then
    data=$1
    nrows=$2
fi

# 用下面的命令从 database concat 原始数据
# find ~/services/kb_easy_ctr_offline_v1/bin/data/database/ -name 'data.txt.raw' | xargs -i cat {} > data.txt

cat ${data} | sed 's/\$/	/g' | sed 's/|/	/g' > pandas.tsv
python parse_data.py ./data.meta ./pandas.tsv $nrows > stats.txt
