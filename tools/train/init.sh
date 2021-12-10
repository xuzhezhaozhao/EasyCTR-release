#! /usr/bin/env bash

source /data1/11090252/common/easyctr/tools/train/model.conf.default
source model.conf

chmod +x *.py > /dev/null 2>&1
chmod +x *.sh > /dev/null 2>&1

for d in data
do
    mkdir -p $d
done

ln -sf ${meta_file} ./data.meta

rm -f tools # 先删除软连接
ln -sf /data1/11090252/common/easyctr/tools/ tools

# add crontab
crontab -l >tmp.conf 2>/dev/null
echo "*/10 * * * * bash $(pwd)/crontab_train.sh" >> tmp.conf
cat tmp.conf | sort | uniq > tmp.conf.uniq
crontab tmp.conf.uniq
rm -f tmp.conf tmp.conf.uniq
