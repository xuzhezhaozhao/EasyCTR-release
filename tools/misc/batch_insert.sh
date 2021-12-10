#! /usr/bin/env bash

set -e


if [ $# != 4 ] ; then
    echo "Usage: <meta_file> <ip> <port> <password>"
    exit -1
fi


meta_file=$1
ip=$2
port=$3
password=$4

while read line
do
    group_name=`echo $line | cut -d " " -f1`
    field_name=`echo $line | cut -d " " -f2`
    data_type=`echo $line | cut -d " " -f3`
    field_type=`echo $line | cut -d " " -f4`

    sql="insert into kb_feature_center_field_config (name, operator, available_time, status, belong_group, data_type, field_type) values ('${field_name}', 'zhezhaoxu', '2017-07-23 09:00:00', 1, '${group_name}', '${data_type}', '${field_type}')"

    mysql -h${ip} -P${port} -uKanDianApp1 -p${password} -D kd_video_kuaibao --default-character-set=utf8 -e "${sql}"

    echo "insert ${field_name} done"
done < $1
