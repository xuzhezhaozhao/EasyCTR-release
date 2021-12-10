#! /usr/bin/env bash
set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

hadoop_bin="hadoop fs "

#待检测的HDFS目录
data1_file=$1
num_outdate_days=$2

#将待检测的目录(可以为多个)加载至数组中
array_check=($data1_file)

# 当前时间戳
today_timestamp=$(date -d "$(date +"%Y-%m-%d %H:%M")" +%s)

#Func: 删除指定时间之前的过期
removeOutDate(){
        $hadoop_bin -ls $1 > .temp.txt
        cat .temp.txt | while read quanxian temp user group size day hour filepath
        do
            current_file_time="$day $hour"
            current_file_timestamp=$(date -d "$current_file_time" +%s)
            if [ $(($today_timestamp-$current_file_timestamp)) -ge $((${num_outdate_days}*24*60*60)) ];then
                echo "$(date +'%Y-%m-%d %H:%M:%S') $filepath"
                $hadoop_bin -rm -r $filepath > /dev/null 2>&1
            fi
        done
}

#Func: 执行删除
execute(){
        echo -e "\n\n"
        echo "$(date +'%Y-%m-%d %H:%M:%S') start to remove outdate files in hdfs"
        echo "$(date +'%Y-%m-%d %H:%M:%S') today is: $(date +"%Y-%m-%d %H:%M:%S")"

        for i in ${array_check[@]}
        do
            echo "$(date +'%Y-%m-%d %H:%M:%S') processing filepath: $i"
            removeOutDate $i
            echo -e "\n"
        done

        echo "$(date +'%Y-%m-%d %H:%M:%S') remove outdate files in hdfs finished"
        echo -e "\n\n"
}

# 开始执行
execute
