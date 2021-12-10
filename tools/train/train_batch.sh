#! /usr/bin/env bash

source /usr/local/services/kd_anaconda2-1.0/anaconda2_profile >/dev/null 2>&1
source /usr/local/services/kd_anaconda2_gpu-1.0/anaconda2_profile >/dev/null 2>&1

set -e

begin_ts=`date +%s`

echo "************** train.sh ********************"
cat ${BASH_SOURCE[0]}
echo "**********************************************"

source /usr/local/services/kd_tools_easy_ctr-1.0/tools/train/model.conf.default
source model.conf
echo "************** model.conf ********************"
cat model.conf
echo "**********************************************"

old_model_conf=log/model.conf.old
touch ${old_model_conf}
current_model_conf=log/model.conf.current
cp model.conf ${current_model_conf}

num_train_data_hours=${num_train_data_hours}  # 训练数据
num_train_data_expired_hours=${num_train_data_expired_hours}  # 训练数据过期时间

easyctr_dir=/usr/local/services/kd_tools_easy_ctr-1.0
data_dir=`pwd -P`/data/data
database_dir=`pwd -P`/data/database
parent_export_model_dir=`pwd -P`/data/export_model_dir
parent_model_dir=`pwd -P`/data/model_dir
dict_dir=`pwd -P`/data/data/

mkdir -p ${data_dir}
if [[ ${remove_model_dir} == 'true' ]]; then
    rm -rf ${model_dir}.bak
    if [[ -d ${model_dir} ]]; then
        mv ${model_dir} ${model_dir}.bak
    fi
fi
mkdir -p ${parent_model_dir}

# 创建软链接
ln -sf ${easyctr_dir}/tools/conf_generator.py .
ln -sf ${easyctr_dir}/tools/make_conf_list.py .
ln -sf ${conf_script} conf.py

merge_data_begin_ts=`date +%s`
if [[ ${do_merge_data} == 'true' ]]; then
    echo "merge data ..."
    python ${merge_data_python_script} \
        ${database_dir} \
        ${num_train_data_hours} \
        ${data_dir} \
        ${num_train_data_expired_hours} \
        ${shuffle_data} \
        ${test_data_type} \
        ${num_test_data_hours} \
        ${use_spark_fuel} \
        false
    echo "merge data done"
else
    echo "Do not merge data, use old version"
fi

if [[ ${run_mode} == 'easyrecall' ]]; then
    echo "download candidate items ..."
    bash ${easyrecall_download_candidata_items_script}
    echo "download candidate items done"
fi

merge_data_end_ts=`date +%s`
merge_data_run_hours=`echo "scale=2;(${merge_data_end_ts}-${merge_data_begin_ts})/3600.0"|bc`

generate_conf_begin_ts=`date +%s`
if [[ ${do_generate_conf} == 'true' ]]; then
    echo "generate conf ..."
    python conf.py > conf.json
    csplit conf.json /##/ -n 3 -s {*} -f "conf." -b "%03d.json"
    rm conf.json
    sed -i '/##/d' conf*json
    mkdir -p conf_dir
    mv conf*json conf_dir/.
    echo "generate conf done"
else
    echo "Do not need generate conf, use old version"
fi
generate_conf_end_ts=`date +%s`
generate_conf_run_hours=`echo "scale=2;(${generate_conf_end_ts}-${generate_conf_begin_ts})/3600.0"|bc`

# train model
epoch=${epoch}
batch_size=${batch_size}
total_lines=`cat ${data_dir}/total_train_lines.txt`
echo "total_lines = " ${total_lines}
total_steps=`bc <<< ${epoch}*${total_lines}/${batch_size} | awk '{print int($1)}'`
echo 'total_steps = ' ${total_steps}

# tensorflow 1.12.0 read_buffer_size_mb 这个参数比较奇怪，比训练数据大的话某些情况下会出错
total_size_mb=`du -sm data/database/ --exclude='tmp' | awk '{print $1}'`
if [[ ${read_buffer_size_mb} -gt ${total_size_mb} ]]; then
    let read_buffer_size_mb=total_size_mb-1
fi
if [[ ${read_buffer_size_mb} -lt 0 ]]; then
    let read_buffer_size_mb=0
fi
echo 'read_buffer_size_mb = ' ${read_buffer_size_mb}

# 训练分为两步骤，第一使用训练数据训练，然后测试计算 auc;
# 第二步再接着使用测试集训练(测试集为 average 则省去这一步)
do_train=true
do_eval=true
do_export=false # set false
train_data_path=${data_dir}/train_files.txt
eval_data_path=${data_dir}/eval_files.txt
predict_data_path=${data_dir}/predict_files.txt
predict_keys_path=${data_dir}/candidate_items.txt

use_tfrecord=false
#conf_path=`pwd -P`/conf.json
assembler_ops_path=${easyctr_dir}/ops/libassembler_ops.so
serving_warmup_file=${easyctr_dir}/easyctr/tf_serving_warmup_requests
if [[ ${run_mode} == 'easyrecall' ]]; then
    serving_warmup_file=${easyctr_dir}/easyctr/tf_serving_warmup_requests_easyrecall
fi


# thread control
tmp_fifofile="/tmp/$$.fifo"               ##其中$$为该进程的pid
mkfifo $tmp_fifofile                      ##创建命名管道
exec 6<>$tmp_fifofile                     ##把文件描述符6和FIFO进行绑定
rm -f $tmp_fifofile                       ##绑定后，该文件就可以删除了
thread=10                                  ## 进程并发数为10，用这个数字来控制一次提交的请求数
for ((i=0;i<$thread;i++));
do
    echo >&6                              ##写一个空行到管道里，因为管道文件的读取以行为单位
done


ts=`date +%Y_%m_%d_%H_%M`
logfolder=`pwd -P`/log/process.log.${ts}
mkdir -p ${logfolder}
for file in $(ls ./conf_dir/conf*json|awk -F'/' '{print $NF}')
do
    read -u6
    {
        echo ${file}
        # get ${params_str}
        conf_path=`pwd -P`/conf_dir/${file}
        model_dir=${parent_model_dir}/${file}
        export_model_dir=${parent_export_model_dir}/${file}
        mkdir -p ${logfolder}/${file}
        mkdir -p ${model_dir}
        mkdir -p ${export_model_dir}
        source ${easyctr_dir}/tools/train/train_params.sh
        declare -i rd=$RANDOM%10
        sleep $rd
        train_begin_ts=`date +%s`
        python ${easyctr_dir}/main.py ${params_str} \
            --do_train=${do_train} \
            --do_eval=${do_eval} \
            --do_export=${do_export} \
            --train_data_path=${train_data_path} \
            --eval_data_path=${eval_data_path} \
            --predict_data_path=${predict_data_path} \
            --predict_keys_path=${predict_keys_path} \
            &> ${logfolder}/${file}/log &
        wait
        # sleep 10
        cat conf_dir/${file} > ${logfolder}/${file}/conf
        train_end_ts=`date +%s`
        train_run_hours=`echo "scale=2;(${train_end_ts}-${train_begin_ts})/3600.0"|bc`
        echo ${file} ":" ${train_run_hours}
        sleep 2
        echo >&6
    } &
done
wait
rm -rf conf_dir/
exec 111>&-
exec 111<&-
echo "all done"
