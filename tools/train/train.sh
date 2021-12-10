#! /usr/bin/env bash

source /data1/11090252/common/anaconda2/anaconda2_profile >/dev/null 2>&1
source /data1/11090252/common/anaconda2_gpu/anaconda2_profile >/dev/null 2>&1

set -e

ts=`date +%Y_%m_%d_%H_%M`

log_dir=data/log
mkdir -p ${log_dir}

begin_ts=`date +%s`

# 引入配置变量
source /data1/11090252/common/easyctr/tools/train/model.conf.default
source model.conf
echo "************** [script-begin] model.conf ********************"
cat model.conf
echo "************** [script-end] model.conf ********************"

echo "************** [script-begin] train.sh ********************"
cat ${BASH_SOURCE[0]}
echo "************** [script-end] train.sh ********************"

# 保存当前 model.conf
old_model_conf=${log_dir}/model.conf.old
touch ${old_model_conf}
current_model_conf=${log_dir}/model.conf.current
cp model.conf ${current_model_conf}

num_train_data=${num_train_data}  # 训练数据
num_train_data_expired=${num_train_data_expired}  # 训练数据过期时间

easyctr_dir=/data1/11090252/common/easyctr/
data_dir=`pwd -P`/data/data
database_dir=`pwd -P`/data/database
export_model_dir=`pwd -P`/data/export_model_dir
model_dir=`pwd -P`/data/model_dir
dict_dir=`pwd -P`/data/data/
mkdir -p ${data_dir}

# 检查是否是增量训练模式
incremental_training_check_file=${data_dir}/use_incremental_training.check
if [[ ${use_incremental_training} != 'true' ]]; then
    rm -rf ${incremental_training_check_file}
fi
if [[ ! -f ${incremental_training_check_file} && ${remove_model_dir} == 'true' ]]; then
    rm -rf ${model_dir}.bak
    if [[ -d ${model_dir} ]]; then
        mv ${model_dir} ${model_dir}.bak
    fi
fi
mkdir -p ${model_dir}

# 清理 tensorboard 需要的 events 文件，因为在增量模式下这个文件会一直累积
clean_dir=${model_dir}
keep_max=5
keep=0
files=$(ls ${clean_dir}/events.out.* 2>/dev/null | sort -r)
for file in ${files}
do
    keep=$((keep+1))
    echo "check ${file} ..."
    if [[ ${keep} -gt ${keep_max} ]]; then
        echo "rm -rf ${file} ..."
        rm -rf ${file}
    fi
done
clean_dir=${model_dir}/eval
keep_max=5
keep=0
files=$(ls ${clean_dir}/events.out.* 2>/dev/null | sort -r)
for file in ${files}
do
    keep=$((keep+1))
    echo "check ${file} ..."
    if [[ ${keep} -gt ${keep_max} ]]; then
        echo "rm -rf ${file} ..."
        rm -rf ${file}
    fi
done

# 创建 conf.py 软链接
ln -sf ${conf_script} conf.py

# 增量模式下测试数据只能用 1 小时
if [[ ${use_incremental_training} == 'true' && ${num_test_data} != 1 ]]; then
    echo "error: use_incremental_training is true but num_test_data is not 1"
    exit -1
fi
if [[ ${use_incremental_training} == 'true' && ${max_eval_steps} -lt 0 ]]; then
    max_eval_steps=1000
fi

# 聚合训练数据
merge_data_begin_ts=`date +%s`
if [[ ${do_merge_data} == 'true' ]]; then
    echo "merge data ..."
    rm -rf ${data_dir}/train_files.txt
    rm -rf ${data_dir}/eval_files.txt
    python ${merge_data_python_script} \
        ${database_dir} \
        ${num_train_data} \
        ${data_dir} \
        ${num_train_data_expired} \
        ${shuffle_data} \
        ${test_data_type} \
        ${num_test_data} \
        ${use_spark_fuel} \
        ${use_incremental_training} \
        ${skip_num_thr}
    echo "merge data done"
else
    echo "Do not merge data, use old version"
fi

# easyrecall 或者 负采样模式需要下载额外的候选数据
if [[ (${run_mode} == 'easyrecall' || ${use_negative_sampling} == 'true') && ${do_download_candidata_items} == 'true' ]]; then
    echo "download candidate items ..."
    bash ${easyrecall_download_candidata_items_script}
    echo "download candidate items done"
fi

# 必须先merge data再执行conf.py，因为shared embedding需要用到词典
generate_conf_begin_ts=`date +%s`
if [[ ${do_generate_conf} == 'true' ]]; then
    echo "generate conf ..."
    python conf.py > conf.json
    echo "generate conf done"
else
    echo "Do not need generate conf, use old version"
fi
generate_conf_end_ts=`date +%s`
generate_conf_run_hours=`echo "scale=2;(${generate_conf_end_ts}-${generate_conf_begin_ts})/3600.0"|bc`

merge_data_end_ts=`date +%s`
merge_data_run_hours=`echo "scale=2;(${merge_data_end_ts}-${merge_data_begin_ts})/3600.0"|bc`

# train model
if [[ ${use_adagrad_shrinkage} == 'true' ]]; then
    echo "do adagrad shrinkage ..."
    python ${easyctr_dir}/tools/adagrad_shrinkage.py ${model_dir} ${adagrad_shrinkage_rate}
    echo "do adagrad shrinkage done"
fi

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
do_train=${do_train}
do_eval=${do_eval}
do_predict=${do_predict}
do_export=${do_export}
train_data_path=${data_dir}/train_files.txt
eval_data_path=${data_dir}/eval_files.txt

if [[ ${test_data_type} == 'last' ]]; then
    do_export=false
fi

use_tfrecord=false
conf_path=`pwd -P`/conf.json
assembler_ops_path=${easyctr_dir}/ops/libassembler_ops.so
serving_warmup_file=${easyctr_dir}/easyctr/tf_serving_warmup_requests
predictor_warmup_file=${easyctr_dir}/easyctr/predictor_warmup_requests
if [[ ${run_mode} == 'easyrecall' ]]; then
    serving_warmup_file=${easyctr_dir}/easyctr/tf_serving_warmup_requests_easyrecall
    predictor_warmup_file=${easyctr_dir}/easyctr/predictor_warmup_requests_easyrecall
fi

# get ${params_str}
source ${easyctr_dir}/tools/train/train_params.sh
train_begin_ts=`date +%s`
python ${easyctr_dir}/main.py ${params_str} \
    --do_train=${do_train} \
    --do_eval=${do_eval} \
    --do_predict=${do_predict} \
    --do_export=${do_export} \
    --train_data_path=${train_data_path} \
    --eval_data_path=${eval_data_path} \
    --predict_data_path=${predict_data_path} \
    --predict_keys_path=${predict_keys_path} \
    --predict_output_file=${predict_output_file}
train_end_ts=`date +%s`
train_run_hours=`echo "scale=2;(${train_end_ts}-${train_begin_ts})/3600.0"|bc`

plus_train_run_hours='None'
if [[ ${test_data_type} == 'last' ]]; then
    # 如果使用最后若干数据作为测试集合，可以选择是否重新将测试数据拿来训练
    if [[ ${do_plus_train} == 'false' && ${use_incremental_training} == 'true' ]]; then
        echo 'error: use_incremental_training = true but do_plus_train = false'
        exit -1
    fi

    if [[ ${do_plus_train} == 'true' && ${do_train} == 'true' ]]; then
        do_train=true
    else
        do_train=false
    fi
    do_eval=false
    do_predict=false
    do_export=true
    train_data_path=${eval_data_path}
    plus_train_begin_ts=`date +%s`
    python ${easyctr_dir}/main.py ${params_str} \
        --do_train=${do_train} \
        --do_eval=${do_eval} \
        --do_predict=${do_predict} \
        --do_export=${do_export} \
        --train_data_path=${train_data_path} \
        --eval_data_path=${eval_data_path} \
        --predict_data_path=${predict_data_path} \
        --predict_keys_path=${predict_keys_path} \
        --predict_output_file=${predict_output_file}
    plus_train_end_ts=`date +%s`
    plus_train_run_hours=`echo "scale=2;(${plus_train_end_ts}-${plus_train_begin_ts})/3600.0"|bc`
fi

# 清理导出模型目录
clean_dir=${export_model_dir}
rm -rf ${clean_dir}/temp-*
keep_max=3
echo "clean ${clean_dir} ..."
echo "keep_max = ${keep_max}"
keep=0
files=$(ls ${clean_dir} | sort -r)
for file in ${files}
do
    keep=$((keep+1))
    echo "check ${clean_dir}/${file} ..."
    if [[ ${keep} -gt ${keep_max} ]]; then
        echo "rm -rf ${clean_dir}/${file} ..."
        rm -rf ${clean_dir}/${file}
    fi
done
echo "clean ${clean_dir} OK"

# 推送模型到 hdfs
hadoop_bin="hadoop fs "
s=$(dirname `pwd -P`)
push_model_run_hours='None'
if [[ ${do_push_to_hdfs} == 'true' ]]; then
    echo 'push model to hdfs ...'
    push_model_begin_ts=`date +%s`
    # 将导出模型推送到 hdfs
    target_dir=${export_model_hdfs_basedir}/${model_name}
    latest_model=`ls -d ${export_model_dir}/1* | sort | tail -n1`
    latest_model_basename=`basename ${latest_model}`

    echo "hdfs target dir = ${target_dir}"

    ${hadoop_bin} -mkdir -p ${target_dir}
    ${hadoop_bin} -put ${latest_model} ${target_dir}/tmp_${latest_model_basename}
    ${hadoop_bin} -mv ${target_dir}/tmp_${latest_model_basename} ${target_dir}/${latest_model_basename}
    ${hadoop_bin} -chmod 777 ${target_dir}
    ${hadoop_bin} -chmod -R 777 ${target_dir}/${latest_model_basename}
    echo "hdfs export model dir = ${target_dir}/${latest_model_basename}"

    # 清理 hdfs 模型目录
    # bash /usr/local/services/kd_tools_easy_ctr-1.0/tools/train/hdfs_remove_outdate.sh ${target_dir} ${export_model_num_outdate_days}

    push_model_end_ts=`date +%s`
    push_model_run_hours=`echo "scale=2;(${push_model_end_ts}-${push_model_begin_ts})/3600.0"|bc`
    echo 'push model to hdfs done'
fi

end_ts=`date +%s`
total_run_hours=`echo "scale=2;(${end_ts}-${begin_ts})/3600.0"|bc`

tmp_eval_result_file=`pwd`/data/log/eval_result.log.tmp
eval_result_file=`pwd`/data/log/eval_result.log
rm -rf ${tmp_eval_result_file}
cat ${result_output_file} > ${tmp_eval_result_file}
# calc diff
echo 'calc diff ...'
python ${easyctr_dir}/tools/misc/calc_diff.py \
    ${predict_output_file} \
    $(cat ${eval_data_path}) \
    ${calc_diff_extra_id} \
    ${predict_and_target_file} \
    ${hive_media_type} | tee -a ${tmp_eval_result_file}
echo 'calc diff done'

cp ${tmp_eval_result_file} ${eval_result_file}
cp ${eval_result_file} ${eval_result_file}.${ts}

ip=`/usr/sbin/ifconfig | grep -A 1 eth1 | tail -n 1 | awk '{print $2}'`
if [[ ${msg_title} == '' ]]; then
    msg_title=${model_name}
fi
msg_file=${log_dir}/msg.txt

rm -rf ${msg_file}
touch ${msg_file}
set +e
rc=`diff ${old_model_conf} ${current_model_conf}`
set -e
if [[ ${rc} != '' ]]; then
    echo ${msg_title} >> ${msg_file}
    echo ${ip} >> ${msg_file}
    echo '' >> ${msg_file}
    cat ${current_model_conf} >> ${msg_file}
    echo '' >> ${msg_file}
    echo '----------------------------' >> ${msg_file}

    cp ${current_model_conf} ${old_model_conf}
fi

echo ${msg_title} >> ${msg_file}
echo ${ip} >> ${msg_file}
echo '' >> ${msg_file}
echo 'merge_data_run_hours =' ${merge_data_run_hours} | tee -a ${msg_file}
echo 'generate_conf_run_hours =' ${generate_conf_run_hours} | tee -a ${msg_file}
echo 'train_run_hours =' ${train_run_hours} | tee -a ${msg_file}
echo 'plus_train_run_hours =' ${plus_train_run_hours} | tee -a ${msg_file}
echo 'push_model_run_hours =' ${push_model_run_hours} | tee -a ${msg_file}
echo 'total_run_hours =' ${total_run_hours} | tee -a ${msg_file}
echo '' | tee -a ${msg_file}
echo 'total_train_lines =' ${total_lines} | tee -a ${msg_file}
echo 'total_train_size_G =' $(( total_size_mb/1024 )) | tee -a ${msg_file}
echo '' >> ${msg_file}
cat ${eval_result_file} >> ${msg_file}
echo '----------------------------' >> ${msg_file}

echo 'send msg ...'
# TODO(zhifeng)
echo 'send msg done'

echo '------------------------'
echo 'send msg:'
cat ${msg_file}
echo '------------------------'
