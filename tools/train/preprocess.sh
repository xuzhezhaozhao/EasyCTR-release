#! /usr/bin/env bash

source /usr/local/services/kd_anaconda2-1.0/anaconda2_profile >/dev/null 2>&1
source /usr/local/services/kd_anaconda2_gpu-1.0/anaconda2_profile >/dev/null 2>&1

set -e

source /usr/local/services/kd_tools_easy_ctr-1.0/tools/train/model.conf.default
source model.conf

hadoop_bin="/usr/local/services/tdw_hdfs_client-1.0/bin/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_qqkd_sng_buluo_kd_video_group "
string_index_bin='/usr/local/services/kd_tools_easy_ctr-1.0/tools/string_indexer '
pigz_bin=/usr/local/services/kd_tools_easy_ctr-1.0/tools/pigz

database_dir=$1
p=$2
curdir=$3

# grep -E '^1\||^0\|' ${curdir}/data.txt.raw > ${curdir}/data.txt
# 添加 python import 路径
export PYTHONPATH=/usr/local/services/kd_tools_easy_ctr-1.0/tools/train/:$PYTHONPATH

# download data
mkdir -p ${curdir}/in
bash /usr/local/services/kd_tools_easy_ctr-1.0/tools/train/pull_data_from_hdfs.sh \
    ${p}/in/ \
    ${curdir}/in \
    ${hdfs_download_thread}

if [[ ${generate_samples_script} != '' ]]; then
    if [[ ${use_optimized_gs_script} == 'true' ]]; then
        cat ${curdir}/in/attempt_* | python ${generate_samples_script} \
            ./data.meta \
            ${curdir}/data.txt.raw \
            ${curdir}/data.txt.tmp \
            ${curdir}/label_stats.txt
        rm -rf ${curdir}/in/attempt_*
    else
        cat ${curdir}/in/attempt_* > ${curdir}/data.txt.raw
        rm -rf ${curdir}/in/attempt_*
        python ${generate_samples_script} \
            ./data.meta \
            ${curdir}/data.txt.raw \
            ${curdir}/data.txt.tmp \
            ${curdir}/label_stats.txt
        rm -rf ${curdir}/data.txt.raw
    fi
    echo -e ${generate_samples_script} '\n\n' > ${curdir}/generate_samples_script.txt
    cat ${generate_samples_script} >> ${curdir}/generate_samples_script.txt
else
    cat ${curdir}/in/attempt_* > ${curdir}/data.txt.tmp
    rm -rf ${curdir}/in/attempt_*
fi

if [[ ${shuffle_data_in_hour} == 'true' ]]; then
    shuf ${curdir}/data.txt.tmp -o ${curdir}/data.txt
    rm -rf ${curdir}/data.txt.tmp
else
    mv ${curdir}/data.txt.tmp ${curdir}/data.txt
fi

# split train and eval data
total_lines=$(wc -l ${curdir}/data.txt | awk '{print $1}')
if [[ ${total_lines} -le ${min_line_count_hourly} ]]; then
    echo "Download data too small, less than ${min_line_count_hourly} lines, datadir = ${curdir}"
    exit -1
fi

if [[ ${test_data_type} == 'average' ]]; then
    train_lines=`echo "scale=2;${total_lines}*0.95"|bc|awk '{print int($1)}'`
    test_lines=`echo "scale=2;${total_lines}*0.05"|bc|awk '{print int($1)}'`
    head ${curdir}/data.txt -n ${train_lines} > ${curdir}/train.txt
    tail ${curdir}/data.txt -n ${test_lines} > ${curdir}/eval.txt
    rm -rf ${curdir}/data.txt
else
    train_lines=${total_lines}
    mv ${curdir}/data.txt ${curdir}/train.txt
fi

if [[ ${do_negative_sampling} == 'true' ]]; then
    mv ${curdir}/train.txt ${curdir}/train.txt.tmp
    python /usr/local/services/kd_tools_easy_ctr-1.0/tools/train/negative_sampling.py \
        ${curdir}/train.txt.tmp ${curdir}/train.txt ${num_negative_sample}
    rm -rf ${curdir}/train.txt.tmp
    train_lines=$(wc -l ${curdir}/train.txt | awk '{print $1}')
fi

echo ${train_lines} > ${curdir}/train_lines.txt

# string index
if [[ ${use_incremental_training} == 'false' && ${do_string_index} == 'true' ]]; then
    ${string_index_bin} ./data.meta ${curdir}/train.txt ${curdir}
else
    echo "Do not do string index"
fi

# shuffle 或者压缩数据
if [[ -f ${curdir}/eval.txt ]]; then
    if [[ ${compression_type} == 'GZIP' ]]; then
        echo 'pigz eval.txt ...'
        time ${pigz_bin} -1 -p ${num_compression_thread} ${curdir}/eval.txt
        echo 'pigz eval.txt done'
    fi
fi

if [[ ${shuffle_data} == 'true' || ${use_spark_fuel} == 'true' ]]; then
    # 将 train.txt 分成小份
    split -a 3 -d -l ${split_lines_per_file} ${curdir}/train.txt ${curdir}/train.txt.
    rm -rf ${curdir}/train.txt
    if [[ ${compression_type} == 'GZIP' ]]; then
        echo 'pigz train.txt ...'
        time ls ${curdir}/train.txt.* | xargs ${pigz_bin} -1 -p ${num_compression_thread}
        echo 'pigz train.txt done'
    fi
else
    mv ${curdir}/train.txt ${curdir}/train.txt.000
    if [[ ${compression_type} == 'GZIP' ]]; then
        echo 'pigz train.txt ...'
        time ${pigz_bin} -1 -p ${num_compression_thread} ${curdir}/train.txt.000
        echo 'pigz train.txt done'
    fi
fi

basename=`basename ${curdir}`
rm -rf ${curdir}/hdfs_train_files.txt
rm -rf ${curdir}/hdfs_eval_files.txt
if [[ ${use_spark_fuel} == 'true' ]]; then
    if [[ ${sf_hdfs_root_dir} == '' ]]; then
        echo 'sf_hdfs_root_dir should not be empty'
        exit -1
    fi
    for f in `find ${curdir} -name "train.txt*" | xargs -i basename {} | sort`
    do
        echo "${sf_hdfs_root_dir}/database/${basename}/${f}" >> ${curdir}/hdfs_train_files.txt
    done
    for f in `find ${curdir} -name "eval.txt*" | xargs -i basename {} | sort`
    do
        echo "${sf_hdfs_root_dir}/database/${basename}/${f}" >> ${curdir}/hdfs_eval_files.txt
    done

    # push current dir to hdfs
    echo "push ${curdir} to ${sf_hdfs_root_dir}/database ..."
    ${hadoop_bin} -mkdir -p ${sf_hdfs_root_dir}/database
    ${hadoop_bin} -rm -r -f ${sf_hdfs_root_dir}/database/${basename}
    ${hadoop_bin} -put ${curdir} ${sf_hdfs_root_dir}/database
    echo "push ${curdir} to ${sf_hdfs_root_dir}/database done"

    # delete data but keep dict files
    rm -rf ${curdir}/train.txt*
    rm -rf ${curdir}/eval.txt*
fi
