
source ../conf/model.conf
model_dir=export_model_dir

ts=`date +%Y%m%d%H%M%S`

s=$(dirname `pwd -P`)
pkg_name=$(basename $s)
ln -sfT /usr/local/services/kd_tools_easy_ctr-1.0/tools/tensorflow_model_server_EasyCTR ${pkg_name}

# 创建模型目录
cd ../data/
ext_dir=/data/${pkg_name}/data/
echo ${ext_dir}
mkdir -p ${ext_dir}
rm -rf data
ln -fs ${ext_dir} data
mkdir -p data/export_model_dir
cd ../bin

./${pkg_name} \
    --max_num_load_retries=12 \
    --load_retry_interval_micros=300000000 \
    --model_name="easyctr" \
    --port=${port} \
    --file_system_poll_wait_seconds=60 \
    --model_base_path=$(pwd -P)/../data/data/${model_dir} > ../log/run_serving.log 2>& 1 &

bash ../data/rsync_export_model_dir.sh > ../log/start_rsync_model.log 2>&1 &
