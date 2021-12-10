
source ../conf/model.conf

s=$(dirname `pwd -P`)
pkg_name=$(basename $s)

ext_dir=/data/${pkg_name}/data/
echo ${ext_dir}
mkdir -p ${ext_dir}
rm -rf data
ln -fs ${ext_dir} data

rm -rf ./data/export_model_dir
mkdir -p ./data/rsync_dir
secret_file=/usr/local/services/${pkg_name}/data/rsync.secret
chmod 600 ${secret_file}
rsync -azv --delay-updates rsync://test@${rsync_addr}${rsync_file} ./data/rsync_dir --password-file=${secret_file}

ln -fs `pwd -P`/data/rsync_dir/export_model_dir ./data/export_model_dir
