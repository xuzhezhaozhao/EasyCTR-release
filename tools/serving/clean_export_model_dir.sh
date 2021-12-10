
source ../conf/model.conf

clean_dir=`pwd -P`/data/rsync_dir/export_model_dir/
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

# 若日志文件大于指定行，则截断
logfile=`pwd -P`/../log/run_serving.log
total_lines=$(wc -l ${logfile} | awk '{print $1}')
if [[ ${total_lines} -gt 10000 ]]; then
    cp ${logfile} ${logfile}.bak
    cat /dev/null > ${logfile}
fi

echo "clean ${clean_dir} OK"
