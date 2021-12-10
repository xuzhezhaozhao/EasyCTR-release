echo 'download hdfs://ss-sng-dc-v2/data/SPARK/SNG/g_sng_qqkd_sng_buluo_kd_video_group/easyctr/kd_tools_easy_ctr-1.0 ...'
hadoop fs -get hdfs://ss-sng-dc-v2/data/SPARK/SNG/g_sng_qqkd_sng_buluo_kd_video_group/easyctr/kd_tools_easy_ctr-1.0 ./
echo 'run kd_tools_easy_ctr-1.0/tools/train/tesla_train.sh ...'
bash kd_tools_easy_ctr-1.0/tools/train/tesla_train.sh
