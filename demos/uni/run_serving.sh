
killall -9 tensorflow_model_server_EasyCTR > /dev/null 2>&1
sleep 2
tensorflow_model_server_EasyCTR \
    --model_name="easyctr" \
    --port=9000 \
    --model_base_path=$(pwd)/export_model_dir > serving.log 2>&1 &
