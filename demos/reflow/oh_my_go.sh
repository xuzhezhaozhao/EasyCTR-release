#! /usr/bin/env bash

set -e

cd "$( dirname "${BASH_SOURCE[0]}" )"

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir
dict_dir=`pwd`/data/dict
mkdir -p ${dict_dir}
rm -rf ${model_dir}

export LD_LIBRARY_PATH="/home/xzz/anaconda2_tf1_14/lib/python2.7/site-packages/tensorflow/:$LD_LIBRARY_PATH"
../../build/tools/string_indexer/string_indexer ./data/data.meta ./data/train.txt ${dict_dir}

python data/conf.py > data/conf.json

python ../../main.py \
    --model_dir=${model_dir} \
    --export_model_dir=${export_model_dir} \
    --dict_dir=${dict_dir} \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --do_export=true \
    --train_data_path=./data/train_files.txt \
    --eval_data_path=./data/eval_files.txt \
    --predict_data_path=./data/predict_files.txt \
    --predict_keys_path=./data/predict_keys.txt \
    --predict_output_file=/tmp/predict_output.txt \
    --conf_path=./data/conf.json \
    --assembler_ops_path=../../build/ops/libassembler_ops.so \
    --batch_size=8 \
    --eval_batch_size=32 \
    --max_train_steps=-1 \
    --epoch=10 \
    --save_summary_steps=1 \
    --log_step_count_steps=10 \
    --serving_warmup_file=../../easyctr/tf_serving_warmup_requests \
    --predictor_warmup_file=../../easyctr/predictor_warmup_requests \
    --result_output_file=eval_result.log \
    --loss_fn=mape \
    --huber_loss_delta=1.5 \
    --mape_loss_delta=1.5 \
    --label_fn=log \
    --run_mode=easyctr \
    --model_type=multi_dnn \
    --multi_dnn_shared_hidden_units='' \
    --multi_dnn_use_shared_embedding=false \
    --multi_dnn_tower_use_shared_embedding=false \
    --dnn_l2_regularizer=0.0 \
    --deepx_mode='regressor' \
    --deep_optimizer=adagrad \
    --deep_learning_rate=0.0001 \
    --use_lua_sampler=false \
    --lua_sampler_script=./test/lua/sampler.v1.lua \
    --map_num_parallel_calls=1
