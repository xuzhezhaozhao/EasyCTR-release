#! /usr/bin/env bash

set -e

cd "$( dirname "${BASH_SOURCE[0]}" )"

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir
dict_dir=`pwd`/data/dict
mkdir -p ${dict_dir}
rm -rf ${model_dir}

export LD_LIBRARY_PATH="/home/xzz/anaconda2_tf1_14/lib/python2.7/site-packages/tensorflow/:$LD_LIBRARY_PATH"
./build/tools/string_indexer/string_indexer ./data/data.meta ./data/train.txt ${dict_dir}

cd data/
python conf.py > conf.json
cd ..

python main.py \
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
    --assembler_ops_path=./build/ops/libassembler_ops.so \
    --batch_size=32 \
    --eval_batch_size=256 \
    --max_train_steps=-1 \
    --epoch=1 \
    --save_summary_steps=1 \
    --log_step_count_steps=1 \
    --serving_warmup_file=easyctr/tf_serving_warmup_requests \
    --predictor_warmup_file=easyctr/predictor_warmup_requests \
    --result_output_file=eval_result.log \
    --loss_fn=default \
    --label_fn=default \
    --run_mode=easyctr \
    --model_type=dssm \
    --dssm_mode=cosine \
    --dssm1_batch_norm=true \
    --dssm2_batch_norm=true \
    --dnn_l2_regularizer=0.0 \
    --deepx_mode='classifier' \
    --wkfm_use_selector=false \
    --fgcnn_use_project=true \
    --deep_learning_rate_decay_type=fixed \
    --deep_warmup_rate=0.3 \
    --deep_optimizer=adagrad \
    --deep_learning_rate=0.01 \
    --total_steps=100 \
    --fm_use_project=true \
    --fwfm_use_project=true \
    --afm_use_project=true \
    --iafm_use_project=true \
    --cin_use_project=true \
    --autoint_use_project=true \
    --nfm_use_project=true \
    --ccpm_use_project=true \
    --ipnn_use_project=true \
    --fibinet_use_project=true \
    --ifm_use_project=true \
    --use_seperated_logits=false \
    --use_weighted_logits=false \
    --auc_thr=0.0 \
    --use_negative_sampling=false \
    --nce_count=5 \
    --nce_items_path=./data/nce_items.txt \
    --use_lua_sampler=false \
    --lua_sampler_script=./test/lua/sampler.v1.lua \
    --map_num_parallel_calls=10
