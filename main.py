#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf
import numpy as np

from common import transform
from common.common import get_file_content
from common.common import create_profile_hooks
from common.common import parse_scheme

from easyctr import input_fn
from easyctr.dssm import dssm
from easyctr.essm import essm
from easyctr.essm import doubletower
from easyctr.deepx import deepx
from easyctr.flags import parser
from easyctr.configure_lr_and_opt import get_optimizer
from easyctr.configure_loss_fn import configure_loss_fn
from easyctr.configure_extra_options import get_extra_options
from easyctr.model_slots_map import MODEL_SLOTS_MAP

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'


def get_config(opts):
    config_keys = {}
    config_keys['model_dir'] = opts.model_dir
    config_keys['tf_random_seed'] = None
    config_keys['save_summary_steps'] = opts.save_summary_steps
    config_keys['save_checkpoints_steps'] = opts.save_checkpoints_steps
    config_keys['session_config'] = None
    config_keys['keep_checkpoint_max'] = opts.keep_checkpoint_max
    config_keys['keep_checkpoint_every_n_hours'] = 10000
    config_keys['log_step_count_steps'] = opts.log_step_count_steps
    if opts.num_gpus > 1:
        distribution = tf.contrib.distribute.MirroredStrategy(
            num_gpus=opts.num_gpus)
        config_keys['train_distribute'] = distribution
    run_config = tf.estimator.RunConfig(**config_keys)

    return run_config


def build_estimator(opts, scheme, conf, item_embeddings=None, item_keys=None):
    trans = transform.Transform(conf, scheme)

    group_columns = trans.group_columns

    wide_columns = group_columns.get('wide', [])
    deep_columns = group_columns.get('deep', [])
    dssm1_columns = group_columns.get('dssm1', [])
    dssm2_columns = group_columns.get('dssm2', [])
    selector_columns = group_columns.get('selector', [])

    run_config = get_config(opts)
    wide_optimizer, deep_optimizer = get_optimizer(opts)
    loss_fn = configure_loss_fn(opts)
    if opts.loss_reduction == 'mean':
        loss_reduction = tf.losses.Reduction.MEAN
    elif opts.loss_reduction == 'sum':
        loss_reduction = tf.losses.Reduction.SUM
    else:
        raise ValueError("Unknown loss_reduction '{}'".format(loss_reduction))

    extra_options = get_extra_options(opts)
    if opts.run_mode == 'easyrecall':
        extra_options['item_embeddings'] = item_embeddings
        extra_options['item_keys'] = item_keys

    model_type = opts.model_type
    model_slots = opts.model_slots

    if model_type in MODEL_SLOTS_MAP:
        model_slots = MODEL_SLOTS_MAP[model_type]
        model_type = 'deepx'

    if model_type == 'dssm':
        estimator = dssm.DSSMEstimator(
            dssm1_columns=dssm1_columns,
            dssm2_columns=dssm2_columns,
            model_dir=opts.model_dir,
            n_classes=2,
            weight_column=input_fn.WEIGHT_COL,
            optimizer=deep_optimizer,
            config=run_config,
            loss_fn=loss_fn,
            loss_reduction=loss_reduction,
            run_mode=opts.run_mode,
            extra_options=extra_options)
    elif model_type == 'deepx':
        assert opts.run_mode == 'easyctr', "'deepx' only supports 'easyctr' mode"
        if opts.deepx_mode == 'classifier':
            estimator = deepx.DeepXClassifier(
                model_dir=opts.model_dir,
                linear_feature_columns=wide_columns,
                linear_optimizer=wide_optimizer,
                deep_feature_columns=deep_columns,
                deep_optimizer=deep_optimizer,
                selector_feature_columns=selector_columns,
                model_slots=model_slots,
                extend_feature_mode=opts.extend_feature_mode,
                n_classes=2,
                weight_column=input_fn.WEIGHT_COL,
                config=run_config,
                linear_sparse_combiner=opts.linear_sparse_combiner,
                loss_fn=loss_fn,
                loss_reduction=loss_reduction,
                use_seperated_logits=opts.use_seperated_logits,
                use_weighted_logits=opts.use_weighted_logits,
                extra_options=extra_options)
        elif opts.deepx_mode == 'regressor':
            estimator = deepx.DeepXRegressor(
                model_dir=opts.model_dir,
                linear_feature_columns=wide_columns,
                linear_optimizer=wide_optimizer,
                deep_feature_columns=deep_columns,
                deep_optimizer=deep_optimizer,
                selector_feature_columns=selector_columns,
                model_slots=model_slots,
                extend_feature_mode=opts.extend_feature_mode,
                weight_column=input_fn.WEIGHT_COL,
                config=run_config,
                linear_sparse_combiner=opts.linear_sparse_combiner,
                loss_fn=loss_fn,
                loss_reduction=loss_reduction,
                use_seperated_logits=opts.use_seperated_logits,
                use_weighted_logits=opts.use_weighted_logits,
                extra_options=extra_options)
        else:
            raise ValueError("Unknow 'deepx_mode' '{}'".format(opts.deepx_mode))
    elif model_type == 'essm':
        estimator = essm.ESSMEstimator(
            group_columns=group_columns,
            model_dir=opts.model_dir,
            n_classes=2,
            weight_column=input_fn.WEIGHT_COL,
            optimizer=deep_optimizer,
            config=run_config,
            loss_fn=loss_fn,
            loss_reduction=loss_reduction,
            run_mode=opts.run_mode,
            extra_options=extra_options)
    elif model_type == 'DtowerEstimator':
        estimator = doubletower.DtowerEstimator(
            group_columns=group_columns,
            model_dir=opts.model_dir,
            n_classes=2,
            weight_column=input_fn.WEIGHT_COL,
            optimizer=deep_optimizer,
            config=run_config,
            loss_fn=loss_fn,
            loss_reduction=loss_reduction,
            run_mode=opts.run_mode,
            extra_options=extra_options)
    elif model_type == 'GBDT_REG':
        estimator = tf.estimator.BoostedTreesRegressor(
            feature_columns=wide_columns,
            n_batches_per_layer=100,
            model_dir=None,
            label_dimension=1,
            weight_column=None,
            n_trees=100,
            max_depth=6,
            learning_rate=0.1,
            l1_regularization=0.0,
            l2_regularization=0.0,
            tree_complexity=0.0,
            min_node_weight=0.0,
            config=None,
            center_bias=False,
            pruning_mode='none',
            quantile_sketch_epsilon=0.01,
            train_in_memory=False)
    else:
        raise ValueError("Unknow model_type '{}'".format(opts.model_type))

    tf.logging.info("estimator: {}".format(estimator))
    return estimator


def do_train(opts, scheme, conf, train_files):
    tf.logging.info("Training model ...")
    estimator = build_estimator(opts, scheme, conf)
    max_train_steps = None if opts.max_train_steps < 0 else opts.max_train_steps
    train_input_fn = input_fn.input_fn(
        opts,
        filenames=train_files,
        is_eval=False,
        is_predict=False,
        scheme=scheme,
        epoch=opts.epoch,
        use_negative_sampling=opts.use_negative_sampling)
    hooks = None
    if opts.do_profile:
        hooks = create_profile_hooks(opts.profile_every_steps, opts.model_dir)

    estimator.train(
        input_fn=train_input_fn,
        max_steps=max_train_steps,
        hooks=hooks)
    tf.logging.info("Training model done")


def do_evaluate(opts, scheme, conf, eval_files):
    tf.logging.info("Evaluating model ...")
    estimator = build_estimator(opts, scheme, conf)
    eval_input_fn = input_fn.input_fn(
        opts,
        filenames=eval_files,
        is_eval=True,
        is_predict=False,
        scheme=scheme)
    eval_steps = opts.max_eval_steps if opts.max_eval_steps >= 0 else None
    result = estimator.evaluate(input_fn=eval_input_fn,
                                steps=eval_steps)

    if 'auc' in result:
        if result['auc'] < opts.auc_thr:
            raise ValueError("oops, auc less than {}".format(opts.auc_thr))

    tf.logging.info("evaluate result:")
    with open(opts.result_output_file, 'w') as f:
        for key in result:
            try:
                value = float(result[key])
            except Exception:
                value = result[key]
            if isinstance(value, float):
                value = format(value, '.7g')
            s = '{} = {}'.format(key, value)
            tf.logging.info(s)
            f.write(s + '\n')

    tf.logging.info("Evaluating model done")


def do_easyrecall_predict(opts, scheme, conf, predict_files, predict_keys):
    assert opts.run_mode == 'easyrecall', \
        "'do_easyrecall_predict' only support 'easyrecall'"
    tf.logging.info("EasyRecall predict ...")
    estimator = build_estimator(opts, scheme, conf)
    predict_input_fn = input_fn.input_fn(
        opts,
        filenames=predict_files,
        is_eval=False,
        is_predict=True,
        scheme=scheme)

    checkpoint_path = tf.train.latest_checkpoint(opts.model_dir)
    results = estimator.predict(
        input_fn=predict_input_fn,
        checkpoint_path=checkpoint_path,
        yield_single_examples=False)
    data = None
    total = len(predict_keys)
    index = 0
    for result in results:
        if data is None:
            dim = result['item_embeddings'].shape[1]
            data = np.zeros([total, dim], dtype=np.float32)
        batch_size = result['item_embeddings'].shape[0]
        data[index:index+batch_size, :] = result['item_embeddings']
        index += batch_size
    assert index == total, "predict error, keys and predict size not matched"
    tf.logging.info("EasyRecall predict data: {}".format(data))
    tf.logging.info("EasyRecall predict data shape: {}".format(data.shape))
    tf.logging.info("EasyRecall predict done")

    return data.transpose()


def do_predict(opts, scheme, conf, predict_files, predict_keys):
    tf.logging.info("Predict ...")
    estimator = build_estimator(opts, scheme, conf)
    predict_input_fn = input_fn.input_fn(
        opts,
        filenames=predict_files,
        is_eval=False,
        is_predict=True,
        scheme=scheme)
    checkpoint_path = tf.train.latest_checkpoint(opts.model_dir)
    results = estimator.predict(
        input_fn=predict_input_fn,
        checkpoint_path=checkpoint_path,
        yield_single_examples=True)
    with open(opts.predict_output_file, 'w') as f:
        idx = 0
        for result in results:
            j = {key: result[key].tolist() for key in result}

            if predict_keys is not None:
                if idx >= len(predict_keys):
                    raise ValueError("keys and predictions size not matched")
                KEY_ID = '__ID__'
                if KEY_ID in j:
                    raise ValueError("error: KEY_ID '{}' in result".format(KEY_ID))
                j[KEY_ID] = predict_keys[idx]
                idx += 1

            j = json.dumps(j)
            f.write(j)
            f.write('\n')
    tf.logging.info("Predict done")


def do_export(opts, scheme, conf, item_embeddings=None, keys=None):
    tf.logging.info("Export model ...")
    estimator = build_estimator(opts, scheme, conf, item_embeddings, keys)
    assets_extra = {}
    assets_extra['tf_serving_warmup_requests'] = opts.serving_warmup_file
    assets_extra['predictor_warmup_requests'] = opts.predictor_warmup_file
    assets_extra['conf_file'] = opts.conf_path
    assets_extra['result_output_file'] = opts.result_output_file

    if opts.run_mode == 'easyctr' and opts.model_type == 'dssm':  # 粗排
        assets_extra['dssm_output.txt'] = opts.predict_output_file

    if opts.dict_dir != '':
        dict_files = [line for line in os.listdir(opts.dict_dir)
                      if line.endswith('.dict')]
        for dict_file in dict_files:
            assets_extra[dict_file] = os.path.join(opts.dict_dir, dict_file)

    export_path = estimator.export_saved_model(
        opts.export_model_dir,
        serving_input_receiver_fn=input_fn.build_serving_input_fn(opts, scheme),
        assets_extra=assets_extra)
    tf.logging.info("Export model done, export path '{}'".format(export_path))


def do_spark_fuel_run(opts, scheme, conf, train_files, eval_files, predict_files,
                      predict_keys):
    assert opts.run_mode == 'easyctr', "'sparkfuel' not support 'easyrecall'"
    estimator = build_estimator(opts, scheme, conf)
    train_input_fn = input_fn.input_fn(
        opts,
        filenames=train_files,
        is_eval=False,
        is_predict=False,
        scheme=scheme,
        epoch=opts.epoch,
        use_negative_sampling=opts.use_negative_sampling)

    eval_input_fn = input_fn.input_fn(
        opts,
        filenames=eval_files,
        is_eval=True,
        is_predict=False,
        scheme=scheme)

    if opts.do_train:
        # TODO(zhezhaoxu) Officially, train_and_eval only support max_steps stop
        # condition
        max_train_steps = None if opts.max_train_steps < 0 else opts.max_train_steps
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                            max_steps=max_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                          steps=opts.eval_steps_every_checkpoint,
                                          exporters=None,
                                          start_delay_secs=opts.start_delay_secs,
                                          throttle_secs=opts.throttle_secs)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if opts.do_eval:
        import sparkfuel as sf
        job_type, _ = sf.get_tf_identity()
        tf.logging.info("job_type = {}".format(job_type))
        if job_type == 'chief':
            tf.logging.info("Evaluating test dataset ...")
            eval_input_fn = input_fn.input_fn(
                opts,
                filenames=eval_files,
                is_eval=True,
                is_predict=False,
                scheme=scheme)
            estimator.evaluate(input_fn=eval_input_fn)
            tf.logging.info("Evaluating test dataset done")
        else:
            tf.logging.info("Do not do evaluation in non-chief node")

    if opts.do_export:
        import sparkfuel as sf
        job_type, _ = sf.get_tf_identity()
        tf.logging.info("job_type = {}".format(job_type))
        if job_type == 'chief':
            tf.logging.info("Exporting model ...")
            assets_extra = {}
            assets_extra['tf_serving_warmup_requests'] = opts.serving_warmup_file
            serving_fn = input_fn.build_serving_input_fn(opts, scheme)
            export_path = estimator.export_saved_model(
                opts.export_model_dir,
                serving_input_receiver_fn=serving_fn,
                assets_extra=assets_extra)
            tf.logging.info("Export model done, export path '{}'"
                            .format(export_path))
        else:
            tf.logging.info("Do not do export in non-chief node")


def do_normal_run(opts, scheme, conf, train_files, eval_files, predict_files,
                  predict_keys):
    if opts.do_train:
        do_train(opts, scheme, conf, train_files)
    if opts.do_eval:
        do_evaluate(opts, scheme, conf, eval_files)
    if opts.do_predict:
        do_predict(opts, scheme, conf, predict_files, predict_keys)
    if opts.do_export:
        item_embeddings = None
        if opts.run_mode == 'easyrecall':
           item_embeddings = do_easyrecall_predict(opts, scheme, conf,
                                                   predict_files, predict_keys)
        do_export(opts, scheme, conf, item_embeddings, predict_keys)


def run(opts, conf, train_files, eval_files, predict_files, predict_keys):
    scheme = parse_scheme(
        conf_path=opts.conf_path,
        ops_path=opts.assembler_ops_path,
        use_spark_fuel=opts.use_spark_fuel)
    # 词典太大，不好打印
    # tf.logging.info("assembler scheme = \n{}".format(scheme))  #

    if opts.use_spark_fuel:
        # In distributed mode, we must use train_and_eval
        do_spark_fuel_run(opts, scheme, conf, train_files, eval_files,
                          predict_files, predict_keys)
    else:
        do_normal_run(opts, scheme, conf, train_files, eval_files,
                      predict_files, predict_keys)


def get_train_and_eval_files(opts):
    spark = None
    if opts.use_spark_fuel:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

    # we must read hdfs file before dive into spark-fuel
    conf = get_file_content(opts.conf_path, opts.use_spark_fuel, spark)
    conf = json.loads(conf)

    train_files = get_file_content(opts.train_data_path, opts.use_spark_fuel,
                                   spark)
    train_files = train_files.split('\n')
    train_files = [x.strip() for x in train_files if x.strip() != '']
    tf.logging.info("train files = {}".format(train_files))

    eval_files = get_file_content(opts.eval_data_path, opts.use_spark_fuel,
                                  spark)
    eval_files = eval_files.split('\n')
    eval_files = [x.strip() for x in eval_files if x.strip() != '']
    tf.logging.info("eval files = {}".format(eval_files))

    predict_files = None
    predict_keys = None
    if opts.do_predict or opts.run_mode == 'easyrecall':
        predict_files = get_file_content(opts.predict_data_path,
                                         opts.use_spark_fuel, spark)
        predict_files = predict_files.split('\n')
        predict_files = [x.strip() for x in predict_files if x.strip() != '']
        tf.logging.info("predict files = {}".format(predict_files))

    if opts.run_mode == 'easyrecall' or \
            (opts.do_predict and opts.model_type == 'dssm'):
        predict_keys = get_file_content(opts.predict_keys_path,
                                        opts.use_spark_fuel, spark)
        predict_keys = predict_keys.split('\n')
        predict_keys = [x.strip() for x in predict_keys if x.strip() != '']
        tf.logging.info("predict keys = {} ...".format(predict_keys[:10]))

    return train_files, eval_files, predict_files, predict_keys, conf


if __name__ == '__main__':
    opts = parser.parse_args()

    # 防止日志重复打印两次
    # https://stackoverflow.com/questions/33662648/tensorflow-causes-logging-messages-to-double
    logger = tf.get_logger()
    logger.propagate = False
    tf.logging.set_verbosity(tf.logging.INFO)

    (train_files, eval_files, predict_files,
     predict_keys, conf) = get_train_and_eval_files(opts)

    if opts.use_spark_fuel:
        import sparkfuel as sf
        from pyspark.sql import SparkSession
        from common.common import hdfs_remove_dir
        spark = SparkSession.builder.getOrCreate()
        if opts.remove_model_dir:
            succ = hdfs_remove_dir(spark, opts.model_dir)
            if succ:
                tf.logging.info("remove model dir '{}' done"
                                .format(opts.model_dir))
            else:
                tf.logging.info("remove model dir '{}' failed"
                                .format(opts.model_dir))
                raise RuntimeError("remove model dir failed")

        with sf.TFSparkSession(spark, num_ps=opts.num_ps,
                               with_eval=True) as sess:
            sess.run(run, opts, conf, train_files, eval_files, predict_files,
                     predict_keys)
    else:
        run(opts, conf, train_files, eval_files, predict_files, predict_keys)
