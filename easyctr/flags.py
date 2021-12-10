#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from common.define_flags import DEFINE_string
from common.define_flags import DEFINE_integer
from common.define_flags import DEFINE_float
from common.define_flags import DEFINE_bool
from common.define_flags import DEFINE_list


parser = argparse.ArgumentParser()

DEFINE_string(parser, 'run_mode', 'easyctr', 'easyctr, easyrecall')

DEFINE_string(parser, 'model_dir', '', '')
DEFINE_string(parser, 'export_model_dir', '', '')
DEFINE_string(parser, 'dict_dir', '', '')
DEFINE_bool(parser, 'use_tfrecord', False, '')
DEFINE_bool(parser, 'do_profile', False, '')
DEFINE_integer(parser, 'profile_every_steps', 1000, '')
DEFINE_string(parser, 'conf_path', '', 'conf path')
DEFINE_string(parser, 'assembler_ops_path', '', 'assembler_ops_path')

DEFINE_bool(parser, 'do_train', True, '')
DEFINE_bool(parser, 'do_eval', True, '')
DEFINE_bool(parser, 'do_predict', False, '')
DEFINE_bool(parser, 'do_export', True, '')
DEFINE_string(parser, 'train_data_path', '', 'train data path')
DEFINE_string(parser, 'eval_data_path', '', 'eval data path')
DEFINE_string(parser, 'predict_data_path', '', 'predict data path')
DEFINE_string(parser, 'predict_keys_path', '', 'predict keys path')
DEFINE_string(parser, 'predict_output_file', '', 'predict output file')

DEFINE_string(parser, 'model_type', 'wide_deep', '')
DEFINE_list(parser, 'model_slots', 'dnn', '')
DEFINE_string(parser, 'extend_feature_mode', '', 'fgcnn')
DEFINE_string(parser, 'deepx_mode', 'classifier', "'classifier' or 'regressor'")

# dssm
DEFINE_string(parser, 'dssm_mode', 'dot', 'dot, concat, cosine')
DEFINE_list(parser, 'dssm1_hidden_units', '512,256,128', '')
DEFINE_string(parser, 'dssm1_activation_fn', 'relu', '')
DEFINE_float(parser, 'dssm1_dropout', 0.0, '')
DEFINE_float(parser, 'dssm1_gaussian_dropout', 0.0, '')
DEFINE_bool(parser, 'dssm1_batch_norm', False, '')
DEFINE_bool(parser, 'dssm1_layer_norm', False, '')
DEFINE_bool(parser, 'dssm1_use_resnet', False, '')
DEFINE_bool(parser, 'dssm1_use_densenet', False, '')
DEFINE_list(parser, 'dssm2_hidden_units', '512,256,128', '')
DEFINE_string(parser, 'dssm2_activation_fn', 'relu', '')
DEFINE_float(parser, 'dssm2_dropout', 0.0, '')
DEFINE_float(parser, 'dssm2_gaussian_dropout', 0.0, '')
DEFINE_bool(parser, 'dssm2_batch_norm', False, '')
DEFINE_bool(parser, 'dssm2_layer_norm', False, '')
DEFINE_bool(parser, 'dssm2_use_resnet', False, '')
DEFINE_bool(parser, 'dssm2_use_densenet', False, '')
DEFINE_float(parser, 'dssm_cosine_gamma', 10.0, 'dssm cosine放大系数')

# deepx models' options
DEFINE_bool(parser, 'use_seperated_logits', False, '')
DEFINE_bool(parser, 'use_weighted_logits', False, '')

# fm
DEFINE_bool(parser, 'fm_use_shared_embedding', True, '')
DEFINE_bool(parser, 'fm_use_project', False, '')
DEFINE_integer(parser, 'fm_project_size', 32, '')

# fwfm
DEFINE_bool(parser, 'fwfm_use_shared_embedding', True, '')
DEFINE_bool(parser, 'fwfm_use_project', False, '')
DEFINE_integer(parser, 'fwfm_project_size', 32, '')

# afm
DEFINE_bool(parser, 'afm_use_shared_embedding', True, '')
DEFINE_bool(parser, 'afm_use_project', False, '')
DEFINE_integer(parser, 'afm_project_size', 32, '')
DEFINE_integer(parser, 'afm_hidden_unit', 32, '')

# iafm
DEFINE_bool(parser, 'iafm_use_shared_embedding', True, '')
DEFINE_bool(parser, 'iafm_use_project', False, '')
DEFINE_integer(parser, 'iafm_project_size', 32, '')
DEFINE_integer(parser, 'iafm_hidden_unit', 32, '')
DEFINE_integer(parser, 'iafm_field_dim', 8, '')

# ifm
DEFINE_bool(parser, 'ifm_use_shared_embedding', True, '')
DEFINE_bool(parser, 'ifm_use_project', False, '')
DEFINE_integer(parser, 'ifm_project_size', 32, '')
DEFINE_integer(parser, 'ifm_hidden_unit', 32, '')
DEFINE_integer(parser, 'ifm_field_dim', 8, '')

# kfm
DEFINE_bool(parser, 'kfm_use_shared_embedding', True, '')
DEFINE_bool(parser, 'kfm_use_project', False, '')
DEFINE_integer(parser, 'kfm_project_size', 32, '')

# wkfm
DEFINE_bool(parser, 'wkfm_use_shared_embedding', True, '')
DEFINE_bool(parser, 'wkfm_use_project', False, '')
DEFINE_integer(parser, 'wkfm_project_size', 32, '')
DEFINE_bool(parser, 'wkfm_use_selector', False, '')

# nifm
DEFINE_bool(parser, 'nifm_use_shared_embedding', True, '')
DEFINE_bool(parser, 'nifm_use_project', False, '')
DEFINE_integer(parser, 'nifm_project_size', 32, '')
DEFINE_list(parser, 'nifm_hidden_units', '32,16', '')
DEFINE_string(parser, 'nifm_activation_fn', 'relu', '')
DEFINE_float(parser, 'nifm_dropout', 0.0, '')
DEFINE_bool(parser, 'nifm_batch_norm', False, '')
DEFINE_bool(parser, 'nifm_layer_norm', False, '')
DEFINE_bool(parser, 'nifm_use_resnet', False, '')
DEFINE_bool(parser, 'nifm_use_densenet', False, '')

# cin
DEFINE_bool(parser, 'cin_use_shared_embedding', True, '')
DEFINE_bool(parser, 'cin_use_project', False, '')
DEFINE_integer(parser, 'cin_project_size', 32, '')
DEFINE_list(parser, 'cin_hidden_feature_maps', '128,128', '')
DEFINE_bool(parser, 'cin_split_half', True, '')

# cross
DEFINE_bool(parser, 'cross_use_shared_embedding', True, '')
DEFINE_bool(parser, 'cross_use_project', False, '')
DEFINE_integer(parser, 'cross_project_size', 32, '')
DEFINE_integer(parser, 'cross_num_layers', 4, '')

# autoint
DEFINE_bool(parser, 'autoint_use_shared_embedding', True, '')
DEFINE_bool(parser, 'autoint_use_project', False, '')
DEFINE_integer(parser, 'autoint_project_size', 32, '')
DEFINE_integer(parser, 'autoint_size_per_head', 16, '')
DEFINE_integer(parser, 'autoint_num_heads', 6, '')
DEFINE_integer(parser, 'autoint_num_blocks', 2, '')
DEFINE_float(parser, 'autoint_dropout', 0.0, '')
DEFINE_bool(parser, 'autoint_has_residual', True, '')

# dnn
DEFINE_bool(parser, 'dnn_use_shared_embedding', True, '')
DEFINE_bool(parser, 'dnn_use_project', False, '')
DEFINE_integer(parser, 'dnn_project_size', 32, '')
DEFINE_list(parser, 'dnn_hidden_units', '512,256,128', '')
DEFINE_string(parser, 'dnn_activation_fn', 'relu', '')
DEFINE_float(parser, 'dnn_dropout', 0.0, '')
DEFINE_bool(parser, 'dnn_batch_norm', False, '')
DEFINE_bool(parser, 'dnn_layer_norm', False, '')
DEFINE_bool(parser, 'dnn_use_resnet', False, '')
DEFINE_bool(parser, 'dnn_use_densenet', False, '')
DEFINE_float(parser, 'dnn_l2_regularizer', 0.0, '')

# multi_dnn
DEFINE_bool(parser, 'multi_dnn_use_shared_embedding', True, '')
DEFINE_bool(parser, 'multi_dnn_use_project', False, '')
DEFINE_integer(parser, 'multi_dnn_project_size', 32, '')
DEFINE_list(parser, 'multi_dnn_shared_hidden_units', '512,256,128', '')
DEFINE_list(parser, 'multi_dnn_tower_hidden_units', '512,256,128', '')
DEFINE_bool(parser, 'multi_dnn_tower_use_shared_embedding', True, '')
DEFINE_string(parser, 'multi_dnn_activation_fn', 'relu', '')
DEFINE_float(parser, 'multi_dnn_dropout', 0.0, '')
DEFINE_bool(parser, 'multi_dnn_batch_norm', False, '')
DEFINE_bool(parser, 'multi_dnn_layer_norm', False, '')
DEFINE_bool(parser, 'multi_dnn_use_resnet', False, '')
DEFINE_bool(parser, 'multi_dnn_use_densenet', False, '')
DEFINE_float(parser, 'multi_dnn_l2_regularizer', 0.0, '')

# nfm
DEFINE_bool(parser, 'nfm_use_shared_embedding', True, '')
DEFINE_bool(parser, 'nfm_use_project', False, '')
DEFINE_integer(parser, 'nfm_project_size', 32, '')
DEFINE_list(parser, 'nfm_hidden_units', '512,256,128', '')
DEFINE_string(parser, 'nfm_activation_fn', 'relu', '')
DEFINE_float(parser, 'nfm_dropout', 0.0, '')
DEFINE_bool(parser, 'nfm_batch_norm', False, '')
DEFINE_bool(parser, 'nfm_layer_norm', False, '')
DEFINE_bool(parser, 'nfm_use_resnet', False, '')
DEFINE_bool(parser, 'nfm_use_densenet', False, '')

# nkfm
DEFINE_bool(parser, 'nkfm_use_shared_embedding', True, '')
DEFINE_bool(parser, 'nkfm_use_project', False, '')
DEFINE_integer(parser, 'nkfm_project_size', 32, '')
DEFINE_list(parser, 'nkfm_hidden_units', '512,256,128', '')
DEFINE_string(parser, 'nkfm_activation_fn', 'relu', '')
DEFINE_float(parser, 'nkfm_dropout', 0.0, '')
DEFINE_bool(parser, 'nkfm_batch_norm', False, '')
DEFINE_bool(parser, 'nkfm_layer_norm', False, '')
DEFINE_bool(parser, 'nkfm_use_resnet', False, '')
DEFINE_bool(parser, 'nkfm_use_densenet', False, '')

# ccpm
DEFINE_bool(parser, 'ccpm_use_shared_embedding', True, '')
DEFINE_bool(parser, 'ccpm_use_project', False, '')
DEFINE_integer(parser, 'ccpm_project_size', 32, '')
DEFINE_list(parser, 'ccpm_hidden_units', '512,256,128', '')
DEFINE_string(parser, 'ccpm_activation_fn', 'relu', '')
DEFINE_float(parser, 'ccpm_dropout', 0.0, '')
DEFINE_bool(parser, 'ccpm_batch_norm', False, '')
DEFINE_bool(parser, 'ccpm_layer_norm', False, '')
DEFINE_bool(parser, 'ccpm_use_resnet', False, '')
DEFINE_bool(parser, 'ccpm_use_densenet', False, '')
DEFINE_list(parser, 'ccpm_kernel_sizes', '3,3,3', '')
DEFINE_list(parser, 'ccpm_filter_nums', '4,3,2', '')

# ipnn
DEFINE_bool(parser, 'ipnn_use_shared_embedding', True, '')
DEFINE_bool(parser, 'ipnn_use_project', False, '')
DEFINE_integer(parser, 'ipnn_project_size', 32, '')
DEFINE_list(parser, 'ipnn_hidden_units', '512,256,128', '')
DEFINE_string(parser, 'ipnn_activation_fn', 'relu', '')
DEFINE_float(parser, 'ipnn_dropout', 0.0, '')
DEFINE_bool(parser, 'ipnn_batch_norm', False, '')
DEFINE_bool(parser, 'ipnn_layer_norm', False, '')
DEFINE_bool(parser, 'ipnn_use_resnet', False, '')
DEFINE_bool(parser, 'ipnn_use_densenet', False, '')
DEFINE_bool(parser, 'ipnn_unordered_inner_product', False, '')
DEFINE_bool(parser, 'ipnn_concat_project', False, '')

# kpnn
DEFINE_bool(parser, 'kpnn_use_shared_embedding', True, '')
DEFINE_bool(parser, 'kpnn_use_project', False, '')
DEFINE_integer(parser, 'kpnn_project_size', 32, '')
DEFINE_list(parser, 'kpnn_hidden_units', '512,256,128', '')
DEFINE_string(parser, 'kpnn_activation_fn', 'relu', '')
DEFINE_float(parser, 'kpnn_dropout', 0.0, '')
DEFINE_bool(parser, 'kpnn_batch_norm', False, '')
DEFINE_bool(parser, 'kpnn_layer_norm', False, '')
DEFINE_bool(parser, 'kpnn_use_resnet', False, '')
DEFINE_bool(parser, 'kpnn_use_densenet', False, '')
DEFINE_bool(parser, 'kpnn_concat_project', False, '')

# pin
DEFINE_bool(parser, 'pin_use_shared_embedding', True, '')
DEFINE_bool(parser, 'pin_use_project', False, '')
DEFINE_integer(parser, 'pin_project_size', 32, '')
DEFINE_list(parser, 'pin_hidden_units', '512,256,128', '')
DEFINE_string(parser, 'pin_activation_fn', 'relu', '')
DEFINE_float(parser, 'pin_dropout', 0.0, '')
DEFINE_bool(parser, 'pin_batch_norm', False, '')
DEFINE_bool(parser, 'pin_layer_norm', False, '')
DEFINE_bool(parser, 'pin_use_resnet', False, '')
DEFINE_bool(parser, 'pin_use_densenet', False, '')
DEFINE_bool(parser, 'pin_use_concat', False, '')
DEFINE_bool(parser, 'pin_concat_project', False, '')
DEFINE_list(parser, 'pin_subnet_hidden_units', '64,32', '')

# fibinet
DEFINE_bool(parser, 'fibinet_use_shared_embedding', True, '')
DEFINE_bool(parser, 'fibinet_use_project', False, '')
DEFINE_integer(parser, 'fibinet_project_size', 32, '')
DEFINE_list(parser, 'fibinet_hidden_units', '512,256,128', '')
DEFINE_string(parser, 'fibinet_activation_fn', 'relu', '')
DEFINE_float(parser, 'fibinet_dropout', 0.0, '')
DEFINE_bool(parser, 'fibinet_batch_norm', False, '')
DEFINE_bool(parser, 'fibinet_layer_norm', False, '')
DEFINE_bool(parser, 'fibinet_use_resnet', False, '')
DEFINE_bool(parser, 'fibinet_use_densenet', False, '')
DEFINE_bool(parser, 'fibinet_use_se', True, '')
DEFINE_bool(parser, 'fibinet_use_deep', True, '')
DEFINE_string(
    parser,
    'fibinet_interaction_type', 'bilinear',
    '"inner", "hadamard", "bilinear"')
DEFINE_string(
    parser,
    'fibinet_se_interaction_type', 'bilinear',
    '"inner", "hadamard", "bilinear"')
DEFINE_bool(parser, 'fibinet_se_use_shared_embedding', False, '')

# fgcnn
DEFINE_bool(parser, 'fgcnn_use_shared_embedding', False, '')
DEFINE_bool(parser, 'fgcnn_use_project', False, '')
DEFINE_integer(parser, 'fgcnn_project_dim', 32, '')
DEFINE_list(parser, 'fgcnn_filter_nums', '6,8', '')
DEFINE_list(parser, 'fgcnn_kernel_sizes', '7,7', '')
DEFINE_list(parser, 'fgcnn_pooling_sizes', '2,2', '')
DEFINE_list(parser, 'fgcnn_new_map_sizes', '3,3', '')

# activation functions
DEFINE_float(parser, 'leaky_relu_alpha', 0.2, '')
DEFINE_float(parser, 'swish_beta', 1.0, '')

# train flags
DEFINE_string(parser, 'loss_reduction', 'sum', '"mean", "sum"')
DEFINE_string(parser, 'loss_fn', 'default',
              '"default", "bi_tempered", "focal", "mape", "huber"')
DEFINE_float(parser, 'bi_tempered_loss_t1', 1.0, '')
DEFINE_float(parser, 'bi_tempered_loss_t2', 1.0, '')
DEFINE_float(parser, 'bi_tempered_loss_label_smoothing', 0.0, '')
DEFINE_integer(parser, 'bi_tempered_loss_num_iters', 5, '')
DEFINE_float(parser, 'focal_loss_gamma', 2.0, '')
DEFINE_float(parser, 'mape_loss_delta', 0.0, '')
DEFINE_float(parser, 'huber_loss_delta', 1.0, '')
DEFINE_string(parser, 'label_fn', 'default', '"default", "log"')
DEFINE_string(parser, 'linear_sparse_combiner', 'sum', 'sum or mean')
DEFINE_integer(parser, 'batch_size', 512, 'batch size')
DEFINE_integer(parser, 'eval_batch_size', 1024, 'eval batch size')
DEFINE_integer(parser, 'max_train_steps', -1, '')
DEFINE_integer(parser, 'max_eval_steps', -1, '')
DEFINE_integer(parser, 'epoch', 1, '')
DEFINE_integer(
    parser,
    'total_steps', 1,
    'total train steps inferenced from train data')
DEFINE_integer(parser, 'num_gpus', 0, 'num gpus')

# dataset flags
DEFINE_integer(parser, 'prefetch_size', 4096, '')
DEFINE_integer(parser, 'shuffle_size', 2000, '')
DEFINE_bool(parser, 'shuffle_batch', False, '')
DEFINE_integer(parser, 'map_num_parallel_calls', 10, '')
DEFINE_integer(
    parser,
    'num_parallel_reads', 1,
    ' the number of files to read in parallel.')
DEFINE_integer(parser, 'read_buffer_size_mb', 1024, '')

# log flags
DEFINE_integer(parser, 'save_summary_steps', 1000, '')
DEFINE_integer(parser, 'save_checkpoints_steps', 100000, '')
DEFINE_integer(parser, 'keep_checkpoint_max', 3, '')
DEFINE_integer(parser, 'log_step_count_steps', 1000, '')
DEFINE_string(parser, 'serving_warmup_file', '', '')
DEFINE_string(parser, 'predictor_warmup_file', '', '')

# optimizer
DEFINE_string(
    parser,
    'deep_optimizer', 'adagrad',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam", "lazy_adam", '
    '"ftrl", "momentum", "sgd", "rmsprop", "proximal_adagrad", '
    '"yellowfin", "adamw".')
DEFINE_float(parser, 'deep_learning_rate', 0.1, 'learning rate')

DEFINE_float(
    parser,
    'deep_adadelta_rho', 0.95,
    'The decay rate for adadelta.')

DEFINE_float(
    parser,
    'deep_adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

DEFINE_float(
    parser,
    'deep_adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

DEFINE_float(
    parser,
    'deep_adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

DEFINE_float(
    parser,
    'deep_lazy_adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

DEFINE_float(
    parser,
    'deep_lazy_adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

DEFINE_float(
    parser,
    'deep_opt_epsilon', 1e-6,
    'Epsilon term for the optimizer.')

DEFINE_float(
    parser,
    'deep_ftrl_learning_rate_power', -0.5,
    'deep_The learning rate power.')

DEFINE_float(
    parser,
    'deep_ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

DEFINE_float(
    parser,
    'deep_ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

DEFINE_float(
    parser,
    'deep_ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

DEFINE_float(
    parser,
    'deep_momentum', 0.9,
    'The momentum for the MomentumOptimizer, RMSPropOptimizer and YFOptimizer.')

DEFINE_float(parser, 'deep_rmsprop_momentum', 0.9, 'Momentum.')
DEFINE_float(parser, 'deep_rmsprop_decay', 0.9, 'Decay term for RMSProp.')
DEFINE_float(
    parser,
    'deep_proximal_adagrad_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

DEFINE_float(
    parser,
    'deep_proximal_adagrad_l1', 0.0,
    'The ProximalAdagrad l1 regularization strength.')

DEFINE_float(
    parser,
    'deep_proximal_adagrad_l2', 0.0,
    'The ProximalAdagrad l2 regularization strength.')

DEFINE_float(parser, 'deep_adamw_weight_decay_rate', 0.01, '')
DEFINE_float(parser, 'deep_adamw_beta1', 0.9, '')
DEFINE_float(parser, 'deep_adamw_beta2', 0.999, '')

DEFINE_string(
    parser,
    'deep_learning_rate_decay_type',
    'fixed',
    'Specifies how the learning rate is decayed. One of "fixed", '
    '"exponential", "polynomial", or "warmup"')

DEFINE_float(
    parser,
    'deep_end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

DEFINE_integer(
    parser,
    'deep_decay_steps', 1000,
    'The decay steps used by a polynomial decay learning rate.')

DEFINE_float(
    parser,
    'deep_learning_rate_decay_factor', 0.99,
    'The decay rate used by a exponential decay learning rate.')
DEFINE_float(parser, 'deep_cosine_decay_alpha', 0.0, '')
DEFINE_float(parser, 'deep_polynomial_decay_power', 1.0, '')
DEFINE_float(parser, 'deep_warmup_rate', 0.3, '')

DEFINE_string(
    parser,
    'wide_optimizer', 'ftrl',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam", "lazy_adam", '
    '"ftrl", "momentum", "sgd", "rmsprop", "proximal_adagrad", "yellowfin".')
DEFINE_float(parser, 'wide_learning_rate', 0.005, 'learning rate')

DEFINE_float(parser, 'wide_adadelta_rho', 0.95, 'The decay rate for adadelta.')

DEFINE_float(
    parser,
    'wide_adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

DEFINE_float(
    parser,
    'wide_adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

DEFINE_float(
    parser,
    'wide_adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

DEFINE_float(
    parser,
    'wide_lazy_adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

DEFINE_float(
    parser,
    'wide_lazy_adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

DEFINE_float(
    parser,
    'wide_opt_epsilon', 1e-6,
    'Epsilon term for the optimizer.')

DEFINE_float(
    parser,
    'wide_ftrl_learning_rate_power', -0.5,
    'The learning rate power.')

DEFINE_float(
    parser,
    'wide_ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

DEFINE_float(parser, 'wide_ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
DEFINE_float(parser, 'wide_ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
DEFINE_float(
    parser,
    'wide_momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

DEFINE_float(parser, 'wide_rmsprop_momentum', 0.9, 'Momentum.')
DEFINE_float(parser, 'wide_rmsprop_decay', 0.9, 'Decay term for RMSProp.')

DEFINE_float(
    parser,
    'wide_proximal_adagrad_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

DEFINE_float(
    parser,
    'wide_proximal_adagrad_l1', 0.0,
    'The ProximalAdagrad l1 regularization strength.')

DEFINE_float(
    parser,
    'wide_proximal_adagrad_l2', 0.0,
    'The ProximalAdagrad l2 regularization strength.')

DEFINE_string(
    parser,
    'wide_learning_rate_decay_type', 'fixed',
    'Specifies how the learning rate is decayed. One of "fixed", '
    '"exponential", or "polynomial"')

DEFINE_float(
    parser,
    'wide_end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

DEFINE_integer(
    parser,
    'wide_decay_steps', 1000,
    'The decay steps used by a polynomial decay learning rate.')

DEFINE_float(
    parser,
    'wide_learning_rate_decay_factor', 0.99,
    'The decay rate used by a exponential decay learning rate.')
DEFINE_float(parser, 'wide_polynomial_decay_power', 1.0, '')

DEFINE_string(parser, 'compression_type', '', '"", "GZIP"')

DEFINE_string(
    parser, 'result_output_file', 'result.log',
    'output evaluate result to a file')

DEFINE_float(parser, 'auc_thr', 0.6, 'auc threshold.')

DEFINE_bool(parser, 'use_spark_fuel', False, '')
DEFINE_integer(parser, 'num_ps', 1, 'Number of ps nodes.')
DEFINE_integer(parser, 'start_delay_secs', 120, '')
DEFINE_integer(parser, 'throttle_secs', 600, '')
DEFINE_integer(parser, 'eval_steps_every_checkpoint', 100, '')

# TODO(zhezhaoxu) Local mode not support it, now only used in spark fuel mode
DEFINE_bool(parser, 'remove_model_dir', True, '')

# nce flags
DEFINE_bool(parser, 'use_negative_sampling', False, '')
DEFINE_integer(parser, 'nce_count', 5, '')
DEFINE_integer(parser, 'nce_items_min_count', 10, '')
DEFINE_integer(parser, 'nce_items_top_k', -1, '')
DEFINE_string(parser, 'nce_items_path', '', '')

# lua sampler flags
DEFINE_bool(parser, 'use_lua_sampler', False, '')
DEFINE_string(parser, 'lua_sampler_script', '', '')
