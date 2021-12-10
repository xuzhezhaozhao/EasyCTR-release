#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_extra_options(opts):

    extra_options = {}
    extra_options['leaky_relu_alpha'] = opts.leaky_relu_alpha
    extra_options['swish_beta'] = opts.swish_beta

    extra_options['dssm_mode'] = opts.dssm_mode
    extra_options['dssm1_hidden_units'] = [int(x) for x in opts.dssm1_hidden_units]
    extra_options['dssm1_activation_fn'] = opts.dssm1_activation_fn
    extra_options['dssm1_dropout'] = opts.dssm1_dropout
    extra_options['dssm1_gaussian_dropout'] = opts.dssm1_gaussian_dropout
    extra_options['dssm1_batch_norm'] = opts.dssm1_batch_norm
    extra_options['dssm1_layer_norm'] = opts.dssm1_layer_norm
    extra_options['dssm1_use_resnet'] = opts.dssm1_use_resnet
    extra_options['dssm1_use_densenet'] = opts.dssm1_use_densenet
    extra_options['dssm2_hidden_units'] = [int(x) for x in opts.dssm2_hidden_units]
    extra_options['dssm2_activation_fn'] = opts.dssm2_activation_fn
    extra_options['dssm2_dropout'] = opts.dssm2_dropout
    extra_options['dssm2_gaussian_dropout'] = opts.dssm2_gaussian_dropout
    extra_options['dssm2_batch_norm'] = opts.dssm2_batch_norm
    extra_options['dssm2_layer_norm'] = opts.dssm2_layer_norm
    extra_options['dssm2_use_resnet'] = opts.dssm2_use_resnet
    extra_options['dssm2_use_densenet'] = opts.dssm2_use_densenet
    extra_options['dssm_cosine_gamma'] = opts.dssm_cosine_gamma

    extra_options['fm_use_shared_embedding'] = opts.fm_use_shared_embedding
    extra_options['fm_use_project'] = opts.fm_use_project
    extra_options['fm_project_size'] = opts.fm_project_size

    extra_options['fwfm_use_shared_embedding'] = opts.fwfm_use_shared_embedding
    extra_options['fwfm_use_project'] = opts.fwfm_use_project
    extra_options['fwfm_project_size'] = opts.fwfm_project_size

    extra_options['afm_use_shared_embedding'] = opts.afm_use_shared_embedding
    extra_options['afm_use_project'] = opts.afm_use_project
    extra_options['afm_project_size'] = opts.afm_project_size
    extra_options['afm_hidden_unit'] = opts.afm_hidden_unit

    extra_options['iafm_use_shared_embedding'] = opts.iafm_use_shared_embedding
    extra_options['iafm_use_project'] = opts.iafm_use_project
    extra_options['iafm_project_size'] = opts.iafm_project_size
    extra_options['iafm_hidden_unit'] = opts.iafm_hidden_unit
    extra_options['iafm_field_dim'] = opts.iafm_field_dim

    extra_options['ifm_use_shared_embedding'] = opts.ifm_use_shared_embedding
    extra_options['ifm_use_project'] = opts.ifm_use_project
    extra_options['ifm_project_size'] = opts.ifm_project_size
    extra_options['ifm_hidden_unit'] = opts.ifm_hidden_unit
    extra_options['ifm_field_dim'] = opts.ifm_field_dim

    extra_options['kfm_use_shared_embedding'] = opts.kfm_use_shared_embedding
    extra_options['kfm_use_project'] = opts.kfm_use_project
    extra_options['kfm_project_size'] = opts.kfm_project_size

    extra_options['wkfm_use_shared_embedding'] = opts.wkfm_use_shared_embedding
    extra_options['wkfm_use_project'] = opts.wkfm_use_project
    extra_options['wkfm_project_size'] = opts.wkfm_project_size
    extra_options['wkfm_use_selector'] = opts.wkfm_use_selector

    extra_options['nifm_use_shared_embedding'] = opts.nifm_use_shared_embedding
    extra_options['nifm_use_project'] = opts.nifm_use_project
    extra_options['nifm_project_size'] = opts.nifm_project_size
    extra_options['nifm_hidden_units'] = [int(x) for x in opts.nifm_hidden_units]
    extra_options['nifm_activation_fn'] = opts.nifm_activation_fn
    extra_options['nifm_dropout'] = opts.nifm_dropout
    extra_options['nifm_batch_norm'] = opts.nifm_batch_norm
    extra_options['nifm_layer_norm'] = opts.nifm_layer_norm
    extra_options['nifm_use_resnet'] = opts.nifm_use_resnet
    extra_options['nifm_use_densenet'] = opts.nifm_use_densenet

    extra_options['cin_use_shared_embedding'] = opts.cin_use_shared_embedding
    extra_options['cin_use_project'] = opts.cin_use_project
    extra_options['cin_project_size'] = opts.cin_project_size
    extra_options['cin_hidden_feature_maps'] = [int(x) for x in opts.cin_hidden_feature_maps]
    extra_options['cin_split_half'] = opts.cin_split_half

    extra_options['cross_use_shared_embedding'] = opts.cross_use_shared_embedding
    extra_options['cross_use_project'] = opts.cross_use_project
    extra_options['cross_project_size'] = opts.cross_project_size
    extra_options['cross_num_layers'] = opts.cross_num_layers

    extra_options['autoint_use_shared_embedding'] = opts.autoint_use_shared_embedding
    extra_options['autoint_use_project'] = opts.autoint_use_project
    extra_options['autoint_project_size'] = opts.autoint_project_size
    extra_options['autoint_size_per_head'] = opts.autoint_size_per_head
    extra_options['autoint_num_heads'] = opts.autoint_num_heads
    extra_options['autoint_num_blocks'] = opts.autoint_num_blocks
    extra_options['autoint_dropout'] = opts.autoint_dropout
    extra_options['autoint_has_residual'] = opts.autoint_has_residual

    extra_options['dnn_use_shared_embedding'] = opts.dnn_use_shared_embedding
    extra_options['dnn_use_project'] = opts.dnn_use_project
    extra_options['dnn_project_size'] = opts.dnn_project_size
    extra_options['dnn_hidden_units'] = [int(x) for x in opts.dnn_hidden_units]
    extra_options['dnn_activation_fn'] = opts.dnn_activation_fn
    extra_options['dnn_dropout'] = opts.dnn_dropout
    extra_options['dnn_batch_norm'] = opts.dnn_batch_norm
    extra_options['dnn_layer_norm'] = opts.dnn_layer_norm
    extra_options['dnn_use_resnet'] = opts.dnn_use_resnet
    extra_options['dnn_use_densenet'] = opts.dnn_use_densenet
    extra_options['dnn_l2_regularizer'] = opts.dnn_l2_regularizer

    extra_options['multi_dnn_use_shared_embedding'] = opts.multi_dnn_use_shared_embedding
    extra_options['multi_dnn_use_project'] = opts.multi_dnn_use_project
    extra_options['multi_dnn_project_size'] = opts.multi_dnn_project_size
    extra_options['multi_dnn_shared_hidden_units'] = [int(x) for x in opts.multi_dnn_shared_hidden_units]
    extra_options['multi_dnn_tower_hidden_units'] = [int(x) for x in opts.multi_dnn_tower_hidden_units]
    extra_options['multi_dnn_tower_use_shared_embedding'] = opts.multi_dnn_tower_use_shared_embedding
    extra_options['multi_dnn_activation_fn'] = opts.multi_dnn_activation_fn
    extra_options['multi_dnn_dropout'] = opts.multi_dnn_dropout
    extra_options['multi_dnn_batch_norm'] = opts.multi_dnn_batch_norm
    extra_options['multi_dnn_layer_norm'] = opts.multi_dnn_layer_norm
    extra_options['multi_dnn_use_resnet'] = opts.multi_dnn_use_resnet
    extra_options['multi_dnn_use_densenet'] = opts.multi_dnn_use_densenet
    extra_options['multi_dnn_l2_regularizer'] = opts.multi_dnn_l2_regularizer

    extra_options['nfm_use_shared_embedding'] = opts.nfm_use_shared_embedding
    extra_options['nfm_use_project'] = opts.nfm_use_project
    extra_options['nfm_project_size'] = opts.nfm_project_size
    extra_options['nfm_hidden_units'] = [int(x) for x in opts.nfm_hidden_units]
    extra_options['nfm_activation_fn'] = opts.nfm_activation_fn
    extra_options['nfm_dropout'] = opts.nfm_dropout
    extra_options['nfm_batch_norm'] = opts.nfm_batch_norm
    extra_options['nfm_layer_norm'] = opts.nfm_layer_norm
    extra_options['nfm_use_resnet'] = opts.nfm_use_resnet
    extra_options['nfm_use_densenet'] = opts.nfm_use_densenet

    extra_options['nkfm_use_shared_embedding'] = opts.nkfm_use_shared_embedding
    extra_options['nkfm_use_project'] = opts.nkfm_use_project
    extra_options['nkfm_project_size'] = opts.nkfm_project_size
    extra_options['nkfm_hidden_units'] = [int(x) for x in opts.nkfm_hidden_units]
    extra_options['nkfm_activation_fn'] = opts.nkfm_activation_fn
    extra_options['nkfm_dropout'] = opts.nkfm_dropout
    extra_options['nkfm_batch_norm'] = opts.nkfm_batch_norm
    extra_options['nkfm_layer_norm'] = opts.nkfm_layer_norm
    extra_options['nkfm_use_resnet'] = opts.nkfm_use_resnet
    extra_options['nkfm_use_densenet'] = opts.nkfm_use_densenet

    extra_options['ccpm_use_shared_embedding'] = opts.ccpm_use_shared_embedding
    extra_options['ccpm_use_project'] = opts.ccpm_use_project
    extra_options['ccpm_project_size'] = opts.ccpm_project_size
    extra_options['ccpm_hidden_units'] = [int(x) for x in opts.ccpm_hidden_units]
    extra_options['ccpm_activation_fn'] = opts.ccpm_activation_fn
    extra_options['ccpm_dropout'] = opts.ccpm_dropout
    extra_options['ccpm_batch_norm'] = opts.ccpm_batch_norm
    extra_options['ccpm_layer_norm'] = opts.ccpm_layer_norm
    extra_options['ccpm_use_resnet'] = opts.ccpm_use_resnet
    extra_options['ccpm_use_densenet'] = opts.ccpm_use_densenet
    extra_options['ccpm_kernel_sizes'] = [int(x) for x in opts.ccpm_kernel_sizes]
    extra_options['ccpm_filter_nums'] = [int(x) for x in opts.ccpm_filter_nums]

    extra_options['ipnn_use_shared_embedding'] = opts.ipnn_use_shared_embedding
    extra_options['ipnn_use_project'] = opts.ipnn_use_project
    extra_options['ipnn_project_size'] = opts.ipnn_project_size
    extra_options['ipnn_hidden_units'] = [int(x) for x in opts.ipnn_hidden_units]
    extra_options['ipnn_activation_fn'] = opts.ipnn_activation_fn
    extra_options['ipnn_dropout'] = opts.ipnn_dropout
    extra_options['ipnn_batch_norm'] = opts.ipnn_batch_norm
    extra_options['ipnn_layer_norm'] = opts.ipnn_layer_norm
    extra_options['ipnn_use_resnet'] = opts.ipnn_use_resnet
    extra_options['ipnn_use_densenet'] = opts.ipnn_use_densenet
    extra_options['ipnn_unordered_inner_product'] = opts.ipnn_unordered_inner_product
    extra_options['ipnn_concat_project'] = opts.ipnn_concat_project

    extra_options['kpnn_use_shared_embedding'] = opts.kpnn_use_shared_embedding
    extra_options['kpnn_use_project'] = opts.kpnn_use_project
    extra_options['kpnn_project_size'] = opts.kpnn_project_size
    extra_options['kpnn_hidden_units'] = [int(x) for x in opts.kpnn_hidden_units]
    extra_options['kpnn_activation_fn'] = opts.kpnn_activation_fn
    extra_options['kpnn_dropout'] = opts.kpnn_dropout
    extra_options['kpnn_batch_norm'] = opts.kpnn_batch_norm
    extra_options['kpnn_layer_norm'] = opts.kpnn_layer_norm
    extra_options['kpnn_use_resnet'] = opts.kpnn_use_resnet
    extra_options['kpnn_use_densenet'] = opts.kpnn_use_densenet
    extra_options['kpnn_concat_project'] = opts.kpnn_concat_project

    extra_options['pin_use_shared_embedding'] = opts.pin_use_shared_embedding
    extra_options['pin_use_project'] = opts.pin_use_project
    extra_options['pin_project_size'] = opts.pin_project_size
    extra_options['pin_hidden_units'] = [int(x) for x in opts.pin_hidden_units]
    extra_options['pin_activation_fn'] = opts.pin_activation_fn
    extra_options['pin_dropout'] = opts.pin_dropout
    extra_options['pin_batch_norm'] = opts.pin_batch_norm
    extra_options['pin_layer_norm'] = opts.pin_layer_norm
    extra_options['pin_use_resnet'] = opts.pin_use_resnet
    extra_options['pin_use_densenet'] = opts.pin_use_densenet
    extra_options['pin_use_concat'] = opts.pin_use_concat
    extra_options['pin_concat_project'] = opts.pin_concat_project
    extra_options['pin_subnet_hidden_units'] = [int(x) for x in opts.pin_subnet_hidden_units]

    extra_options['fibinet_use_shared_embedding'] = opts.fibinet_use_shared_embedding
    extra_options['fibinet_use_project'] = opts.fibinet_use_project
    extra_options['fibinet_project_size'] = opts.fibinet_project_size
    extra_options['fibinet_hidden_units'] = [int(x) for x in opts.fibinet_hidden_units]
    extra_options['fibinet_activation_fn'] = opts.fibinet_activation_fn
    extra_options['fibinet_dropout'] = opts.fibinet_dropout
    extra_options['fibinet_batch_norm'] = opts.fibinet_batch_norm
    extra_options['fibinet_layer_norm'] = opts.fibinet_layer_norm
    extra_options['fibinet_use_resnet'] = opts.fibinet_use_resnet
    extra_options['fibinet_use_densenet'] = opts.fibinet_use_densenet
    extra_options['fibinet_use_se'] = opts.fibinet_use_se
    extra_options['fibinet_use_deep'] = opts.fibinet_use_deep
    extra_options['fibinet_interaction_type'] = opts.fibinet_interaction_type
    extra_options['fibinet_se_interaction_type'] = opts.fibinet_se_interaction_type
    extra_options['fibinet_se_use_shared_embedding'] = opts.fibinet_se_use_shared_embedding

    extra_options['fgcnn_use_shared_embedding'] = opts.fgcnn_use_shared_embedding
    extra_options['fgcnn_use_project'] = opts.fgcnn_use_project
    extra_options['fgcnn_project_dim'] = opts.fgcnn_project_dim
    extra_options['fgcnn_filter_nums'] = [int(x) for x in opts.fgcnn_filter_nums]
    extra_options['fgcnn_kernel_sizes'] = [int(x) for x in opts.fgcnn_kernel_sizes]
    extra_options['fgcnn_pooling_sizes'] = [int(x) for x in opts.fgcnn_pooling_sizes]
    extra_options['fgcnn_new_map_sizes'] = [int(x) for x in opts.fgcnn_new_map_sizes]

    return extra_options
