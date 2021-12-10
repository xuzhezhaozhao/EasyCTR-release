#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../../tools/conf_generator")

import os
from conf_generator import Assembler, Transform, InputData, Configuration

base_dir = './data/dict'
meta_file = './data/data.meta'

wide = ['wide']
deep = ['deep']
wide_deep = ['wide', 'deep']
dssm1 = ['dssm1', 'deep', 'essm_bottom_1', 'dtower_bottom_1']
dssm2 = ['dssm2', 'deep', 'essm_bottom_2', 'dtower_bottom_2']

# parse dtypes from meta
dtypes = {'label': 'float', 'weight': 'float'}
for lineindex, line in enumerate(open('./data/data.meta')):
    if lineindex == 0:
        continue   # skip header
    line = line.strip()
    if line == "" or line[0] == '#':
        continue
    tokens = line.split()
    name = tokens[1]
    dtype = tokens[2]
    dtypes[name] = dtype

columns = [
    {'in_use': 1, 'name': 't_click_label', 'default': 0},

    {'in_use': 1, 'name': 'u_session_cpc_click_ads', 'min_count': 10, 'min_weight': 0.0, 'width':30, 'oov_buckets': 50 , 'top_k': None},
    {'in_use': 1, 'name': 'u_session_cpc_exp_ads', 'min_count': 10, 'min_weight': 0.0, 'width':30, 'oov_buckets': 50, 'top_k': None},
    {'in_use': 1, 'name': 'u_phone_model', 'min_count': 10, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'u_age', 'min_count': 10, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'u_gender', 'min_count': 10, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'u_region', 'min_count': 10, 'oov_buckets': 20, 'top_k': None},
    {'in_use': 1, 'name': 'u_com_province', 'min_count': 50, 'oov_buckets': 50, 'top_k': None},
    {'in_use': 1, 'name': 'u_com_city', 'min_count': 20, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'u_phone_series', 'min_count': 10, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'u_phone_price_zone', 'min_count': 10, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'u_phone_prize_zone_type', 'min_count': 10, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'u_current_province', 'min_count': 50, 'oov_buckets': 50, 'top_k': None},
    {'in_use': 1, 'name': 'u_current_city', 'min_count': 20, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'u_city_grade', 'min_count': 20, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'i_ad_id', 'min_count': 10, 'oov_buckets': 50, 'top_k': None},
    {'in_use': 1, 'name': 'i_app_type_id', 'min_count': 5, 'oov_buckets': 6, 'top_k': None, 'default_key': '125'},
    {'in_use': 1, 'name': 'i_industry_id', 'min_count': 5, 'oov_buckets': 64, 'top_k': None},
    {'in_use': 1, 'name': 'i_sec_industry_id', 'min_count': 5, 'oov_buckets': 35, 'top_k': None},
    {'in_use': 1, 'name': 'i_max_spend', 'min_count': 5, 'width': 20, 'oov_buckets': 2200, 'top_k': None},

    #{'in_use': 1, 'name': 'i_flow_type', 'min_count': 1, 'oov_buckets': 6, 'top_k': None},
    {'in_use': 1, 'name': 'i_trade_category', 'min_count': 5, 'oov_buckets': 70, 'top_k': None},
    {'in_use': 1, 'name': 'i_pay_type_id', 'min_count': 5, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'i_install_type_id', 'min_count': 5, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'i_network_type_id', 'min_count': 5, 'oov_buckets': 6, 'top_k': None},
    #{'in_use': 1, 'name': 'i_advertiser_type_id', 'min_count': 1, 'oov_buckets': 6, 'top_k': None},
    #{'in_use': 1, 'name': 'i_advertiser_id', 'min_count': 1, 'oov_buckets': 15000, 'top_k': None},
    {'in_use': 1, 'name': 'i_is_h5', 'min_count': 5, 'oov_buckets': 6, 'top_k': None},
    {'in_use': 1, 'name': 'i_is_deeplink', 'min_count': 5, 'oov_buckets': 6, 'top_k': None},
    #{'in_use': 1, 'name': 'i_display_mode_id', 'min_count': 1, 'oov_buckets': 20, 'top_k': None},
    {'in_use': 1, 'name': 'i_is_direct', 'min_count': 5, 'oov_buckets': 6, 'top_k': None},
]


def get_dict_file(column):
    dict_file = column.get('dict_file', None)
    if dict_file is None:
        dict_file = os.path.join(base_dir, name + '.dict')
    return ''
    #return dict_file


# add features to assembler
assembler = Assembler()
for column in columns:
    try:
        if column['in_use'] != 1:
            continue
        name = column['name']
        dtype = dtypes[name]
        if dtype == 'int':
            assembler.add_int(name, column['default'], alias=column.get('alias', None))
        elif dtype == 'float':
            assembler.add_float(name, column['default'], alias=column.get('alias', None))
        elif dtype == 'string':
            assembler.add_string(
                iname=name,
                alias=column.get('alias', None),
                dict_file=get_dict_file(column),
                min_count=column['min_count'],
                oov_buckets=column.get('oov_buckets', None),
                top_k=column.get('top_k', None),
                default_key=column.get('default_key', None))
        elif dtype == 'weighted_string_list':
            assembler.add_weighted_string_list(
                iname=name,
                alias=column.get('alias', None),
                dict_file=get_dict_file(column),
                min_count=column['min_count'],
                width=column['width'],
                top_k=column.get('top_k', None),
                oov_buckets=column.get('oov_buckets', None),
                min_weight=column['min_weight'],
                default_key=column.get('default_key', None))
        elif dtype == 'string_list':
            assembler.add_string_list(
                iname=name,
                alias=column.get('alias', None),
                dict_file=get_dict_file(column),
                min_count=column['min_count'],
                width=column['width'],
                oov_buckets=column.get('oov_buckets', None),
                top_k=column.get('top_k', None),
                default_key=column.get('default_key', None))
        else:
            raise ValueError("Unknow dtype '{}'".format(dtype))
    except Exception as e:
        print(column)
        raise e


# {{{ Transform =========================================================
transform = Transform()

#transform.add_numeric('t_click_label', ['essm_second_target'])

# user
transform.add_categorical_identity('u_session_cpc_click_ads', [])
transform.add_embedding('u_session_cpc_click_ads.identity', dssm1, 30)

transform.add_categorical_identity('u_session_cpc_exp_ads', [])
transform.add_embedding('u_session_cpc_exp_ads.identity', dssm1, 30)

# add new user feature
transform.add_categorical_identity('u_phone_model', [])
transform.add_indicator('u_phone_model.identity', dssm1)

transform.add_categorical_identity('u_age', [])
transform.add_indicator('u_age.identity', dssm1)

transform.add_categorical_identity('u_gender', [])
transform.add_indicator('u_gender.identity', dssm1)

transform.add_categorical_identity('u_region', [])
transform.add_indicator('u_region.identity', dssm1)

transform.add_categorical_identity('u_com_province', [])
transform.add_indicator('u_com_province.identity', dssm1)

transform.add_categorical_identity('u_com_city', [])
transform.add_indicator('u_com_city.identity', dssm1)

transform.add_categorical_identity('u_phone_series',[])
transform.add_embedding('u_phone_series.identity', dssm1, 10)   # embedding

transform.add_categorical_identity('u_phone_price_zone', [])
transform.add_indicator('u_phone_price_zone.identity', dssm1)

transform.add_categorical_identity('u_phone_prize_zone_type', [])
transform.add_indicator('u_phone_prize_zone_type.identity', dssm1)

transform.add_categorical_identity('u_current_province', [])
transform.add_indicator('u_current_province.identity', dssm1)

transform.add_categorical_identity('u_current_city', [])
transform.add_indicator('u_current_city.identity', dssm1)

transform.add_categorical_identity('u_city_grade', [])
transform.add_indicator('u_city_grade.identity', dssm1)

# item
transform.add_categorical_identity('i_ad_id',[])
transform.add_embedding('i_ad_id.identity', dssm2, 50)   # embedding

transform.add_categorical_identity('i_app_type_id', [])
transform.add_indicator('i_app_type_id.identity', dssm2)  # one hot

transform.add_categorical_identity('i_industry_id', [])
transform.add_indicator('i_industry_id.identity', dssm2)

transform.add_categorical_identity('i_sec_industry_id', [])
transform.add_indicator('i_sec_industry_id.identity', dssm2)

transform.add_categorical_identity('i_max_spend', [])
transform.add_indicator('i_max_spend.identity', dssm2)

#transform.add_categorical_identity('i_flow_type', [])
#transform.add_indicator('i_flow_type.identity', dssm1)

transform.add_categorical_identity('i_trade_category', [])
transform.add_indicator('i_trade_category.identity', dssm2)

transform.add_categorical_identity('i_pay_type_id', [])
transform.add_indicator('i_pay_type_id.identity', dssm2)

transform.add_categorical_identity('i_install_type_id', [])
transform.add_indicator('i_install_type_id.identity', dssm2)

transform.add_categorical_identity('i_network_type_id', [])
transform.add_indicator('i_network_type_id.identity', dssm2)

#transform.add_categorical_identity('i_advertiser_type_id', [])
#transform.add_indicator('i_advertiser_type_id.identity', dssm1)

#transform.add_categorical_identity('i_advertiser_id', [])
#transform.add_embedding('i_advertiser_id.identity', dssm1, 20)

transform.add_categorical_identity('i_is_h5', [])
transform.add_indicator('i_is_h5.identity', dssm2)

transform.add_categorical_identity('i_is_deeplink', [])
transform.add_indicator('i_is_deeplink.identity', dssm2)

#transform.add_categorical_identity('i_display_mode_id', [])
#transform.add_indicator('i_display_mode_id.identity', dssm1)

transform.add_categorical_identity('i_is_direct', [])
transform.add_indicator('i_is_direct.identity', dssm2)


model_name = "vivo_deepRecallV1"
input_data = InputData()
InputData.meta_file = meta_file
cf = Configuration(assembler, transform, input_data, model_name)
print(cf.to_json())

# vim:foldmethod=marker
