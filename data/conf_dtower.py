#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from conf_generator import Assembler, Transform, InputData, Configuration


base_dir = './data/dtower_dict'
meta_file = './data/data_dtower.meta'

wide = ['wide']
deep = ['deep']
wide_deep = ['wide', 'deep']
dssm1 = ['dssm1', 'deep', 'dtower_bottom_1']
dssm2 = ['dssm2', 'deep', 'dtower_bottom_2']

# parse dtypes from meta
dtypes = {'label': 'float', 'weight': 'float'}
for lineindex, line in enumerate(open('./data_dtower.meta')):
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
    {'in_use': 1, 'name': 't_convert_label', 'default': 1},
    {'in_use': 1, 'name': 'u_model', 'min_count': 20, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'u_age', 'min_count': 20, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'u_sex', 'min_count': 20, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'u_com_province', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'u_com_city', 'min_count': 20, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'u_city_grade','default':0, 'min_count': 20, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'u_price_zone_type','default':0, 'min_count': 20, 'oov_buckets': None},
    {'in_use': 1, 'name': 'u_ads_expo_his', 'min_count': 1, 'oov_buckets': None, 'top_k': None, 'width':100, 'min_weight':0.0},
    {'in_use': 1, 'name': 'u_ads_click_his', 'min_count': 1, 'oov_buckets': None, 'top_k': None, 'width':100, 'min_weight':0.0},
    {'in_use': 1, 'name': 'u_app_expo_browser_search_his', 'min_count': 1, 'oov_buckets': None, 'top_k': None, 'width':100, 'min_weight':0.0},
    {'in_use': 1, 'name': 'u_app_click_browser_search_his', 'min_count': 1, 'oov_buckets': None, 'top_k': None, 'width':100, 'min_weight':0.0},
    {'in_use': 1, 'name': 'u_installed_list_d', 'min_count': 1, 'oov_buckets': None, 'top_k': None, 'width':100, 'min_weight':0.0},

    {'in_use': 1, 'name': 'i_advertiser_id', 'min_count': 1, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_industry_id', 'min_count': 1, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_sec_industry_id', 'min_count': 1, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_display_mode_id', 'min_count': 1, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_display_type_id', 'min_count': 1, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_flow_type', 'min_count': 1, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_advertiser_type_id', 'min_count': 1, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_ad_id', 'min_count': 1, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_app_id', 'min_count': 1, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_app_type_id', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_max_spend', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_trade_category', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_pay_type_id', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_install_type_id', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_network_type_id', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_is_h5', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_is_deeplink', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_is_direct', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_sex_tag', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_age_tag', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_region_tag', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_feature_tag', 'min_count': 50, 'oov_buckets': None, 'top_k': None, 'width':10},
    {'in_use': 1, 'name': 'i_root_company', 'min_count': 50, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_last_day_pctr','default': 0},
    {'in_use': 1, 'name': 'i_last_day_pcvr', 'default': 0},
    {'in_use': 1, 'name': 'i_last_3day_pctr','default': 0},
    {'in_use': 1, 'name': 'i_last_3day_pcvr', 'default': 0},
    {'in_use': 1, 'name': 'i_last_7day_pctr','default': 0},
    {'in_use': 1, 'name': 'i_last_7day_pcvr', 'default': 0},
    {'in_use': 1, 'name': 'i_last_30day_pctr', 'default': 0},
    {'in_use': 1, 'name': 'i_last_30day_pcvr', 'default': 0},
    {'in_use': 1, 'name': 'i_last_day_smooth_ctr', 'default': 0},
    {'in_use': 1, 'name': 'i_last_3day_smooth_ctr', 'default': 0},
    {'in_use': 1, 'name': 'i_last_7day_smooth_ctr', 'default': 0},
    {'in_use': 1, 'name': 'i_last_30day_smooth_ctr', 'default': 0}
]


def get_dict_file(column):
    dict_file = column.get('dict_file', None)
    if dict_file is None:
        dict_file = os.path.join(base_dir, name + '.dict')
    return dict_file


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

transform = Transform()

# user
transform.add_categorical_identity('u_model',[])
transform.add_indicator('u_model.identity', dssm1)

transform.add_categorical_identity('u_age',[])      # 7
transform.add_indicator('u_age.identity', dssm1)   # one hot

transform.add_categorical_identity('u_sex', [])  # 2
transform.add_indicator('u_sex.identity', dssm1)  # one hot

transform.add_categorical_identity('u_com_province', [])
transform.add_indicator('u_com_province.identity', dssm1)

transform.add_categorical_identity('u_com_city', [])
transform.add_embedding('u_com_city.identity', dssm1, 10)

transform.add_categorical_identity('u_city_grade', [])
transform.add_indicator('u_city_grade.identity', dssm1)

transform.add_categorical_identity('u_price_zone_type', [])  # 11
transform.add_indicator('u_price_zone_type.identity', dssm1)

transform.add_categorical_identity('u_ads_expo_his', [])   # 96946
transform.add_embedding('u_ads_expo_his.identity', dssm1, 50)

transform.add_categorical_identity('u_ads_click_his', [])  # 96951
transform.add_embedding('u_ads_click_his.identity', dssm1, 50)

transform.add_categorical_identity('u_app_expo_browser_search_his', [])   # 10716
transform.add_embedding('u_app_expo_browser_search_his.identity', dssm1, 10)

transform.add_categorical_identity('u_app_click_browser_search_his', []) 
transform.add_embedding('u_app_click_browser_search_his.identity', dssm1, 10)

transform.add_categorical_identity('u_installed_list_d', [])   # 14257维
transform.add_embedding('u_installed_list_d.identity', dssm1, 10)



# item
transform.add_categorical_identity('i_advertiser_id', [])
transform.add_indicator('i_advertiser_id.identity', dssm2)

transform.add_categorical_identity('i_industry_id', [])
transform.add_indicator('i_industry_id.identity', dssm2)

transform.add_categorical_identity('i_sec_industry_id', [])
transform.add_indicator('i_sec_industry_id.identity', dssm2)

transform.add_categorical_identity('i_display_mode_id', [])
transform.add_indicator('i_display_mode_id.identity', dssm2)

transform.add_categorical_identity('i_display_type_id', [])
transform.add_indicator('i_display_type_id.identity', dssm2)

transform.add_categorical_identity('i_flow_type', [])
transform.add_indicator('i_flow_type.identity', dssm2)

transform.add_categorical_identity('i_advertiser_type_id', [])
transform.add_indicator('i_advertiser_type_id.identity', dssm2)

# transform.add_categorical_identity('i_ad_name', [])
# transform.add_indicator('i_ad_name.identity', dssm2)

transform.add_categorical_identity('i_app_id', [])
transform.add_indicator('i_app_id.identity', dssm2)

transform.add_categorical_identity('i_app_type_id', [])
transform.add_indicator('i_app_type_id.identity', dssm2)

transform.add_categorical_identity('i_max_spend', [])
transform.add_indicator('i_max_spend.identity', dssm2)

transform.add_categorical_identity('i_trade_category', [])
transform.add_indicator('i_trade_category.identity', dssm2)

#transform.add_categorical_identity('i_pay_type_id', [])
#transform.add_indicator('i_pay_type_id.identity', dssm2)

transform.add_categorical_identity('i_install_type_id', [])
transform.add_indicator('i_install_type_id.identity', dssm2)

transform.add_categorical_identity('i_network_type_id', [])
transform.add_indicator('i_network_type_id.identity', dssm2)

transform.add_categorical_identity('i_is_h5', [])
transform.add_indicator('i_is_h5.identity', dssm2)

transform.add_categorical_identity('i_is_deeplink', [])
transform.add_indicator('i_is_deeplink.identity', dssm2)

transform.add_categorical_identity('i_is_direct', [])
transform.add_indicator('i_is_direct.identity', dssm2)

transform.add_categorical_identity('i_sex_tag', [])
transform.add_indicator('i_sex_tag.identity', dssm2)

transform.add_categorical_identity('i_age_tag', [])
transform.add_indicator('i_age_tag.identity', dssm2)

transform.add_categorical_identity('i_region_tag', [])
transform.add_indicator('i_region_tag.identity', dssm2)
#transform.add_categorical_identity('i_feature_tag', [])
#transform.add_embedding('i_feature_tag.identity', dssm2, 10)
transform.add_categorical_identity('i_root_company', [])
transform.add_indicator('i_root_company.identity', dssm2)


transform.add_numeric('i_last_day_pctr', [])
transform.add_bucketized('i_last_day_pctr.numeric', dssm2, [0.1, 0.2, 0.3])   # one hot
# transform.add_embedding('i_last_day_pctr.numeric', dssm2, 5)  # embedding
transform.add_numeric('i_last_day_pcvr', [])
transform.add_bucketized('i_last_day_pcvr.numeric', dssm2, [0.01, 0.02, 0.5])   # one hot
# transform.add_embedding('i_last_day_pcvr.numeric', dssm2, 5)  # embedding

transform.add_numeric('i_last_3day_pctr', [])
transform.add_bucketized('i_last_3day_pctr.numeric', dssm2, [0.08, 0.1, 0.17])
# transform.add_embedding('i_last_3day_pctr.numeric', dssm2, 5)  # embedding
transform.add_numeric('i_last_3day_pcvr', [])
# transform.add_embedding('i_last_3day_pcvr.numeric', dssm2, 5)  # embedding
transform.add_bucketized('i_last_3day_pcvr.numeric', dssm2, [0.034, 0.05, 0.07])


transform.add_numeric('i_last_7day_pctr', [])
transform.add_bucketized('i_last_7day_pctr.numeric', dssm2, [0.08, 0.1, 0.13])
# transform.add_embedding('i_last_7day_pctr.numeric', dssm2, 5)  # embedding
transform.add_numeric('i_last_7day_pcvr', [])
transform.add_bucketized('i_last_7day_pcvr.numeric', dssm2, [0.01, 0.03, 0.05])
# transform.add_embedding('i_last_7day_pcvr.numeric', dssm2, 5)  # embedding


transform.add_numeric('i_last_30day_pctr', [])
transform.add_bucketized('i_last_30day_pctr.numeric', dssm2, [0.1, 0.11, 0.13])
# transform.add_embedding('i_last_30day_pctr.numeric', dssm2, 5)  # embedding
transform.add_numeric('i_last_30day_pcvr', [])
transform.add_bucketized('i_last_30day_pcvr.numeric', dssm2, [0.01, 0.03, 0.08])
# transform.add_embedding('i_last_30day_pcvr.numeric', dssm2, 5)  # embedding

transform.add_numeric('i_last_day_smooth_ctr', [])
transform.add_bucketized('i_last_day_smooth_ctr.numeric', dssm2, [0.03, 0.04, 0.08])
# transform.add_embedding('i_last_day_smooth_ctr.numeric', dssm2, 5)  # embedding

transform.add_numeric('i_last_3day_smooth_ctr', [])
transform.add_bucketized('i_last_3day_smooth_ctr.numeric', dssm2, [0.03, 0.0576, 0.06])
# transform.add_embedding('i_last_3day_smooth_ctr.numeric', dssm2, 5)  # embedding

transform.add_numeric('i_last_7day_smooth_ctr', [])
transform.add_bucketized('i_last_7day_smooth_ctr.numeric', dssm2, [0.03, 0.0576, 0.06])
# transform.add_embedding('i_last_7day_smooth_ctr.numeric', dssm2, 5)  # embedding

transform.add_numeric('i_last_30day_smooth_ctr', [])
transform.add_bucketized('i_last_30day_smooth_ctr.numeric', dssm2, [0.03, 0.0576, 0.06])


model_name = "bizpoint_cvr_v4"
input_data = InputData()
InputData.meta_file = meta_file
cf = Configuration(assembler, transform, input_data, model_name)
print(cf.to_json())
