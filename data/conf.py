#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from math import log

sys.path.append("./tools/conf_generator")

from conf_generator import Assembler, Transform, InputData, Configuration


# based on v6, no spent and pncv features


base_dir = './data/dict'
meta_file = './data/data.meta'

wide = ['wide']
deep = ['deep']
wide_deep = ['wide', 'deep']
selector = ['selector']
dssm1 = ['dssm1', 'deep']
dssm2 = ['dssm2', 'deep']

# parse dtypes from meta
dtypes = {'label': 'float', 'weight': 'float'}
for lineindex, line in enumerate(open('./data.meta')):
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
    {'in_use': 1, 'name': 'i_cvtype', 'default': 0.0},
    {'in_use': 1, 'name': 'i_cvtype', 'default': 0.0, 'alias': 'i_cvtype_v2'},
    {'in_use': 1, 'name': 'i_cvtype', 'default': 0.0, 'alias': 'i_cvtype_selector'},
    {'in_use': 1, 'name': 'i_adid', 'min_count': 2, 'oov_buckets': 100, 'top_k': None},
    {'in_use': 1, 'name': 'i_ad_group_id', 'min_count': 2, 'oov_buckets': 100, 'top_k': None},
    {'in_use': 1, 'name': 'i_ad_group_id', 'min_count': 2, 'oov_buckets': 100, 'top_k': None, 'alias': 'i_ad_group_id_v2'},
    {'in_use': 1, 'name': 'i_app_id', 'min_count': 2, 'oov_buckets': 100, 'top_k': None},
    {'in_use': 1, 'name': 'i_app_id', 'min_count': 2, 'oov_buckets': 100, 'top_k': None, 'alias': 'i_app_id_v2'},
    {'in_use': 1, 'name': 'i_app_type_id', 'min_count': 2, 'oov_buckets': None, 'top_k': None },
    {'in_use': 1, 'name': 'i_app_type_id', 'min_count': 2, 'oov_buckets': None, 'top_k': None , 'alias': 'i_app_type_id_v2'},
    {'in_use': 1, 'name': 'i_industry_id', 'min_count': 2, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'i_industry_id', 'min_count': 2, 'oov_buckets': 10, 'top_k': None, 'alias': 'i_industry_id_v2'},
    {'in_use': 1, 'name': 'i_sec_industry_id', 'min_count': 2, 'oov_buckets': 10, 'top_k': None},
    {'in_use': 1, 'name': 'i_sec_industry_id', 'min_count': 2, 'oov_buckets': 10, 'top_k': None, 'alias': 'i_sec_industry_id_v2'},
    {'in_use': 1, 'name': 'i_max_spend', 'default': 0.0},
    {'in_use': 1, 'name': 'i_max_spend', 'default': 0.0, 'alias': 'i_max_spend_v2'},
    {'in_use': 1, 'name': 'i_flow_type', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_flow_type', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_flow_type_v2'},
    {'in_use': 1, 'name': 'i_trade_category', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_trade_category', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_trade_category_v2'},
    {'in_use': 0, 'name': 'i_ad_promote_type_id', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 0, 'name': 'i_ad_promote_type_id', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_ad_promote_type_id_v2'},
    {'in_use': 0, 'name': 'i_pay_type_id', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 0, 'name': 'i_pay_type_id', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_pay_type_id_v2'},
    {'in_use': 1, 'name': 'i_install_type_id', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_install_type_id', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_install_type_id_v2'},
    {'in_use': 1, 'name': 'i_network_type_id', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_network_type_id', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_network_type_id_v2'},
    {'in_use': 1, 'name': 'i_advertiser_type_id', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_advertiser_type_id', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_advertiser_type_id_v2'},
    {'in_use': 1, 'name': 'i_root_company', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_root_company', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_root_company_v2'},
    {'in_use': 1, 'name': 'i_is_h5', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_is_h5', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_is_h5_v2'},
    {'in_use': 1, 'name': 'i_is_deeplink', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_is_deeplink', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_is_deeplink_v2'},
    {'in_use': 1, 'name': 'i_display_mode_id', 'min_count': 2, 'oov_buckets':None, 'top_k': None},
    {'in_use': 1, 'name': 'i_display_mode_id', 'min_count': 2, 'oov_buckets':None, 'top_k': None, 'alias': 'i_display_mode_id_v2'},
    {'in_use': 1, 'name': 'i_is_direct', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_is_direct', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_is_direct_v2'},
    {'in_use': 1, 'name': 'i_is_installed', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_is_installed', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_is_installed_v2'},
    {'in_use': 1, 'name': 'i_ocpc_conversion_type', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_ocpc_conversion_type', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_ocpc_conversion_type_v2'},
    {'in_use': 1, 'name': 'i_start_now_duration_day', 'default': 0},
    {'in_use': 1, 'name': 'i_start_now_duration_day', 'default': 0, 'alias': 'i_start_now_duration_day_v2'},
    {'in_use': 1, 'name': 'i_ocpc_conversion_deep_type', 'min_count': 2, 'oov_buckets': None, 'top_k': None},
    {'in_use': 1, 'name': 'i_ocpc_conversion_deep_type', 'min_count': 2, 'oov_buckets': None, 'top_k': None, 'alias': 'i_ocpc_conversion_deep_type_v2'},
    {'in_use': 1, 'name': 'i_gender_oritention', 'min_count': 20, 'width': 3},
    {'in_use': 1, 'name': 'i_gender_oritention', 'min_count': 20, 'width': 3, 'alias': 'i_gender_oritention_v2'},
    {'in_use': 1, 'name': 'i_age_oritention', 'min_count': 20, 'width': 20},
    {'in_use': 1, 'name': 'i_age_oritention', 'min_count': 20, 'width': 20, 'alias': 'i_age_oritention_v2'},
    {'in_use': 1, 'name': 'i_crowd_oritention', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'i_crowd_oritention', 'min_count': 20, 'width': 200, 'alias': 'i_crowd_oritention_v2'},
    {'in_use': 1, 'name': 'i_location_oritention', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'i_location_oritention', 'min_count': 20, 'width': 200, 'alias': 'i_location_oritention_v2'},

    {'in_use': 1, 'name': 'u_session_cpc_click_ads', 'min_count': 5, 'width': 200, 'min_weight': 0.0},
    {'in_use': 1, 'name': 'u_session_cpd_click_ads', 'min_count': 5, 'width': 200, 'min_weight'
: 0.0},
    {'in_use': 1, 'name': 'u_session_feed_click_articles', 'min_count': 5, 'width': 200, 'min_weight': 0.0},
    {'in_use': 1, 'name': 'u_session_feed_click_videos', 'min_count': 5, 'width': 200, 'min_weight': 0.0},
    #{'in_use': 1, 'name': 'u_session_feed_play_videos', 'min_count': 20, 'width': 200},
    #{'in_use': 1, 'name': 'u_session_installed_apps', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_session_used_apps', 'min_count': 5, 'width': 200, 'min_weight': 0.0},
    #{'in_use': 1, 'name': 'u_session_installed_apps_by_duration', 'min_count': 20, 'width': 200},

    {'in_use': 1, 'name': 'u_phone_model', 'min_count': 5, 'width': 200},
    {'in_use': 1, 'name': 'u_age', 'min_count': 5, 'width': 200},
    {'in_use': 1, 'name': 'u_gender', 'min_count': 5, 'width': 200},
    {'in_use': 1, 'name': 'u_region', 'min_count': 5, 'width': 200},
    {'in_use': 1, 'name': 'u_com_province', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_com_city', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_phone_series', 'min_count': 20, 'width': 200},
    #{'in_use': 1, 'name': 'u_phone_price_zone', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_phone_prize_zone_type', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_current_province', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_current_city', 'min_count': 20, 'width': 200},

    {'in_use': 1, 'name': 'u_is_cpc_day_active', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_is_cpc_week_active', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_is_cpc_month_active', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_is_cpd_day_active', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_is_cpd_week_active', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_is_cpd_month_active', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_is_brand_day_active', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_is_brand_week_active', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'u_is_brand_month_active', 'min_count': 20, 'width': 200},

    {'in_use': 1, 'name': 'c_net', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'c_device_model', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'c_vc', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'c_media_uuid', 'min_count': 20, 'width': 200},
    {'in_use': 1, 'name': 'c_position_uuids', 'min_count': 20, 'width': 200},
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
        print(column['name'])
        raise e

transform = Transform()

# transform.add_numeric('i_cvtype_selector',[])
# transform.add_bucketized('i_cvtype_selector.numeric',selector, [100])
transform.add_categorical_identity('i_adid', [])
transform.add_embedding('i_adid.identity', dssm2,10)
transform.add_categorical_identity('i_ad_group_id', [])
transform.add_embedding('i_ad_group_id.identity', dssm2,10)
transform.add_categorical_identity('i_app_id', [])
transform.add_embedding('i_app_id.identity', dssm2, 10)
transform.add_categorical_identity('i_app_type_id', [])
#transform.add_indicator('i_app_type_id.identity', dssm2)
#transform.add_categorical_identity('i_industry_id', [])
#transform.add_indicator('i_industry_id.identity', dssm2)
transform.add_categorical_identity('i_sec_industry_id', [])
transform.add_embedding('i_sec_industry_id.identity', dssm2, 4)
#transform.add_categorical_identity('i_trade_category', [])
#transform.add_indicator('i_trade_category.identity', dssm2)
transform.add_categorical_identity('i_install_type_id', [])
#transform.add_indicator('i_install_type_id.identity', dssm2)
transform.add_categorical_identity('i_advertiser_type_id', [])
#transform.add_indicator('i_advertiser_type_id.identity', dssm2)
transform.add_categorical_identity('i_is_h5', [])
#transform.add_indicator('i_is_h5.identity', dssm2)
transform.add_categorical_identity('i_is_deeplink', [])
#transform.add_indicator('i_is_deeplink.identity', dssm2)
transform.add_categorical_identity('i_display_mode_id', [])
#transform.add_indicator('i_display_mode_id.identity',dssm2)
transform.add_categorical_identity('i_is_direct', [])
#transform.add_indicator('i_is_direct.identity', dssm2)
transform.add_categorical_identity('i_is_installed', [])
#transform.add_indicator('i_is_installed.identity', dssm2)
transform.add_categorical_identity('i_ocpc_conversion_type', [])
#transform.add_indicator('i_ocpc_conversion_type.identity', dssm2)

transform.add_categorical_identity('u_session_cpc_click_ads', [])
transform.add_embedding('u_session_cpc_click_ads.identity', dssm1,10)
transform.add_categorical_identity('u_session_cpd_click_ads', [])
transform.add_embedding('u_session_cpd_click_ads.identity', dssm1,10)
transform.add_categorical_identity('u_session_feed_click_articles', [])
transform.add_embedding('u_session_feed_click_articles.identity', dssm1,10)
transform.add_categorical_identity('u_session_feed_click_videos', [])
transform.add_embedding('u_session_feed_click_videos.identity', dssm1,10)
#transform.add_categorical_identity('u_session_feed_play_videos', [])
#transform.add_embedding('u_session_feed_play_videos.identity', dssm1,100)
#transform.add_categorical_identity('u_session_installed_apps', [])
#transform.add_embedding('u_session_installed_apps.identity', dssm1,100)
transform.add_categorical_identity('u_session_used_apps', [])
transform.add_embedding('u_session_used_apps.identity', dssm1,10)
#transform.add_categorical_identity('u_session_installed_apps_by_duration', [])
#transform.add_embedding('u_session_installed_apps_by_duration.identity', dssm1,100)

transform.add_categorical_identity('u_phone_model', [])
transform.add_embedding('u_phone_model.identity', dssm1,10)
transform.add_categorical_identity('u_age', [])
#transform.add_indicator('u_age.identity', dssm1)
transform.add_categorical_identity('u_gender', [])
#transform.add_indicator('u_gender.identity', dssm1)
transform.add_categorical_identity('u_region', [])
#transform.add_indicator('u_region.identity', dssm1)
transform.add_categorical_identity('u_com_province', [])
#transform.add_indicator('u_com_province.identity', dssm1)
transform.add_categorical_identity('u_com_city', [])
transform.add_embedding('u_com_city.identity', dssm1,8)
transform.add_categorical_identity('u_phone_series', [])
transform.add_embedding('u_phone_series.identity', dssm1,2)
transform.add_categorical_identity('u_phone_prize_zone_type', [])
#transform.add_indicator('u_phone_prize_zone_type.identity', dssm1)
transform.add_categorical_identity('u_current_province', [])
#transform.add_indicator('u_current_province.identity', dssm1)
transform.add_categorical_identity('u_current_city', [])
transform.add_embedding('u_current_city.identity', dssm1,8)

#transform.add_categorical_identity('u_is_cpc_day_active', [])
#transform.add_indicator('u_is_cpc_day_active.identity', dssm1)
#transform.add_categorical_identity('u_is_cpc_week_active', [])
#transform.add_indicator('u_is_cpc_week_active.identity', dssm1)
#transform.add_categorical_identity('u_is_cpc_month_active', [])
#transform.add_indicator('u_is_cpc_month_active.identity', dssm1)
#transform.add_categorical_identity('u_is_cpd_day_active', [])
#transform.add_indicator('u_is_cpd_day_active.identity', dssm1)
#transform.add_categorical_identity('u_is_cpd_week_active', [])
#transform.add_indicator('u_is_cpd_week_active.identity', dssm1)
#transform.add_categorical_identity('u_is_cpd_month_active', [])
#transform.add_indicator('u_is_cpd_month_active.identity', dssm1)
#transform.add_categorical_identity('u_is_brand_day_active', [])
#transform.add_indicator('u_is_brand_day_active.identity', dssm1)
#transform.add_categorical_identity('u_is_brand_week_active', [])
#transform.add_indicator('u_is_brand_week_active.identity', dssm1)
#transform.add_categorical_identity('u_is_brand_month_active', [])
#transform.add_indicator('u_is_brand_month_active.identity', dssm1)

transform.add_categorical_identity('c_media_uuid', [])
transform.add_indicator('c_media_uuid.identity', dssm1)
transform.add_categorical_identity('c_position_uuids', [])
transform.add_embedding('c_position_uuids.identity', dssm1,20)

model_name = "uni-cvr"
input_data = InputData()
InputData.meta_file = meta_file
cf = Configuration(assembler, transform, input_data, model_name)
print(cf.to_json())
