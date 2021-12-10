#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from math import log

sys.path.append("../../tools/conf_generator")

from conf_generator import Assembler, Transform, InputData, Configuration


# base v3, no substract mean


base_dir = './data/dict'
meta_file = './data/data.meta'

wide = ['wide']
deep = ['deep']
wide_deep = ['wide', 'deep']

# parse dtypes from meta
dtypes = {'label': 'float', 'weight': 'float'}
for lineindex, line in enumerate(open(meta_file)):
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
    {'in_use': 1, 'name': 'i_click_cnt_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_click_cnt_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_click_cnt_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_expo_cnt_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_expo_cnt_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_spent_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_spent_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_spent_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_pnclick_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_pnclick_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_pnclick_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_pncv_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_pncv_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_pncv_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_xg_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_expo_cnt_xg_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_expo_cnt_xg_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_wifi_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_expo_cnt_wifi_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_expo_cnt_wifi_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_click_cnt_xg_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_click_cnt_xg_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_click_cnt_xg_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_click_cnt_wifi_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_click_cnt_wifi_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_click_cnt_wifi_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_expo_rate_xg_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_expo_rate_xg_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_expo_rate_xg_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_expo_rate_wifi_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_expo_rate_wifi_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_expo_rate_wifi_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_click_rate_xg_cd_0h_to_cm_acc', 'default': 0},
    {'in_use': 1, 'name': 'i_click_rate_xg_cd_0h_to_cm_acc', 'default': 0, 'alias': 'i_click_rate_xg_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_click_rate_wifi_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_click_rate_wifi_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_click_rate_wifi_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_dld_cnt_1d_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_dld_cnt_1d_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_dld_cnt_1d_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_reg_cnt_1d_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_reg_cnt_1d_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_reg_cnt_1d_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_active_cnt_1d_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_active_cnt_1d_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_active_cnt_1d_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_custom_active_cnt_1d_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_custom_active_cnt_1d_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_custom_active_cnt_1d_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_1d_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_1d_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_custom_reg_cnt_1d_cd_0h_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_submit_cnt_1d_cd_0h_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_submit_cnt_1d_cd_0h_to_cm_acc', 'default': 0.0, 'alias': 'i_submit_cnt_1d_cd_0h_to_cm_acc_v2'},
    {'in_use': 0, 'name': 'i_cvtype', 'default': 0.0},
    {'in_use': 0, 'name': 'i_cvtype', 'default': 0.0, 'alias': 'i_cvtype_v2'},
    {'in_use': 1, 'name': 'i_click_cnt_cd_0h_to_b1h_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_click_cnt_cd_0h_to_b1h_acc', 'default': 0.0, 'alias': 'i_click_cnt_cd_0h_to_b1h_acc_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_cd_0h_to_b1h_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_expo_cnt_cd_0h_to_b1h_acc', 'default': 0.0, 'alias': 'i_expo_cnt_cd_0h_to_b1h_acc_v2'},
    {'in_use': 1, 'name': 'i_spent_cd_0h_to_b1h_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_spent_cd_0h_to_b1h_acc', 'default': 0.0, 'alias': 'i_spent_cd_0h_to_b1h_acc_v2'},
    {'in_use': 1, 'name': 'i_pnclick_cd_0h_to_b1h_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_pnclick_cd_0h_to_b1h_acc', 'default': 0.0, 'alias': 'i_pnclick_cd_0h_to_b1h_acc_v2'},
    {'in_use': 1, 'name': 'i_pncv_cd_0h_to_b1h_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_pncv_cd_0h_to_b1h_acc', 'default': 0.0, 'alias': 'i_pncv_cd_0h_to_b1h_acc_v2'},
    {'in_use': 1, 'name': 'i_click_cnt_cd_ch_0m_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_click_cnt_cd_ch_0m_to_cm_acc', 'default': 0.0, 'alias': 'i_click_cnt_cd_ch_0m_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_cd_ch_0m_to_cm_acc', 'default': 0},
    {'in_use': 1, 'name': 'i_expo_cnt_cd_ch_0m_to_cm_acc', 'default': 0, 'alias': 'i_expo_cnt_cd_ch_0m_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_spent_cd_ch_0m_to_cm_acc', 'default': 0},
    {'in_use': 1, 'name': 'i_spent_cd_ch_0m_to_cm_acc', 'default': 0, 'alias': 'i_spent_cd_ch_0m_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_pnclick_cd_ch_0m_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_pnclick_cd_ch_0m_to_cm_acc', 'default': 0.0, 'alias': 'i_pnclick_cd_ch_0m_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_pncv_cd_ch_0m_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_pncv_cd_ch_0m_to_cm_acc', 'default': 0.0, 'alias': 'i_pncv_cd_ch_0m_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_dld_cnt_1d_cd_ch_0m_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_dld_cnt_1d_cd_ch_0m_to_cm_acc', 'default': 0.0, 'alias': 'i_dld_cnt_1d_cd_ch_0m_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_reg_cnt_1d_cd_ch_0m_to_cm_acc', 'default': 0},
    {'in_use': 1, 'name': 'i_reg_cnt_1d_cd_ch_0m_to_cm_acc', 'default': 0, 'alias': 'i_reg_cnt_1d_cd_ch_0m_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_active_cnt_1d_cd_ch_0m_to_cm_acc', 'default': 0},
    {'in_use': 1, 'name': 'i_active_cnt_1d_cd_ch_0m_to_cm_acc', 'default': 0, 'alias': 'i_active_cnt_1d_cd_ch_0m_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_custom_active_cnt_1d_cd_ch_0m_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_custom_active_cnt_1d_cd_ch_0m_to_cm_acc', 'default': 0.0, 'alias': 'i_custom_active_cnt_1d_cd_ch_0m_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_1d_cd_ch_0m_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_1d_cd_ch_0m_to_cm_acc', 'default': 0.0, 'alias': 'i_custom_reg_cnt_1d_cd_ch_0m_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_submit_cnt_1d_cd_ch_0m_to_cm_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_submit_cnt_1d_cd_ch_0m_to_cm_acc', 'default': 0.0, 'alias': 'i_submit_cnt_1d_cd_ch_0m_to_cm_acc_v2'},
    {'in_use': 1, 'name': 'i_click_cnt_b1d_0h_to_ch_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_click_cnt_b1d_0h_to_ch_acc', 'default': 0.0, 'alias': 'i_click_cnt_b1d_0h_to_ch_acc_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_b1d_0h_to_ch_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_expo_cnt_b1d_0h_to_ch_acc', 'default': 0.0, 'alias': 'i_expo_cnt_b1d_0h_to_ch_acc_v2'},
    {'in_use': 1, 'name': 'i_spent_b1d_0h_to_ch_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_spent_b1d_0h_to_ch_acc', 'default': 0.0, 'alias': 'i_spent_b1d_0h_to_ch_acc_v2'},
    {'in_use': 1, 'name': 'i_pnclick_b1d_0h_to_ch_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_pnclick_b1d_0h_to_ch_acc', 'default': 0.0, 'alias': 'i_pnclick_b1d_0h_to_ch_acc_v2'},
    {'in_use': 1, 'name': 'i_pncv_b1d_0h_to_ch_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_pncv_b1d_0h_to_ch_acc', 'default': 0.0, 'alias': 'i_pncv_b1d_0h_to_ch_acc_v2'},
    {'in_use': 1, 'name': 'i_dld_cnt_3d_b1d_0h_to_ch_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_dld_cnt_3d_b1d_0h_to_ch_acc', 'default': 0.0, 'alias': 'i_dld_cnt_3d_b1d_0h_to_ch_acc_v2'},
    {'in_use': 1, 'name': 'i_reg_cnt_3d_b1d_0h_to_ch_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_reg_cnt_3d_b1d_0h_to_ch_acc', 'default': 0.0, 'alias': 'i_reg_cnt_3d_b1d_0h_to_ch_acc_v2'},
    {'in_use': 1, 'name': 'i_active_cnt_3d_b1d_0h_to_ch_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_active_cnt_3d_b1d_0h_to_ch_acc', 'default': 0.0, 'alias': 'i_active_cnt_3d_b1d_0h_to_ch_acc_v2'},
    {'in_use': 1, 'name': 'i_custom_active_cnt_3d_b1d_0h_to_ch_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_custom_active_cnt_3d_b1d_0h_to_ch_acc', 'default': 0.0, 'alias': 'i_custom_active_cnt_3d_b1d_0h_to_ch_acc_v2'},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_3d_b1d_0h_to_ch_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_3d_b1d_0h_to_ch_acc', 'default': 0.0, 'alias': 'i_custom_reg_cnt_3d_b1d_0h_to_ch_acc_v2'},
    {'in_use': 1, 'name': 'i_submit_cnt_3d_b1d_0h_to_ch_acc', 'default': 0.0},
    {'in_use': 1, 'name': 'i_submit_cnt_3d_b1d_0h_to_ch_acc', 'default': 0.0, 'alias': 'i_submit_cnt_3d_b1d_0h_to_ch_acc_v2'},
    {'in_use': 1, 'name': 'i_click_cnt_b1d_ch', 'default': 0.0},
    {'in_use': 1, 'name': 'i_click_cnt_b1d_ch', 'default': 0.0, 'alias': 'i_click_cnt_b1d_ch_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_b1d_ch', 'default': 0},
    {'in_use': 1, 'name': 'i_expo_cnt_b1d_ch', 'default': 0, 'alias': 'i_expo_cnt_b1d_ch_v2'},
    {'in_use': 1, 'name': 'i_spent_b1d_ch', 'default': 0},
    {'in_use': 1, 'name': 'i_spent_b1d_ch', 'default': 0, 'alias': 'i_spent_b1d_ch_v2'},
    {'in_use': 1, 'name': 'i_pnclick_b1d_ch', 'default': 0},
    {'in_use': 1, 'name': 'i_pnclick_b1d_ch', 'default': 0, 'alias': 'i_pnclick_b1d_ch_v2'},
    {'in_use': 1, 'name': 'i_pncv_b1d_ch', 'default': 0},
    {'in_use': 1, 'name': 'i_pncv_b1d_ch', 'default': 0, 'alias': 'i_pncv_b1d_ch_v2'},
    {'in_use': 1, 'name': 'i_dld_cnt_3d_b1d_ch', 'default': 0.0},
    {'in_use': 1, 'name': 'i_dld_cnt_3d_b1d_ch', 'default': 0.0, 'alias': 'i_dld_cnt_3d_b1d_ch_v2'},
    {'in_use': 1, 'name': 'i_reg_cnt_3d_b1d_ch', 'default': 0.0},
    {'in_use': 1, 'name': 'i_reg_cnt_3d_b1d_ch', 'default': 0.0, 'alias': 'i_reg_cnt_3d_b1d_ch_v2'},
    {'in_use': 1, 'name': 'i_active_cnt_3d_b1d_ch', 'default': 0.0},
    {'in_use': 1, 'name': 'i_active_cnt_3d_b1d_ch', 'default': 0.0, 'alias': 'i_active_cnt_3d_b1d_ch_v2'},
    {'in_use': 1, 'name': 'i_custom_active_cnt_3d_b1d_ch', 'default': 0.0},
    {'in_use': 1, 'name': 'i_custom_active_cnt_3d_b1d_ch', 'default': 0.0, 'alias': 'i_custom_active_cnt_3d_b1d_ch_v2'},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_3d_b1d_ch', 'default': 0},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_3d_b1d_ch', 'default': 0, 'alias': 'i_custom_reg_cnt_3d_b1d_ch_v2'},
    {'in_use': 1, 'name': 'i_submit_cnt_3d_b1d_ch', 'default': 0},
    {'in_use': 1, 'name': 'i_submit_cnt_3d_b1d_ch', 'default': 0, 'alias': 'i_submit_cnt_3d_b1d_ch_v2'},
    {'in_use': 1, 'name': 'i_click_cnt_b3d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_click_cnt_b3d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_click_cnt_b3d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_b3d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_expo_cnt_b3d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_expo_cnt_b3d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_spent_b3d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_spent_b3d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_spent_b3d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_pnclick_b3d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_pnclick_b3d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_pnclick_b3d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_pncv_b3d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_pncv_b3d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_pncv_b3d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_dld_cnt_3d_b3d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_dld_cnt_3d_b3d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_dld_cnt_3d_b3d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_reg_cnt_3d_b3d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_reg_cnt_3d_b3d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_reg_cnt_3d_b3d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_active_cnt_3d_b3d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_active_cnt_3d_b3d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_active_cnt_3d_b3d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_custom_active_cnt_3d_b3d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_custom_active_cnt_3d_b3d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_custom_active_cnt_3d_b3d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_3d_b3d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_3d_b3d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_custom_reg_cnt_3d_b3d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_submit_cnt_3d_b3d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_submit_cnt_3d_b3d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_submit_cnt_3d_b3d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_click_cnt_b7d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_click_cnt_b7d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_click_cnt_b7d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_b7d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_expo_cnt_b7d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_expo_cnt_b7d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_spent_b7d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_spent_b7d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_spent_b7d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_pnclick_b7d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_pnclick_b7d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_pnclick_b7d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_pncv_b7d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_pncv_b7d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_pncv_b7d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_dld_cnt_3d_b7d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_dld_cnt_3d_b7d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_dld_cnt_3d_b7d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_reg_cnt_3d_b7d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_reg_cnt_3d_b7d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_reg_cnt_3d_b7d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_active_cnt_3d_b7d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_active_cnt_3d_b7d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_active_cnt_3d_b7d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_custom_active_cnt_3d_b7d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_custom_active_cnt_3d_b7d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_custom_active_cnt_3d_b7d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_3d_b7d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_3d_b7d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_custom_reg_cnt_3d_b7d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_submit_cnt_3d_b7d_to_b1d_ch_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_submit_cnt_3d_b7d_to_b1d_ch_avg', 'default': 0, 'alias': 'i_submit_cnt_3d_b7d_to_b1d_ch_avg_v2'},
    {'in_use': 1, 'name': 'i_nclick_cnt_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_nclick_cnt_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_nclick_cnt_b3d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_expo_cnt_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_expo_cnt_b3d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_spent_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_spent_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_spent_b3d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_pnclick_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_pnclick_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_pnclick_b3d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_pncv_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_pncv_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_pncv_b3d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_dld_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_dld_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_dld_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_reg_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_reg_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_reg_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_active_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_active_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_active_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_custom_active_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_custom_active_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_custom_active_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_custom_reg_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_submit_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_submit_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_submit_cnt_3d_b3d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_click_cnt_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_click_cnt_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_click_cnt_b7d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_expo_cnt_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_expo_cnt_b7d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_spent_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_spent_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_spent_b7d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_pnclick_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_pnclick_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_pnclick_b7d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_pncv_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_pncv_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_pncv_b7d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_dld_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_dld_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_dld_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_reg_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_reg_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_reg_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_active_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_active_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_active_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_custom_active_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_custom_active_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_custom_active_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_custom_reg_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_custom_reg_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_submit_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_submit_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg', 'default': 0, 'alias': 'i_submit_cnt_3d_b7d_to_b1d_0h_to_ch_acc_avg_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_xg_b7d_to_b1d_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_expo_cnt_xg_b7d_to_b1d_avg', 'default': 0, 'alias': 'i_expo_cnt_xg_b7d_to_b1d_avg_v2'},
    {'in_use': 1, 'name': 'i_expo_cnt_wifi_b7d_to_b1d_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_expo_cnt_wifi_b7d_to_b1d_avg', 'default': 0, 'alias': 'i_expo_cnt_wifi_b7d_to_b1d_avg_v2'},
    {'in_use': 1, 'name': 'i_expo_b7d_to_b1d_per_imei', 'default': 0},
    {'in_use': 1, 'name': 'i_expo_b7d_to_b1d_per_imei', 'default': 0, 'alias': 'i_expo_b7d_to_b1d_per_imei_v2'},
    {'in_use': 1, 'name': 'i_n4G_b7d_to_b1d_rate', 'default': 0},
    {'in_use': 1, 'name': 'i_n4G_b7d_to_b1d_rate', 'default': 0, 'alias': 'i_n4G_b7d_to_b1d_rate_v2'},
    {'in_use': 1, 'name': 'i_nwifi_b7d_to_b1d_rate', 'default': 0},
    {'in_use': 1, 'name': 'i_nwifi_b7d_to_b1d_rate', 'default': 0, 'alias': 'i_nwifi_b7d_to_b1d_rate_v2'},
    {'in_use': 1, 'name': 'i_nimei_b7d_to_b1d_avg', 'default': 0},
    {'in_use': 1, 'name': 'i_nimei_b7d_to_b1d_avg', 'default': 0, 'alias': 'i_nimei_b7d_to_b1d_avg_v2'},
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

# transform.add_numeric('i_click_cnt_cd_0h_to_cm_acc',wide_deep,'log-norm',log(830.85),log(2623.59))
transform.add_numeric('i_click_cnt_cd_0h_to_cm_acc_v2',wide_deep,'log-norm',0.0,log(2623.59))
# transform.add_numeric('i_expo_cnt_cd_0h_to_cm_acc',wide_deep,'log-norm',log(18660.81),log(48074.94))
transform.add_numeric('i_expo_cnt_cd_0h_to_cm_acc_v2',wide_deep,'log-norm',0.0,log(48074.94))
# transform.add_numeric('i_spent_cd_0h_to_cm_acc',wide_deep,'log-norm',log(20935091.73),log(64823460.35))
transform.add_numeric('i_spent_cd_0h_to_cm_acc_v2',wide_deep,'log-norm',0.0,log(64823460.35))
# transform.add_numeric('i_pnclick_cd_0h_to_cm_acc',wide_deep,'log-norm',log(1807.62),log(6720.54))
transform.add_numeric('i_pnclick_cd_0h_to_cm_acc_v2',wide_deep,'log-norm',0.0,log(6720.54))
transform.add_numeric('i_pncv_cd_0h_to_cm_acc',[])
transform.add_bucketized('i_pncv_cd_0h_to_cm_acc.numeric', ['selector'], [8, 50, 300])

transform.add_numeric('i_pncv_cd_0h_to_cm_acc_v2',wide_deep,'log-norm',0.0,log(102.67))
# transform.add_numeric('i_dld_cnt_1d_cd_0h_to_cm_acc',wide_deep,'log-norm',log(330.81),log(1293.40))
transform.add_numeric('i_dld_cnt_1d_cd_0h_to_cm_acc_v2',wide_deep,'log-norm',0.0,log(1293.40))
# transform.add_numeric('i_active_cnt_1d_cd_0h_to_cm_acc',wide_deep,'log-norm',log(15.17),log(95.54))
transform.add_numeric('i_active_cnt_1d_cd_0h_to_cm_acc_v2',wide_deep,'log-norm',0.0,log(95.54))
# transform.add_numeric('i_custom_active_cnt_1d_cd_0h_to_cm_acc',wide_deep,'log-norm',log(14.05),log(76.07))
# transform.add_numeric('i_custom_active_cnt_1d_cd_0h_to_cm_acc_v2',wide_deep,'log-norm',0.0,log(76.07))
# transform.add_categorical_identity('i_ad_group_id', [])
# transform.add_embedding('i_ad_group_id.identity', wide_deep, 10)
# transform.add_categorical_identity('i_app_id', [])
# transform.add_embedding('i_app_id.identity', wide_deep, 20)
# transform.add_categorical_identity('i_app_type_id', [])
# transform.add_indicator('i_app_type_id.identity', wide_deep)
# transform.add_categorical_identity('i_industry_id', [])
# transform.add_embedding('i_industry_id.identity', wide_deep, 5)
# transform.add_categorical_identity('i_sec_industry_id', [])
# transform.add_embedding('i_sec_industry_id.identity', wide_deep, 5)
# transform.add_categorical_identity('i_trade_category', [])
# transform.add_indicator('i_trade_category.identity', wide_deep)
# transform.add_categorical_identity('i_install_type_id', [])
# transform.add_indicator('i_install_type_id.identity', wide_deep)
# transform.add_categorical_identity('i_advertiser_type_id', [])
# transform.add_indicator('i_advertiser_type_id.identity', wide_deep)
# transform.add_categorical_identity('i_is_h5', [])
# transform.add_indicator('i_is_h5.identity', wide_deep)
# transform.add_categorical_identity('i_is_deeplink', [])
# transform.add_indicator('i_is_deeplink.identity', wide_deep)
# transform.add_categorical_identity('i_display_mode_id', [])
# transform.add_indicator('i_display_mode_id.identity',wide_deep)
# transform.add_categorical_identity('i_is_direct', [])
# transform.add_indicator('i_is_direct.identity', wide_deep)

model_name = ""
input_data = InputData()
InputData.meta_file = meta_file
cf = Configuration(assembler, transform, input_data, model_name)
print(cf.to_json())
