#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from conf_generator import Assembler, Transform, InputData, Configuration


meta_file = './data.meta'

# see: https://git.code.oa.com/kb_recommend_rank_video/EasyCTR
# Add features to assembler
assembler = Assembler()
assembler.add_string_list(
    iname='i_kd_union_tags',
    dict_file='i_kd_union_tags.dict',
    min_count=1,
    width=5,
    force_add=True)

assembler.add_weighted_string_list(
    iname='u_kd_portrait_tags',
    dict_file='u_kd_portrait_tags.dict',
    width=2,
    reverse=False,
    scan_from='tail',
    force_add=True)

assembler.add_triple_list(
    iname='u_kd_behavior_play_history_longv',
    filter_type='default_filter',
    dict_file='u_kd_behavior_play_history_longv.dict',
    width_pos=2,
    width_neg=2,
    scan_from='head',
    reverse=True,
)

assembler.add_int('u_kd_base_age', 25)


transform = Transform()
transform.add_categorical_identity('u_kd_behavior_play_history_longv.pos', [])
transform.add_numeric('u_kd_base_age', normalizer_fn='norm', substract=0, denominator=1, exp=0.5)

input_data = InputData()
InputData.meta_file = meta_file
cf = Configuration(assembler, transform, input_data)
print(cf.to_json())
