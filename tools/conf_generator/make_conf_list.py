#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import itertools
from conf_generator import Assembler, Transform, InputData, Configuration

def get_single(transform, assembler, input_data, model_name):

    json_list = []
    for block in transform.blocks:

        if block["places"] is  None or block["places"] == []:
            continue
        (transform_alone, inames) = get_linked_blocks(block, transform)
        json_list.append(blocks_to_json(inames, assembler, transform_alone, input_data, model_name + ":" + block["output"]))
    return json_list

def get_all_cross(transform, assembler, input_data, model_name, auto_bucket_size=False):
    json_list = []
    single_list = [block for block in transform.blocks if block["places"] == ["wide"]]
    combined_list = itertools.combinations(single_list, 2)
    for combination in combined_list:
        cross_model_name = model_name + ":" + combination[0]["output"] + "*" + combination[1]["output"]
        c1 = combination[0]
        c2 = combination[1]
        c1["places"] = []
        c2["places"] = []
        (transform_1, inames_1) = get_linked_blocks(c1, transform)
        (transform_2, inames_2) = get_linked_blocks(c2, transform)
        if auto_bucket_size == False:
            bucket_size = 100000
        else:
            bucket_size = get_bucket_size
        cross_transform = Transform()
        cross_transform.add_cross([combination[0]["output"], combination[1]["output"]], bucket_size)
        cross_block = cross_transform.blocks
        all_blocks = transform_1 + transform_2 + cross_block
        all_iname = inames_1 + inames_2
        json_list.append(blocks_to_json(all_iname, assembler, all_blocks, input_data, cross_model_name))
    return json_list

def get_user_item_cross(transform, assembler, input_data, model_name, auto_bucket_size=False)
    json_list = []
    single_list = [block for block in transform.blocks if block["places"] == ["wide"] ]
    user_single_list = [block for block in single_list if (block["output"].startswith("u") or block["output"].startswith("c"))]
    item_single_list = [block for block in single_list if block["output"].startswith("i")]
    combined_list = itertools.product(user_single_list, item_single_list)
    for combination in combined_list:
        cross_model_name = model_name + ":" + combination[0]["output"] + "*" + combination[1]["output"]
        c1 = combination[0]
        c2 = combination[1]
        c1["places"] = []
        c2["places"] = []
        (transform_1, inames_1) = get_linked_blocks(c1, transform)
        (transform_2, inames_2) = get_linked_blocks(c2, transform)
        if auto_bucket_size == False:
            bucket_size = 100000
        else:
            bucket_size = get_bucket_size
        cross_transform = Transform()
        cross_transform.add_cross([combination[0]["output"], combination[1]["output"]], bucket_size)
        cross_block = cross_transform.blocks
        all_blocks = transform_1 + transform_2 + cross_block
        all_iname = inames_1 + inames_2
        json_list.append(blocks_to_json(all_iname, assembler, all_blocks, input_data, cross_model_name))
    return json_list

def get_kfm_cross(transform, assembler, input_data, model_name):
    json_list = []
    single_list = [block for block in transform.blocks if block["places"] == ["deep"]]
    combined_list = itertools.combinations(single_list, 2)
    for combination in combined_list:
        cross_model_name = model_name + ":" + combination[0]["output"] + "*" + combination[1]["output"]
        c1 = combination[0]
        c2 = combination[1]
        (transform_1, inames_1) = get_linked_blocks(c1, transform)
        (transform_2, inames_2) = get_linked_blocks(c2, transform)
        all_blocks = transform_1 + transform_2
        all_iname = inames_1 + inames_2
        json_list.append(blocks_to_json(all_iname, assembler, all_blocks, input_data, cross_model_name))
    return json_list

def get_bucket_size(): #TODO
    return 100000

def get_linked_blocks(head, transform):
    this_block = head
    inames = []
    linked_list = []
    while True:
            #transform_alone.append(this_block)
            linked_list = [this_block] + linked_list
            inames.append(this_block["input"])
            last_block = get_block_by_output(transform.blocks, this_block["input"])
            if last_block is None:
                break
            else:
                this_block = last_block
    return (linked_list, inames)


def get_block_by_output(blocks, output):
    for block in blocks:
        if block["output"] == output:
            return block
    return None


def blocks_to_json(inames, assembler, transform_blocks, input_data, model_name):
    obj = {}
    obj['assembler'] = assembler.to_obj(use_blocks=inames)
    obj['transform'] = transform_blocks
    obj['input_data'] = input_data.to_obj()

    if model_name is not None:
        obj['model_name'] = model_name
    return json.dumps(obj, indent=4)


def main():

    base_dir = './data/data'
    meta_file = 'data.meta'

    wide = ['wide']
    wide = ['wide']
    wide_wide = ['wide', 'wide']

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
        {'in_use': 1, 'name': 'u_kd_base_uin', 'min_count': 20, 'oov_buckets': None, 'top_k': None},
        {'in_use': 1, 'name': 'u_kd_base_gender', 'min_count': 20, 'oov_buckets': None, 'top_k': None},
        {'in_use': 1, 'name': 'u_kd_base_age', 'default': 0},
        {'in_use': 1, 'name': 'u_kd_base_city_level', 'min_count': 20, 'oov_buckets': None, 'top_k': None},
        {'in_use': 1, 'name': 'u_kd_behavior_tl_exposure_history', 'min_count': 20, 'width': 200},
        {'in_use': 1, 'name': 'u_kd_behavior_tl_click_history', 'min_count': 20, 'width': 200},
        {'in_use': 1, 'name': 'u_kd_behavior_pos_view_history', 'min_count': 20, 'width': 200},
        {'in_use': 1, 'name': 'u_kd_behavior_neg_view_history', 'min_count': 20, 'width': 200},
        {'in_use': 1, 'name': 'u_kd_portrait_cid2', 'min_count': 20, 'width': 10, 'min_weight': 0.3, 'oov_buckets': None, 'top_k': None},
        {'in_use': 1, 'name': 'u_kd_portrait_tags', 'min_count': 20, 'width': 20, "min_weight": 0.3, 'oov_buckets': None, 'top_k': None},

        {'in_use': 1, 'name': 'i_kd_rowkey', 'min_count': 20, 'oov_buckets': None, 'top_k': None},
        {'in_use': 1, 'name': 'i_kd_cid1', 'min_count': 20, 'oov_buckets': None, 'top_k': None},
        {'in_use': 1, 'name': 'i_kd_cid2', 'min_count': 20, 'oov_buckets': None, 'top_k': None},
        {'in_use': 1, 'name': 'i_kd_cid3', 'min_count': 0, 'oov_buckets': None, 'top_k': None},
        {'in_use': 1, 'name': 'i_kd_union_tags', 'min_count': 20, 'width': 20, 'oov_buckets': None, 'top_k': None},

        {'in_use': 1, 'name': 'i_kd_duration', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_clarity', 'min_count': 20, 'oov_buckets': None, 'top_k': None},
        {'in_use': 1, 'name': 'i_kd_cover_score', 'min_count': 20, 'oov_buckets': None, 'top_k': None},
        {'in_use': 1, 'name': 'i_kd_subscribe_id', 'min_count': 20, 'oov_buckets': None, 'top_k': None},

        {'in_use': 1, 'name': 'i_kd_in_ts', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_quality', 'min_count': 20, 'oov_buckets': None, 'top_k': None},

        {'in_use': 1, 'name': 'i_kd_exposure_cnt_1d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_click_cnt_1d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_play_cnt_1d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_valid_play_cnt_1d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_like_cnt_1d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_comment_cnt_1d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_valid_play_rate_1d', 'default': 0},

        {'in_use': 1, 'name': 'i_kd_exposure_cnt_3d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_click_cnt_3d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_play_cnt_3d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_valid_play_cnt_3d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_like_cnt_3d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_comment_cnt_3d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_valid_play_rate_3d', 'default': 0},

        {'in_use': 1, 'name': 'i_kd_exposure_cnt_7d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_click_cnt_7d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_play_cnt_7d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_valid_play_cnt_7d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_like_cnt_7d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_comment_cnt_7d', 'default': 0},
        {'in_use': 1, 'name': 'i_kd_valid_play_rate_7d', 'default': 0},
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
                assembler.add_int(name, column['default'])
            elif dtype == 'float':
                assembler.add_float(name, column['default'])
            elif dtype == 'string':
                assembler.add_string(
                    iname=name,
                    dict_file=get_dict_file(column),
                    min_count=column['min_count'],
                    oov_buckets=column.get('oov_buckets', None),
                    top_k=column.get('top_k', None))
            elif dtype == 'weighted_string_list':
                assembler.add_weighted_string_list(
                    iname=name,
                    dict_file=get_dict_file(column),
                    min_count=column['min_count'],
                    width=column['width'],
                    top_k=column.get('top_k', None),
                    min_weight=column['min_weight'],
                    oov_buckets=column.get('oov_buckets', None))
            elif dtype == 'string_list':
                assembler.add_string_list(
                    iname=name,
                    dict_file=get_dict_file(column),
                    min_count=column['min_count'],
                    width=column['width'],
                    oov_buckets=column.get('oov_buckets', None),
                    top_k=column.get('top_k', None))
            else:
                raise ValueError("Unknow dtype '{}'".format(dtype))
        except Exception as e:
            print(column)
            raise e


    # {{{ Transform =========================================================
    transform = Transform()

    # user
    transform.add_categorical_identity('u_kd_base_gender', wide)

    transform.add_numeric('u_kd_base_age')
    transform.add_bucketized('u_kd_base_age.numeric', wide, [10, 13, 16, 20, 25, 30, 35, 40, 50])

    transform.add_categorical_identity('u_kd_base_city_level', wide)

    transform.add_categorical_identity('u_kd_behavior_pos_view_history', wide)

    transform.add_categorical_identity('u_kd_portrait_cid2', wide)

    transform.add_categorical_identity('u_kd_portrait_tags', wide)

    transform.add_categorical_identity('u_kd_behavior_tl_exposure_history', wide)

    transform.add_categorical_identity('u_kd_behavior_tl_click_history', wide)

    # transform.add_categorical_identity('u_kd_base_uin', [])

    # item
    transform.add_categorical_identity('i_kd_rowkey', wide)

    transform.add_numeric('i_kd_duration')
    transform.add_bucketized('i_kd_duration.numeric', wide, [12, 20, 40, 60, 120, 240, 480])

    transform.add_categorical_identity('i_kd_cid1', wide)

    transform.add_categorical_identity('i_kd_cid2', wide)

    transform.add_categorical_identity('i_kd_union_tags', wide)

    model_name = "qqkd_timeline_video_v1"
    input_data = InputData()
    InputData.meta_file = meta_file
    # cf = Configuration(assembler, transform, input_data, model_name)
    # print(cf.to_json())

    #json_list = get_single(transform, assembler, input_data, model_name)
    json_list = get_all_cross(transform, assembler, input_data, model_name)
    print("\n##\n".join(json_list))


if __name__ == '__main__':
    main()