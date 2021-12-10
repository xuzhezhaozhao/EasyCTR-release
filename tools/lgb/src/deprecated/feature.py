import pandas as pd
import columns as cls
import numpy as np

def get_train_data(source_data):
    source_data.columns = cls.columns
    ##离散化年龄
    source_data['user_age'] = source_data['user_age'].map(lambda x: age_discret(x))
    # ##取用户分类画像权重最高的前5项
    source_data['user_class'] = source_data.apply(lambda row: topk(row['user_class2_weight'], row['user_class2_id'], 5), axis=1)
    user_class_df= source_data['user_class'].str.split(',', expand=True).rename(columns={0: 'user_class2_id_1',
                                                                                         1: 'user_class2_id_2', 2: 'user_class2_id_3',
                                                                                         3: 'user_class2_id_4',4: 'user_class2_id_5'})
    source_data = source_data.drop(['user_class'], axis=1)
    # # ##去用户tag权重最高的前5项
    source_data['user_tag'] = source_data.apply(lambda row: topk(row['user_tag_weight'],row['user_tag_id'], 5), axis=1)
    user_tag_df= source_data['user_tag'].str.split(',', expand=True).rename(columns={0: 'user_tag_id_1',
                                                                                     1: 'user_tag_id_2', 2: 'user_tag_id_3',
                                                                                     3: 'user_tag_id_4',4: 'user_tag_id_5'})
    source_data = source_data.drop(['user_tag'], axis=1)
    # ##取用户浏览历史中浏览权重最高的前5项一级分类
    #source_data['user_cid_history'] = source_data.apply(lambda row: topk(row['user_view_rowkey_weight'], row['user_view_cid_history'], 5), axis=1)
    #user_history_cid_df= source_data['user_cid_history'].str.split(',', expand=True).rename(columns={0: 'user_view_cid_history_1',
    #                                                                                                1: 'user_view_cid_history_2', 2: 'user_view_cid_history_3',
    #                                                                                                 3: 'user_view_cid_history_4',4: 'user_view_cid_history_5'})
    #source_data = source_data.drop(['user_cid_history'], axis=1)
    # # ##取用户浏览历史中浏览权重最高的前5项2级分类
    #source_data['user_cid2_history'] = source_data.apply(lambda row: topk(row['user_view_rowkey_weight'], row['user_view_c2id_history'], 5), axis=1)
    #user_history_cid1_df = source_data['user_cid2_history'].str.split(',', expand=True).rename(columns={0: 'user_view_cid2_history_1',
    #                                                                                                    1: 'user_view_cid2_history_2', 2: 'user_view_cid2_history_3',
    #                                                                                                    3: 'user_view_cid2_history_4',4: 'user_view_cid2_history_5'})
    #source_data = source_data.drop(['user_cid2_history'], axis=1)

    ##获取物品tag
    source_data['video_tag_ids'] = source_data.apply(lambda row: getk(row['video_tag_id'], 5), axis=1)
    video_tag_ids_df = source_data['video_tag_ids'].str.split(',', expand=True).rename(columns={0: 'video_tag_id_1',
                                                                                                        1: 'video_tag_id_2', 2: 'video_tag_id_3',
                                                                                                        3: 'video_tag_id_4', 4: 'video_tag_id_5'})
    source_data = source_data.drop(['video_tag_ids'], axis=1)


    source_data = source_data.drop(cls.drop_columns, axis=1)
    #source_data = pd.concat([source_data, user_class_df, user_tag_df, user_history_cid_df, user_history_cid1_df], axis=1)
    
    source_data = pd.concat([source_data, user_class_df, user_tag_df,video_tag_ids_df], axis=1)

    #source_data = pd.concat([source_data, user_class_df, user_tag_df], axis=1)
    source_data[cls.category_columns_2] = source_data[cls.category_columns_2].astype(float)

    # source_data = source_data.sample(frac=1)

    print(source_data.columns)
    Y = source_data['label']
    X = source_data.drop(['label'], axis=1)
    return X, Y



def topk(weights, ids, k):
    result = ['0' for i in range(k)]
    ids_list = str.split(str(ids), ',')
    weights_list = str.split(str(weights), ',')
    if len(ids_list) == len(weights_list) and str(ids) != '' and str(weights) != '':
        id_weight_dict = dict(zip(weights_list, ids_list))
        tmp = [id_weight_dict[index] for index in sorted(id_weight_dict.keys(), reverse=True)]
        result = [tmp[idx] if len(tmp) > idx and tmp[idx] != '' else '0' for idx, val in enumerate(result)]

    result_str = ",".join(result)
    return result_str

def getk(ids, k):
    result = ['0' for i in range(k)]
    ids_list = str.split(str(ids), ',')
    if str(ids) != '':
        result = [ids_list[idx] if len(ids_list) > idx and ids_list[idx] != '' else '0' for idx, val in enumerate(result)]
    result_str = ','.join(result)
    return result_str



def age_discret(age):
    if age < 6:
        return 0
    elif age < 12:
        return 1
    elif age < 15:
        return 2
    elif age < 18:
        return 3
    elif age < 22:
        return 4
    elif age < 26:
        return 5
    elif age < 30:
        return 6
    elif age < 35:
        return 7
    elif age < 40:
        return 8
    else:
        return 9


if __name__ == '__main__':
    print(topk("0.048677224826094566,0.048677224826094566,0.048677224826094566,0.048677224826094566,0.048677224826094566,0.048677224826094566,0.04353823346687513,0.04353823346687513,0.04353823346687513,0.04353823346687513,0.04353823346687513,0.03679652325789301,0.03679652325789301,0.034419995763873626,0.034419995763873626,0.034419995763873626,0.034419995763873626,0.034419995763873626,0.034419995763873626,0.034419995763873626,0.034419995763873626",
"1912973,1021236,3702837,4073297,108530,108818,66241,53740795,469135,171101,110194,15394105,1573897,30105,57508,532,371411,41368,2779064,17331255,7436",5))
