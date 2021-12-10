# encoding:utf-8

from tqdm import tqdm
import json

'''
1000 128

ID_1 0.016 0.012 0.025 ...

ID_2 0.021 0.051 0.105 ...

...

ID_1000 0.015 0.156 0.098 ...
'''

global_predict_file_path = './data/predict_output.txt'
global_adid_file_path = './data/adid_feature.txt'
global_serialize_name = './data/ad_id.vec'


# 传出一个 [adid, vector]
def read_predict_files(predict_filename, adid_filename, aim):
    ad_vector_file = open(predict_filename, 'r')
    ad_file = open(adid_filename, 'r')

    adids_vectors_dict = dict()

    adids = []
    adid_vectors = []

    for line in tqdm(ad_vector_file):
        adid_vectors.append(json.loads(line)[aim])

    for line in tqdm(ad_file):
        adids.append(line.split('\t')[0].split()[0].split('|')[1])

    for i in range(len(adids)):
        if adids[i] not in adids_vectors_dict:
            adids_vectors_dict[adids[i]] = adid_vectors[i]

    return adid_vectors, adids, adids_vectors_dict

# 传输需要的格式 [ adid_num, vector_dim ] [ adid, vector ] , [...] , [ adid, vector ]
# 并序列化到磁盘
def transform_vector_format_serialization(adid_vectors, adids, global_serialize_name, adids_vectors_dict):
    if len(adid_vectors) == 0:
        print("adid_vectors is zero, please check global_predict_file_path's file")
    adid_num = len(adids_vectors_dict.keys())
    vector_dim = len(adid_vectors[0])
    append_file = open(global_serialize_name, 'a')
    append_file.write(str(adid_num) + ' ' + str(vector_dim) + '\n')
    for ad_key in list(adids_vectors_dict.keys()):
        tmp_string = ''
        tmp_string += str(adids_vectors_dict[ad_key]).strip('[').strip(']')
        append_file.write(str(ad_key) + ' ' + tmp_string + '\n')
    print("finished serialization !")



if __name__ == '__main__':
    adid_vectors, adids, adids_vectors_dict = read_predict_files(global_predict_file_path, global_adid_file_path, 'itower')
    transform_vector_format_serialization(adid_vectors, adids, global_serialize_name, adids_vectors_dict)
