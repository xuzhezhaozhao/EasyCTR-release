#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import inference_pb2
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_serving.apis import regression_pb2


user_feature = [""]
item_features = ["", "", ""]


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# see https://www.tensorflow.org/tfx/serving/saved_model_warmup
def main():
    with open("predictor_warmup_requests", 'wb') as writer:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'user_feature': _bytes_feature(user_feature),
                    'item_features': _bytes_feature(item_features),
                    'is_recall': _int64_feature([0]),
                }
            )
        ).SerializeToString()
        writer.write(example)


if __name__ == "__main__":
    main()
