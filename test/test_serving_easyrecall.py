#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import grpc
import tensorflow as tf
import tensorflow_serving.apis.predict_pb2 as predict_pb2
import tensorflow_serving.apis.prediction_service_pb2_grpc as prediction_service_pb2_grpc

MODEL_NAME = "easyctr"


# user_feature = ["318|1\t317|20\t322|1\t336|4755e007d3b533ah,2715df8b63d955ah,8985df75f09637ah"]
user_feature = ["545|21208444:13975801,21672068:13878904,20510279:12522524,20768788:12262298"]


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def run():
    channel = grpc.insecure_channel('127.0.0.1:9000')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME

    request.model_spec.signature_name = 'serving_default'
    input_name = 'inputs'
    example1 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'user_feature': _bytes_feature(user_feature),
                'item_features': _bytes_feature([""]),
            }
        )
    ).SerializeToString()

    print("example len = {}".format(len(example1)))
    examples = [example1]
    request.inputs[input_name].CopyFrom(
        tf.contrib.util.make_tensor_proto(examples, dtype=tf.string))

    response = stub.Predict(request)
    print("Received: \n{}".format(response))


if __name__ == '__main__':
    run()
