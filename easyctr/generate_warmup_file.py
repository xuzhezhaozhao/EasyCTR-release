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


MODEL_NAME = "easyctr"
user_feature = [""]
item_features = [""]


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# see https://www.tensorflow.org/tfx/serving/saved_model_warmup
def main():
    with tf.python_io.TFRecordWriter("tf_serving_warmup_requests") as writer:
        # replace <request> with one of:
        # predict_pb2.PredictRequest(..)
        # classification_pb2.ClassificationRequest(..)
        # regression_pb2.RegressionRequest(..)
        # inference_pb2.MultiInferenceRequest(..)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = MODEL_NAME

        request.model_spec.signature_name = 'serving_default'
        example1 = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'user_feature': _bytes_feature(user_feature),
                    'item_features': _bytes_feature(item_features),
                    'is_recall': _int64_feature([0]),
                }
            )
        ).SerializeToString()
        print("example len = {}".format(len(example1)))
        examples = [example1]
        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(examples, dtype=tf.string))
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())


if __name__ == "__main__":
    main()
