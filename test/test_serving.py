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


user_feature = [""]
# item_features = ["101|558;102|19774;103|51739337;104|2649.0943;105|19.1883;106|0;107|19774;108|0;109|558;110|0.0000;111|1.0000;112|0.0000;113|1.0000;258|60988600000;259|51739337;260|20332;261|115881300000;262|42821143.9464;114|180;115|0;116|14;117|9;118|0;119|0;120|5;263|609000000;264|0;265|0;266|0;267|0;268|0;269|0;216|532927;217|8;218|2819162;219|-1;220|110001;221|220004;222|500000;223|1;224|网服;225|2;226|2;227|2;228|2;229|1;230|北京字节跳动网络技术有限公司;231|是;232|否;233|1;234|0;235|2;236|5;237|121;238|-1;239|all;240|all;241|all;242|all;270|20814293;271|8;356|558;357|558;358|19774;359|51739337;360|51739337;361|2649.0943;362|19.1883;363|7.5125;364|1674000000;365|3180600000;366|42821143.9464;367|0;368|19774;369|0;370|558;371|0.0000;372|1.0000;373|0.0000;374|1.0000;379|0.0000;380|0.0000;381|0.0000;382|0.0000;383|0.0000;384|0.0000;385|0.0000;386|0.0000;387|0.0000;388|0.0000;389|0.0000;390|0.0000;482|0;483|0;484|0;485|0;486|3000000;487|0;488|0;489|5700000;490|0;491|0;492|0;493|0;494|0.0000;495|0.0000;496|0.0000;497|0.0000;498|17.2464;499|0.0000;500|0.0000;501|9.0771;502|0.0000;503|0.0000;504|0.0000;505|0.0000;561|0.0000;562|0.0000;563|0.0000;564|0.0000;565|1552875521.0000;566|0.0000;391|180;392|0;393|14;394|0;395|9;396|0;397|0;398|0;399|0;400|0;401|0;402|0;506|540000000;507|0;508|42000000;509|0;510|27000000;511|0;512|0;513|0;514|0;515|0;516|0;517|0;285|0.0340;286|0.3998;287|0.0000;288|0.0774;289|0.0007;290|0.0444;291|0.0000;292|0.0000;293|0.0116;294|0.0000;295|0.0000;296|0.0002;297|0.0000", "", ""]
item_features = [""]

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
                'item_features': _bytes_feature(item_features),
                'is_recall': _int64_feature([0]),
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
