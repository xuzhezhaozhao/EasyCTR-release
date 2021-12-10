import tensorflow as tf
import numpy as np
import sys

np.set_printoptions(threshold='nan')

if len(sys.argv) != 3:
    print("usage: <model_dir> <param_name>")
    sys.exit(-1)

checkpoint_path = tf.train.latest_checkpoint(sys.argv[1])
key = sys.argv[2]

reader = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

if key in var_to_shape_map:
    tensor = reader.get_tensor(key)
    shape = np.shape(tensor)
    print("name = {}, shape = {}".format(key, shape))
    print(tensor)
