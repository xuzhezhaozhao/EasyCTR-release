import tensorflow as tf
import numpy as np
import sys

np.set_printoptions(threshold='nan')

if len(sys.argv) != 2:
    print("usage: <model_dir>")
    sys.exit(-1)

checkpoint_path = tf.train.latest_checkpoint(sys.argv[1])

reader = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
total_parameters = 0


def print_feature(key):
    global total_parameters
    tensor = reader.get_tensor(key)
    shape = np.shape(tensor)
    zero_fraction = 1.0 * np.count_nonzero(tensor == 0.0) / tensor.size
    print("name = {}, shape = {}, zero_fraction = {:.2f}".format(key, shape, zero_fraction))
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim
    total_parameters += variable_parameters

    if variable_parameters < 200:
        print(tensor)
        print("\n")


norm_features = []
cross_features = []
for key in var_to_shape_map:
    # if key.find('/Ftrl') >= 0 or key.find('/Adagrad') >= 0:
        # continue

    if key.find('_X_') >= 0:
        cross_features.append(key)
    else:
        norm_features.append(key)

norm_features = sorted(norm_features)
for key in norm_features:
    print_feature(key)

print("\n\n------ cross features -----\n")
cross_features = sorted(cross_features)
for key in cross_features:
    print_feature(key)


print('param num: %d' % total_parameters)
