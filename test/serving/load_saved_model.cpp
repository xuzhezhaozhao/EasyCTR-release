
#include <fstream>
#include <iostream>
#include <string>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/platform/init_main.h"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: <model_dir>" << std::endl;
    exit(-1);
  }
  std::string model_dir = argv[1];
  tensorflow::SavedModelBundle bundle;

  if (!tensorflow::MaybeSavedModelDirectory(model_dir)) {
    std::cout << "No valid model" << std::endl;
    exit(-1);
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  tensorflow::SessionOptions options;

  // tensorflow::ConfigProto& config = options.config;
  // config.set_intra_op_parallelism_threads(24);
  // config.set_inter_op_parallelism_threads(24);

  const std::unordered_set<std::string> tags = {
      tensorflow::kSavedModelTagServe};
  tensorflow::Status status = tensorflow::LoadSavedModel(
      options, tensorflow::RunOptions(), model_dir, tags, &bundle);
  if (!status.ok()) {
    std::cout << "Load model failed, errmsg = " << status.error_message()
              << std::endl;
    exit(-1);
  }

  /////////////////////////////////
  tensorflow::MetaGraphDef meta_graph_def;
  status = ReadMetaGraphDefFromSavedModel(model_dir, tags, &meta_graph_def);
  if (!status.ok()) {
    std::cout << "read meta graph def failed, errmsg = "
              << status.error_message() << std::endl;
    exit(-1);
  }
  const auto& signature_def = meta_graph_def.signature_def();
  for (auto& item : signature_def) {
    std::cout << "signature: " << item.first << std::endl;
  }
  /////////////////////////////////

  tensorflow::Example example;
  auto feature = example.mutable_features()->mutable_feature();
  tensorflow::Feature user_feature, item_features;
  user_feature.mutable_bytes_list()->add_value("");
  item_features.mutable_bytes_list()->add_value("");
  (*feature)["user_feature"] = user_feature;
  (*feature)["item_features"] = item_features;
  std::string serialized;
  example.SerializeToString(&serialized);
  tensorflow::Tensor input_tensor(tensorflow::DT_STRING,
                                  tensorflow::TensorShape({1}));
  input_tensor.flat<std::string>()(0) = serialized;
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;

  // Get input and output tensor name:
  // saved_model_cli show --dir export_model_dir/1585038452/ --tag_set serve
  // --signature_def serving_default
  const auto& inputs_info = signature_def.at("serving_default").inputs();
  for (auto& item : inputs_info) {
    std::cout << "inputs key = " << item.first << std::endl;
    std::cout << "inputs name = " << item.second.name() << std::endl;
    // key is map key
    // name is tensor name
    // such as
    // serialized_tf_example = tf.placeholder(
    //   dtype=tf.string, shape=[None], name='input_example_tensor')
    // receiver_tensors = {'inputs': serialized_tf_example}
    inputs.push_back({item.second.name(), input_tensor});
  }

  std::vector<std::string> output_names;
  // signature 'serving_default' is in Request
  const auto& outputs_info = signature_def.at("serving_default").outputs();
  for (auto& item : outputs_info) {
    std::cout << "outputs key = " << item.first << std::endl;
    std::cout << "outputs name = " << item.second.name() << std::endl;
    output_names.push_back(item.second.name());
  }

  std::vector<tensorflow::Tensor> outputs;

  bool profile = true;

  if (profile) {
    tensorflow::RunOptions options;
    options.set_trace_level(tensorflow::RunOptions_TraceLevel_FULL_TRACE);
    tensorflow::RunMetadata meta_data;
    status = bundle.session->Run(options, inputs, output_names, {}, &outputs,
                                 &meta_data);
    if (!status.ok()) {
      std::cout << "Run error, errmsg = " << status.error_message()
                << std::endl;
      exit(-1);
    }
    std::string out_str;
    meta_data.step_stats().SerializeToString(&out_str);
    std::string file_name = "profile.log";
    std::ofstream ofs(file_name);
    ofs << out_str;
    ofs.close();
  } else {
    status = bundle.session->Run(inputs, output_names, {}, &outputs);
    if (!status.ok()) {
      std::cout << "Run error, errmsg = " << status.error_message()
                << std::endl;
      exit(-1);
    }
  }

  auto f = outputs[0].flat<float>();
  for (int i = 0; i < f.size(); i++) {
    std::cout << f(i) << std::endl;
  }

  return 0;
}
