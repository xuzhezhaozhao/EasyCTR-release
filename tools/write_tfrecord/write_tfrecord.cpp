#include <fstream>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/posix/posix_file_system.h"

#include "assembler/assembler.h"

void write_tfrecord(const std::string& input_file,
                    const std::string& tfrecord_file,
                    assembler::Assembler& assembler) {
  ::tensorflow::Env* env = ::tensorflow::Env::Default();
  std::unique_ptr<::tensorflow::WritableFile> file;
  TF_CHECK_OK(env->NewWritableFile(tfrecord_file, &file));
  ::tensorflow::io::RecordWriter writer(file.get());

  std::ifstream ifs(input_file);
  if (!ifs.is_open()) {
    LOG(FATAL) << "open " << input_file << " failed.";
  }
  int lineindex = 0;
  std::string line;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    ++lineindex;
    if (lineindex % 200000 == 0) {
      std::cout << "Write " << lineindex / 10000 << "w lines ..." << std::endl;
    }
    if (line == "") {
      continue;
    }
    int dim = assembler.feature_size();
    assembler::Example example(assembler::Feature(dim), 0.0, 0.0);
    assembler.GetExample(line, true, &example);
    ::tensorflow::Example texample;
    ::tensorflow::Feature label, weight, inputs;
    label.mutable_float_list()->add_value(example.label);
    weight.mutable_float_list()->add_value(example.weight);
    for (auto f : example.feature) {
      inputs.mutable_float_list()->add_value(f);
    }
    auto feature_map = texample.mutable_features()->mutable_feature();
    (*feature_map)["label"] = label;
    (*feature_map)["weight"] = weight;
    (*feature_map)["inputs"] = inputs;

    std::string serialized;
    texample.SerializeToString(&serialized);
    TF_CHECK_OK(writer.WriteRecord(serialized));
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cout << "Usage: <conf> <input> <output>" << std::endl;
    exit(-1);
  }
  std::string conf_path = std::string(argv[1]);
  std::string input_file = std::string(argv[2]);
  std::string output_file = std::string(argv[3]);

  // TODO Create real ctx
  assembler::Assembler assembler(NULL);
  assembler.Init(conf_path);
  write_tfrecord(input_file, output_file, assembler);
  return 0;
}
