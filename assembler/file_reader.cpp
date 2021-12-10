#include "assembler/file_reader.h"

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"

namespace assembler {

FileReader::FileReader(::tensorflow::Env* env, const std::string& filename,
                       uint64_t buffer_size)
    : env_(env), filename_(filename), buffer_size_(buffer_size) {}

bool FileReader::Init() {
  ::tensorflow::Status s = env_->NewRandomAccessFile(filename_, &file_);
  if (!s.ok()) {
    return false;
  }

  input_stream_ = absl::make_unique<::tensorflow::io::RandomAccessInputStream>(
      file_.get(), false);

  buffered_input_stream_ =
      absl::make_unique<::tensorflow::io::BufferedInputStream>(
          input_stream_.get(), buffer_size_, false);

  return true;
}

bool FileReader::ReadLine(std::string* result) {
  ::tensorflow::Status s = buffered_input_stream_->ReadLine(result);
  return s.ok();
}

bool FileReader::ReadAll(std::string* result) {
  ::tensorflow::Status s = buffered_input_stream_->ReadAll(result);
  return s.ok();
}

}  // namespace assembler
