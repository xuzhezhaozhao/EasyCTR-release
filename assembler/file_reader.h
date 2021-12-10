#ifndef ASSEMBLER_FILE_READER_H_
#define ASSEMBLER_FILE_READER_H_

#include <memory>
#include <string>

#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"

namespace assembler {

class FileReader {
 public:
  explicit FileReader(::tensorflow::Env* env, const std::string& filename,
                      uint64_t buffer_size = 4096);
  bool Init();
  bool ReadLine(std::string* result);
  bool ReadAll(std::string* result);

 private:
  FileReader(const FileReader&) = delete;
  void operator=(const FileReader&) = delete;

  ::tensorflow::Env* env_;
  std::string filename_;
  uint64_t buffer_size_;
  std::unique_ptr<::tensorflow::io::RandomAccessInputStream> input_stream_;
  std::unique_ptr<::tensorflow::io::BufferedInputStream> buffered_input_stream_;
  std::unique_ptr<::tensorflow::RandomAccessFile> file_;
};

}  // namespace assembler

#endif  // ASSEMBLER_FILE_READER_H_
