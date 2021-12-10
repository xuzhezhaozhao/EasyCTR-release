#ifndef TOOLS_STRING_INDEXER_STRING_INDEXER_H_
#define TOOLS_STRING_INDEXER_STRING_INDEXER_H_

#include "assembler/assembler.h"
#include "assembler/meta_manager.h"
#include "tensorflow/core/platform/env.h"

namespace assembler {
namespace tools {

class StringIndexer {
 public:
  StringIndexer(::tensorflow::Env* env, const std::string& meta_path,
                const std::string& output_dir)
      : mm_(env), output_dir_(output_dir) {
    mm_.Init(meta_path);
  }
  void Index(const std::string& data_path);
  void WriteToFiles();

 private:
  StringIndexer(const StringIndexer&) = delete;
  void operator=(const StringIndexer&) = delete;

  MetaManager mm_;
  // feature -> (word -> int)
  std::map<std::string, std::map<std::string, int>> indexer_;

  struct Stat {
    double sum;
    double squares_sum;
    double log_sum;
    double log_squares_sum;
    int64_t cnt;
  };

  // pair: sum(x), sum(x^2)
  std::map<std::string, Stat> stats_;

  std::string output_dir_;
};

}  // namespace tools
}  // namespace assembler

#endif  // TOOLS_STRING_INDEXER_STRING_INDEXER_H_
