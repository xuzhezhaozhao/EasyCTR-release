#ifndef ASSEMBLER_CONF_PARSER_H_
#define ASSEMBLER_CONF_PARSER_H_

#include <memory>
#include <string>
#include <vector>

#include "deps/jsoncpp/json/json.h"
#include "tensorflow/core/platform/env.h"

#include "assembler/column.h"

namespace assembler {

class ConfParser {
 public:
  struct DataPaths {
    std::string meta_file;
  };

  explicit ConfParser(::tensorflow::OpKernelConstruction* ctx) : ctx_(ctx) {}

  bool Parse(const std::string& conf_path, bool load_dict = true);
  const std::vector<std::shared_ptr<BasicColumn>>& columns() const {
    return columns_;
  }
  std::vector<std::shared_ptr<BasicColumn>>& columns() { return columns_; }
  const DataPaths& data_paths() const { return data_paths_; }
  DataPaths& data_paths() { return data_paths_; }
  void PrintDebugInfo();

 private:
  ConfParser(const ConfParser&) = delete;
  void operator=(const ConfParser&) = delete;

  bool CreateNumericColumn(const xJson::Value&, Field::Type type,
                           std::shared_ptr<BasicColumn>*);
  bool CreateStringColumn(const xJson::Value&, std::shared_ptr<BasicColumn>*,
                          bool load_dict);
  bool CreateStringListColumn(const xJson::Value&,
                              std::shared_ptr<BasicColumn>*, bool load_dict);
  bool CreateFloatListColumn(const xJson::Value&,
                             std::shared_ptr<BasicColumn>*);
  bool CreateWeightedStringListColumn(const xJson::Value&,
                                      std::shared_ptr<BasicColumn>*,
                                      bool load_dict);
  bool CreateTripleListColumn(const xJson::Value&,
                              std::shared_ptr<BasicColumn>*, bool load_dict);

  ::tensorflow::OpKernelConstruction* ctx_;
  std::vector<std::shared_ptr<BasicColumn>> columns_;
  DataPaths data_paths_;
};

}  // namespace assembler

#endif  // ASSEMBLER_CONF_PARSER_H_
