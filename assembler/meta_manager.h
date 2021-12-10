#ifndef ASSEMBLER_META_MANAGER_H_
#define ASSEMBLER_META_MANAGER_H_

#include <string>
#include <unordered_map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/env.h"

#include "assembler/feature.h"

namespace assembler {

class MetaManager {
 public:
  explicit MetaManager(::tensorflow::Env* env);

  void Init(const std::string& meta_path);
  const std::unordered_map<std::string, Field>& meta() const { return meta_; }
  const std::unordered_map<std::string, std::string>& name2id() const {
    return name2id_;
  }
  const std::unordered_map<std::string, std::string>& id2name() const {
    return id2name_;
  }

 private:
  MetaManager(const MetaManager&) = delete;
  void operator=(const MetaManager&) = delete;

  ::tensorflow::Env* env_;
  std::unordered_map<std::string, Field> meta_;           // id to Field
  std::unordered_map<std::string, std::string> name2id_;  // name to id
  std::unordered_map<std::string, std::string> id2name_;  // id to name
};

}  // namespace assembler

#endif  // ASSEMBLER_META_MANAGER_H_
