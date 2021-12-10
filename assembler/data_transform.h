#ifndef ASSEMBLER_DATA_TRANSFORM_H_
#define ASSEMBLER_DATA_TRANSFORM_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/env.h"

#include "assembler/column.h"
#include "assembler/conf_parser.h"
#include "assembler/feature.h"
#include "assembler/lua_sampler.h"

namespace assembler {

struct FeatureInfo {
  std::shared_ptr<BasicColumn> bc;
  int pos;  // 特征在向量中的位置
  FeatureInfo() = default;
  FeatureInfo(const std::shared_ptr<BasicColumn>& bc_, int pos_)
      : bc(bc_), pos(pos_) {}
};

class DataTransform {
 public:
  explicit DataTransform(::tensorflow::OpKernelConstruction* ctx);

  void Init(const ConfParser& cp, bool use_lua_sampler,
            const std::string& lua_sampler_script);
  // 格式错误返回 false, 否则返回 true
  bool FillFeature(const std::vector<StringPiece>& tokens, bool need_label,
                   Feature* feature, double* label, double* weight);

  // only used in training
  bool FillFeatureWithNegativeSampling(const std::vector<StringPiece>& tokens,
                                       std::vector<Example>* examples);

  size_t feature_size() const { return feature_size_; }
  const std::vector<std::pair<std::shared_ptr<BasicColumn>, Field>>& columns()
      const {
    return columns_;
  }
  const Feature& feature_def() const { return feature_def_; }

  bool Transform(const std::string& input, bool is_predict,
                 Example* example);
  bool TransformWithNegativeSampling(const std::string& input,
                                     std::vector<Example>* examples);

  void ServingTransform(const std::string& user_feature,
                        const std::vector<std::string>& item_features,
                        bool is_recall, std::vector<Feature>* features);
  void Serialize(xJson::Value* s) const;
  void ReConstruct(const xJson::Value& root);

 private:
  DataTransform(const DataTransform&) = delete;
  void operator=(const DataTransform&) = delete;

  ::tensorflow::OpKernelConstruction* ctx_;
  // id->FeatureInfo
  // multimap 处理 alias 功能
  std::unordered_multimap<std::string, FeatureInfo> column_map_;
  // 与 feature 向量的顺序相同
  std::vector<std::pair<std::shared_ptr<BasicColumn>, Field>> columns_;
  size_t feature_size_ = 0;
  Feature feature_def_;  // 存储默认特征向量

  bool use_lua_sampler_ = false;
  std::string lua_sampler_script_;
  LuaSampler lua_sampler_;
  // feature id -> params pos
  std::unordered_map<std::string, int> lua_params_map_;
};

}  // namespace assembler

#endif  // ASSEMBLER_DATA_TRANSFORM_H_
