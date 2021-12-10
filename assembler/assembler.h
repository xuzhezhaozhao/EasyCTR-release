#ifndef ASSEMBLER_ASSEMBLER_H_
#define ASSEMBLER_ASSEMBLER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "deps/jsoncpp/json/json.h"

#include "assembler/column.h"
#include "assembler/conf_parser.h"
#include "assembler/data_transform.h"
#include "assembler/feature.h"

namespace assembler {

class Assembler {
 public:
  explicit Assembler(::tensorflow::OpKernelConstruction* ctx);

  // Train
  void Init(const std::string& conf_path, bool use_lua_sampler = false,
            const std::string& lua_sampler_script = "", bool load_dict = true,
            bool debug = false);
  // if is_predict is true, label is not needed
  bool GetExample(const std::string& input, bool is_predict, Example* example);
  xJson::Value GetFeatureScheme() const;
  size_t feature_size() const { return transform_.feature_size(); }
  const std::vector<std::pair<std::shared_ptr<BasicColumn>, Field>>& columns()
      const {
    return transform_.columns();
  }

  // Negative Sampling
  void InitNegativeSampling(const std::string& nce_items_path,
                            int nce_items_min_count, int nce_items_top_k,
                            int seed);
  bool GetExamplesWithNegativeSampling(const std::string& input,
                                       std::vector<Example>* examples);

  // Serving
  void GetServingInputs(const std::string& user_feature,
                        const std::vector<std::string>& item_features,
                        bool is_recall, std::vector<Feature>* features);
  void Serialize(std::string* output) const;
  bool ParseFromString(const std::string& input);
  void PrintDebugInfo() const;
  void PrintExample(const Example& e) const;

 private:
  Assembler(const Assembler&) = delete;
  void operator=(const Assembler&) = delete;

  void InitTableNegatives(const std::vector<int64_t>& counts);
  int GetNegative(int target);

  ::tensorflow::OpKernelConstruction* ctx_;
  DataTransform transform_;
  // negative sampling
  std::vector<int> negatives_;
  std::minstd_rand rng_;
  size_t negpos_;
  std::vector<Feature> nce_features_;
  std::vector<std::string> nce_items_;

  static const int32_t NEGATIVE_TABLE_SIZE = 10000000;
};

}  // namespace assembler

#endif  // ASSEMBLER_ASSEMBLER_H_
