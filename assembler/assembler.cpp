#include "assembler/assembler.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

#include "assembler/file_reader.h"
#include "assembler/jsoncpp_helper.hpp"
#include "assembler/utils.h"

namespace assembler {

Assembler::Assembler(::tensorflow::OpKernelConstruction* ctx)
    : ctx_(ctx), transform_(ctx), negpos_(0) {}

void Assembler::Init(const std::string& conf_path, bool use_lua_sampler,
                     const std::string& lua_sampler_script, bool load_dict,
                     bool debug) {
  ConfParser cp(ctx_);
  if (!cp.Parse(conf_path, load_dict)) {
    LOG(ERROR) << "Parse conf error.";
    throw std::logic_error("Parse conf error");
  }
  transform_.Init(cp, use_lua_sampler, lua_sampler_script);
  if (debug) {
    cp.PrintDebugInfo();
  }
}

// Example feature must be pre-allocated
bool Assembler::GetExample(const std::string& input, bool is_predict,
                           Example* example) {
  return transform_.Transform(input, is_predict, example);
}

// user, item and context feature scheme
xJson::Value Assembler::GetFeatureScheme() const {
  xJson::Value obj;
  xJson::Value& feature_columns = obj["feature_columns"];
  feature_columns = xJson::arrayValue;

  for (const auto& c : transform_.columns()) {
    xJson::Value obj;
    c.first->Serialize(&obj);
    feature_columns.append(obj);
  }
  return obj;
}

void Assembler::InitNegativeSampling(const std::string& nce_items_path,
                                     int nce_items_min_count,
                                     int nce_items_top_k, int seed) {
  LOG(INFO) << "Init negative sampling ...";
  rng_.seed(seed);
  FileReader file_reader(ctx_->env(), nce_items_path);
  if (!file_reader.Init()) {
    LOG(ERROR) << "Open nce items file '" << nce_items_path << "' failed.";
    throw std::logic_error("Open nce items file failed");
  }
  std::string line;
  int lineindex = 0;
  std::vector<StringPiece> tokens;
  std::vector<int64_t> counts;
  int dirtycnt = 0;
  int dim = feature_size();
  Example example(Feature(dim), 0.0, 0.0);
  while (file_reader.ReadLine(&line)) {
    if (nce_items_top_k >= 0 &&
        static_cast<int>(nce_items_.size()) >= nce_items_top_k) {
      break;
    }
    ++lineindex;
    if (line.empty()) {
      continue;
    }
    tokens.clear();
    utils::Split(line, ";", &tokens);
    if (tokens.size() != 3) {
      ++dirtycnt;
      continue;
    }
    if (tokens[0].empty() || tokens[1].empty() || tokens[2].empty()) {
      ++dirtycnt;
      continue;
    }
    if (!GetExample(tokens[1].ToString(), true, &example)) {
      ++dirtycnt;
      continue;
    }
    int64_t cnt;
    try {
      cnt = std::stoll(tokens[2].ToString());
    } catch (const std::exception& e) {
      ++dirtycnt;
      continue;
    }
    if (cnt < nce_items_min_count) {
      continue;
    }
    counts.push_back(cnt);
    nce_features_.push_back(example.feature);
    nce_items_.push_back(tokens[0].ToString());
  }
  LOG(INFO) << "nce items size = " << nce_items_.size();
  for (int i = 0; i < std::min<int>(nce_items_.size(), 5); ++i) {
    LOG(INFO) << "nce item: " << nce_items_[i];
  }
  if (dirtycnt > lineindex * 0.05) {
    LOG(ERROR) << "Too many dirty lines, >5%";
    throw std::logic_error("Too many dirty lines, >5%");
  }
  InitTableNegatives(counts);
  LOG(INFO) << "Init negative sampling done";
}

void Assembler::InitTableNegatives(const std::vector<int64_t>& counts) {
  double z = 0.0;
  for (size_t i = 0; i < counts.size(); i++) {
    z += pow(counts[i], 0.5);
  }
  for (size_t i = 0; i < counts.size(); i++) {
    double c = pow(counts[i], 0.5);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
      negatives_.push_back(i);
    }
  }
  std::shuffle(negatives_.begin(), negatives_.end(), rng_);
}

int Assembler::GetNegative(int target) {
  int negative;
  do {
    negative = negatives_[negpos_];
    negpos_ = (negpos_ + 1) % negatives_.size();
  } while (target == negative);
  return negative;
}

// 负采样
bool Assembler::GetExamplesWithNegativeSampling(
    const std::string& input, std::vector<Example>* examples) {
  int negcnt = examples->size() - 1;
  for (int i = 0; i < negcnt; ++i) {
    int neg = GetNegative(-1);  // TODO(zhezhaoxu) use real target
    (*examples)[i + 1].feature = nce_features_[neg];
    (*examples)[i + 1].label = 0;
    (*examples)[i + 1].weight = 1.0;
  }
  return transform_.TransformWithNegativeSampling(input, examples);
}

void Assembler::GetServingInputs(const std::string& user_feature,
                                 const std::vector<std::string>& item_features,
                                 bool is_recall,
                                 std::vector<Feature>* features) {
  transform_.ServingTransform(user_feature, item_features, is_recall, features);
}

void Assembler::Serialize(std::string* serialized) const {
  xJson::Value s;
  transform_.Serialize(&s);
  *serialized = s.toStyledString();
}

bool Assembler::ParseFromString(const std::string& input) {
  std::istringstream iss(input);
  xJson::Value root;
  xJson::CharReaderBuilder rbuilder;
  std::string err;
  bool parse_ok = xJson::parseFromStream(rbuilder, iss, &root, &err);
  if (!parse_ok) {
    LOG(ERROR) << err;
    return false;
  }
  try {
    transform_.ReConstruct(root);
  } catch (const std::exception& e) {
    LOG(ERROR) << "parse from string catch exception: " << e.what();
    return false;
  }
  return true;
}

void Assembler::PrintDebugInfo() const {
  LOG(INFO) << " ---- Debug, Print Assembler info ----";
  LOG(INFO) << "feature_columns size = " << transform_.columns().size();
  for (auto c : transform_.columns()) {
    LOG(INFO) << "###### column info";
    c.first->PrintDebugInfo();
    c.second.PrintDebugInfo();
    LOG(INFO) << "######";
  }
}

void Assembler::PrintExample(const Example& e) const {
  LOG(INFO) << "------- Example -------";
  LOG(INFO) << "label = " << e.label;
  LOG(INFO) << "weight = " << e.weight;
  size_t idx = 0;
  for (auto c : transform_.columns()) {
    std::ostringstream oss;
    for (int i = 0; i < c.first->width(); ++i) {
      oss << e.feature[idx] << " ";
      ++idx;
    }
    LOG(INFO) << "iname = " << c.first->iname()
              << ", alias = " << c.first->alias()
              << ", type = " << Field::TYPE_STR(c.first->type())
              << ", cid = " << Field::CID_STR(c.first->cid())
              << ", width = " << c.first->width()
              << ", feature = " << oss.str();
  }
  LOG(INFO) << "----------------------";
}

}  // namespace assembler
