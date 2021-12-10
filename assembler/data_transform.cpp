#include "assembler/data_transform.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#include "tensorflow/core/platform/file_system.h"

#include "assembler/file_reader.h"
#include "assembler/jsoncpp_helper.hpp"
#include "assembler/meta_manager.h"
#include "assembler/monitor_feature.h"
#include "assembler/utils.h"

namespace assembler {

using ::utils::jsoncpp_helper::checkJsonArgs;
using ::utils::jsoncpp_helper::int32_tag;
using ::utils::jsoncpp_helper::array_tag;
using ::utils::jsoncpp_helper::object_tag;

static const char* delim = "\t;";

DataTransform::DataTransform(::tensorflow::OpKernelConstruction* ctx)
    : ctx_(ctx) {}

void DataTransform::Init(const ConfParser& cp, bool use_lua_sampler,
                         const std::string& lua_sampler_script) {
  use_lua_sampler_ = use_lua_sampler;
  lua_sampler_script_ = lua_sampler_script;

  MetaManager meta(ctx_->env());
  meta.Init(cp.data_paths().meta_file);
  feature_def_.clear();

  // lua sampler TODO(zhezhaoxu) 优化代码
  if (use_lua_sampler) {
    if (!lua_sampler_.Init(lua_sampler_script)) {
      LOG(ERROR) << "Init lua_sampler failed, script = '" << lua_sampler_script
                 << "'";
      throw std::logic_error("Init lua_sampler failed.");
    }
    int idx = 0;
    for (const auto& f : lua_sampler_.features()) {
      auto it = meta.name2id().find(f);
      if (it == meta.name2id().end()) {
        if (f == "label") {
          auto p = lua_params_map_.insert({"1", idx});
          ++idx;
          if (!p.second) {
            LOG(ERROR) << "lua sampler script feature '" << f << "' duplicated";
            throw std::logic_error("lua sampler script 'features' duplicated");
          }
        } else if (f == "weight") {
          auto p = lua_params_map_.insert({"2", idx});
          ++idx;
          if (!p.second) {
            LOG(ERROR) << "lua sampler script feature '" << f << "' duplicated";
            throw std::logic_error("lua sampler script 'features' duplicated");
          }
        } else {
          LOG(ERROR) << "lua sampler script feature '" << f << "' not in meta";
          throw std::logic_error("lua sampler script 'features' not in meta");
        }
      } else {
        auto p = lua_params_map_.insert({it->second, idx});
        ++idx;
        if (!p.second) {
          LOG(ERROR) << "lua sampler script feature '" << f << "' duplicated";
          throw std::logic_error("lua sampler script 'features' duplicated");
        }
      }
    }
  }

  for (auto& c : cp.columns()) {
    if (meta.name2id().count(c->iname()) == 0) {
      LOG(ERROR) << "column '" << c->iname() << "' not in meta.";
      throw std::logic_error("column not in meta");
    }
    std::string id = meta.name2id().at(c->iname());
    Field field = meta.meta().at(id);
    // TODO(zhezhaoxu) cid 不应该在这里设置
    c->set_cid(field.cid);
    if (field.type != c->type()) {
      LOG(ERROR) << "column '" << c->iname() << "' type not matched with meta, "
                 << "meta type = " << Field::TYPE_STR(field.type)
                 << ", conf type = " << Field::CID(c->type());
      throw std::logic_error("column type not matched with meta");
    }
    column_map_.insert({id, FeatureInfo(c, feature_size_)});
    columns_.push_back({c, field});
    feature_size_ += c->width();
    bool b = false;
    std::vector<double> v(c->width());
    c->ToValue("", v.begin(), &b);
    feature_def_.insert(feature_def_.end(), v.begin(), v.end());
  }
  if (feature_size_ != feature_def_.size()) {
    LOG(ERROR) << "Code bug, please contact zhezhaoxu.";
    throw std::logic_error("code bug");
  }
}

bool DataTransform::FillFeature(const std::vector<StringPiece>& tokens,
                                bool need_label, Feature* feature,
                                double* label, double* weight) {
  if (weight) {
    *weight = 1.0;  // default weight
  }

  std::string id;
  std::string value;
  bool good = true;
  bool has_label = false;
  bool has_weight = false;

  std::vector<std::string> lua_params(lua_params_map_.size());

  for (size_t i = 0; i < tokens.size(); i++) {
    if (tokens[i].empty()) {
      continue;
    }
    size_t x = tokens[i].find('|');
    if (x == StringPiece::npos) {
      LOG(ERROR) << "train data error: token no sep '|', token = " << tokens[i];
      good = false;
      continue;
    }
    id = tokens[i].substr(0, x).ToString();
    if (id == "") {
      LOG(ERROR) << "train data error: id empty";
      good = false;
      continue;
    }
    value = tokens[i].substr(x + 1).ToString();
    if (value == "") {
      continue;
    }

    if (use_lua_sampler_ && need_label) {
      auto it = lua_params_map_.find(id);
      if (it != lua_params_map_.end()) {
        lua_params[it->second] = value;
      }
    }

    if (label && id == "1") {
      if (has_label) {
        LOG(ERROR) << "train data error: label duplicated";
      }
      try {
        *label = std::stof(value);
      } catch (const std::exception& e) {
        LOG(FATAL) << "parse label error, input token = " << tokens[i];
      }
      has_label = true;
      continue;
    } else if (weight && id == "2") {
      if (has_weight) {
        LOG(ERROR) << "train data error: weight duplicated";
      }
      try {
        *weight = std::stof(value);
      } catch (const std::exception& e) {
        LOG(FATAL) << "parse weight error, input token = " << tokens[i];
      }
      has_weight = true;
      continue;
    }

    // feature id
    auto pr = column_map_.equal_range(id);
    if (pr.first == column_map_.end()) {
      continue;
    }
    for (auto it = pr.first; it != pr.second; ++it) {
      int pos = it->second.pos;
      const std::shared_ptr<BasicColumn>& bc = it->second.bc;
      bool use_def = false;
      bc->ToValue(value, feature->begin() + pos, &use_def);
      pos += bc->width();
    }
  }

  if (use_lua_sampler_ && need_label) {
    auto p = lua_sampler_.get_label_and_weight(lua_params);
    if (label) {
      *label = p.first;
    }
    if (weight) {
      *weight = p.second;
    }
  }

  if (need_label && !has_label) {
    LOG(ERROR) << "train data error: no label";
    good = false;
  }

  return good;
}

// @examples: first element is real example, others are negative sampling
// samples
// 只用在训练中
bool DataTransform::FillFeatureWithNegativeSampling(
    const std::vector<StringPiece>& tokens, std::vector<Example>* examples) {
  (*examples)[0].weight = 1.0;
  std::string id;
  bool has_label = false;
  bool has_weight = false;
  std::string value;
  bool good = true;
  std::vector<std::string> lua_params(lua_params_map_.size());
  for (size_t i = 0; i < tokens.size(); i++) {
    if (tokens[i].empty()) {
      continue;
    }
    size_t x = tokens[i].find('|');
    if (x == StringPiece::npos) {
      LOG(ERROR) << "train data error: token no sep '|'";
      good = false;
      continue;
    }
    id = tokens[i].substr(0, x).ToString();
    if (id == "") {
      LOG(ERROR) << "train data error: id empty";
      good = false;
      continue;
    }
    value = tokens[i].substr(x + 1).ToString();
    if (value == "") {
      continue;
    }

    if (use_lua_sampler_) {
      auto it = lua_params_map_.find(id);
      if (it != lua_params_map_.end()) {
        lua_params[it->second] = value;
      }
    }

    if (id == "1") {
      if (has_label) {
        LOG(ERROR) << "train data error: label duplicated";
      }
      try {
        (*examples)[0].label = std::stof(value);
      } catch (const std::exception& e) {
        LOG(FATAL) << "parse label error, input token = " << tokens[i];
      }
      has_label = true;
      continue;
    } else if (id == "2") {
      if (has_weight) {
        LOG(ERROR) << "train data error: weight duplicated";
      }
      try {
        (*examples)[0].weight = std::stof(value);
      } catch (const std::exception& e) {
        LOG(FATAL) << "parse weight error, input token = " << tokens[i];
      }
      has_weight = true;
      continue;
    }

    // feature id
    auto pr = column_map_.equal_range(id);
    if (pr.first == column_map_.end()) {
      continue;
    }
    for (auto it = pr.first; it != pr.second; ++it) {
      int pos = it->second.pos;
      const std::shared_ptr<BasicColumn>& bc = it->second.bc;
      bool use_def = false;
      bc->ToValue(value, (*examples)[0].feature.begin() + pos, &use_def);
      if (bc->cid() != Field::CID::ITEM) {
        // copy to negative sampling samples
        for (size_t i = 1; i < examples->size(); i++) {
          std::copy((*examples)[0].feature.begin() + pos,
                    (*examples)[0].feature.begin() + pos + bc->width(),
                    (*examples)[i].feature.begin() + pos);
        }
      }
      pos += bc->width();
      if (use_def) {
        // TODO(zhezhaoxu) add monitor, train and serving
      }
    }
  }

  if (use_lua_sampler_) {
    auto p = lua_sampler_.get_label_and_weight(lua_params);
    (*examples)[0].label = p.first;
    (*examples)[0].weight = p.second;
  }

  if (!has_label) {
    LOG(ERROR) << "train data error: no label";
    good = false;
  }

  return good;
}

// training, exit when error
bool DataTransform::Transform(const std::string& input, bool is_predict,
                              Example* example) {
  std::copy(feature_def_.begin(), feature_def_.end(), example->feature.begin());
  example->label = 0.0;
  example->weight = 0.0;
  std::vector<StringPiece> tokens;
  utils::Split(input, delim, &tokens);
  if (!FillFeature(tokens, !is_predict, &example->feature, &example->label,
                   &example->weight)) {
    // TODO(zhezhaoxu) 出错不 core，兼容目前脏数据情况
    // throw std::logic_error("train data error");
    return false;
  }
  return true;
}

bool DataTransform::TransformWithNegativeSampling(
    const std::string& input, std::vector<Example>* examples) {
  std::vector<StringPiece> tokens;
  utils::Split(input, delim, &tokens);
  if (!FillFeatureWithNegativeSampling(tokens, examples)) {
    // TODO(zhezhaoxu) 出错不 core，兼容目前脏数据情况
    // throw std::logic_error("train data error");
    return false;
  }
  return true;
}

// serving, use default value when error
void DataTransform::ServingTransform(
    const std::string& user_feature,
    const std::vector<std::string>& item_features, bool is_recall,
    std::vector<Feature>* features) {
  std::vector<StringPiece> user_tokens;
  utils::Split(user_feature, delim, &user_tokens);
  std::vector<StringPiece> other_tokens;  // item and ctx features
  Feature feature_user = feature_def_;
  FillFeature(user_tokens, false, &feature_user, nullptr, nullptr);
  if (is_recall) {
    features->push_back(feature_user);
    return;
  }
  Feature feature;
  for (size_t i = 0; i < item_features.size(); ++i) {
    other_tokens.clear();
    utils::Split(item_features[i], delim, &other_tokens);
    feature = feature_user;  // copy user features
    FillFeature(other_tokens, false, &feature, nullptr, nullptr);
    features->push_back(feature);
  }
}

void DataTransform::Serialize(xJson::Value* s) const {
  (*s)["feature_size"] = feature_size_;
  xJson::Value& json_feature_def = (*s)["feature_def"];
  for (auto x : feature_def_) {
    json_feature_def.append(x);
  }
  xJson::Value& cols = (*s)["columns"];
  cols = xJson::arrayValue;
  for (const auto& c : columns_) {
    xJson::Value bc;
    xJson::Value f;
    c.first->Serialize(&bc);  // basic column
    c.second.Serialize(&f);   // field
    xJson::Value col;
    col["basic_column"] = bc;
    col["field"] = f;
    cols.append(col);
  }
}

void DataTransform::ReConstruct(const xJson::Value& root) {
  columns_.clear();
  column_map_.clear();
  feature_def_.clear();
  if (!checkJsonArgs(root, "columns", array_tag, "feature_size", int32_tag,
                     "feature_def", array_tag)) {
    LOG(ERROR) << "DataTransform::ReConstruct: member 'columns', "
                  "'feature_size' or 'feature_def' error";
    throw std::logic_error("ReConstruct error");
  }
  feature_size_ = root["feature_size"].asInt();
  const xJson::Value& json_feature_def = root["feature_def"];
  for (xJson::ArrayIndex i = 0; i < json_feature_def.size(); i++) {
    if (!json_feature_def[i].isDouble()) {
      LOG(ERROR) << "Element type of 'feature_def' is not double";
      throw std::logic_error("ReConstruct error");
    }
    feature_def_.push_back(json_feature_def[i].asDouble());
  }

  const xJson::Value& cols = root["columns"];
  int pos = 0;
  for (xJson::ArrayIndex i = 0; i < cols.size(); ++i) {
    const xJson::Value& col = cols[i];
    if (!checkJsonArgs(col, "basic_column", object_tag, "field", object_tag)) {
      LOG(ERROR) << "DataTransform::ReConstruct error, member 'basic_column' "
                    "or 'field' not exist or not object";
      throw std::logic_error("ReConstruct error");
    }
    std::shared_ptr<BasicColumn> bc =
        BasicColumn::Parse(ctx_, col["basic_column"]);
    Field field;
    field.Parse(col["field"]);
    columns_.push_back({bc, field});
    column_map_.insert({field.id, FeatureInfo(bc, pos)});
    pos += bc->width();
  }
  if (pos != static_cast<int>(feature_size_)) {
    LOG(ERROR) << "ReConstruct error, feature_size not matched, pos = " << pos
               << ", feature_size = " << feature_size_;
    throw std::logic_error("ReConstruct error");
  }
}

}  // namespace assembler
