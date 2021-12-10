#include "assembler/column.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>

#include "assembler/jsoncpp_helper.hpp"
#include "assembler/triple_list_filter.h"
#include "assembler/utils.h"
#include "deps/jsoncpp/json/json.h"

namespace assembler {

using ::utils::jsoncpp_helper::checkJsonArgs;
using ::utils::jsoncpp_helper::string_tag;
using ::utils::jsoncpp_helper::int32_tag;
using ::utils::jsoncpp_helper::double_tag;
using ::utils::jsoncpp_helper::bool_tag;
using ::utils::jsoncpp_helper::array_tag;
using ::utils::jsoncpp_helper::object_tag;

void BasicColumn::Serialize(xJson::Value* bc) const {
  (*bc)["iname"] = iname();
  (*bc)["alias"] = alias();
  (*bc)["type"] = type();
  (*bc)["type_str"] = Field::TYPE_STR(type());
  (*bc)["cid"] = cid();
  (*bc)["cid_str"] = Field::CID_STR(cid());
  (*bc)["def"] = def();
  (*bc)["width"] = width();
}
void BasicColumn::ReConstruct(const xJson::Value& bc) {
  if (!checkJsonArgs(bc, "iname", string_tag, "type", int32_tag, "cid",
                     int32_tag, "def", double_tag, "width", int32_tag)) {
    LOG(ERROR) << " ReConstruct error";
    throw std::logic_error("ReConstruct error");
  }
  iname_ = bc["iname"].asString();

  if (checkJsonArgs(bc, "alias", string_tag)) {
    alias_ = bc["alias"].asString();
  } else {
    alias_ = iname_;
  }

  type_ = static_cast<Field::Type>(bc["type"].asInt());
  cid_ = static_cast<Field::CID>(bc["cid"].asInt());
  def_ = bc["def"].asDouble();
  width_ = bc["width"].asInt();
}

std::shared_ptr<BasicColumn> BasicColumn::Parse(
    ::tensorflow::OpKernelConstruction* ctx, const xJson::Value& bc) {
  if (!checkJsonArgs(bc, "type", int32_tag)) {
    throw std::logic_error("parse basic column error");
  }
  std::shared_ptr<BasicColumn> c;
  Field::Type type = static_cast<Field::Type>(bc["type"].asInt());
  if (type == Field::Type::INT) {
    c.reset(new NumericColumn(ctx));
  } else if (type == Field::Type::FLOAT) {
    c.reset(new NumericColumn(ctx));
  } else if (type == Field::Type::STRING) {
    c.reset(new StringColumn(ctx));
  } else if (type == Field::Type::STRING_LIST) {
    c.reset(new StringListColumn(ctx));
  } else if (type == Field::Type::WEIGHTED_STRING_LIST) {
    c.reset(new WeightedStringListColumn(ctx));
  } else if (type == Field::Type::TRIPLE_LIST) {
    c.reset(new TripleListColumn(ctx));
  } else {
    LOG(ERROR) << "Parse serialized error, unknow type.";
    throw std::logic_error("Parse serialized error, unknow type.");
  }
  c->ReConstruct(bc);
  return c;
}

void BasicColumn::PrintDebugInfo() const {
  LOG(INFO) << "iname = '" << iname() << "'";
  LOG(INFO) << "alias = '" << alias() << "'";
  LOG(INFO) << "type = " << Field::TYPE_STR(type());
  LOG(INFO) << "cid = " << Field::CID_STR(cid());
  LOG(INFO) << "default = " << def();
  LOG(INFO) << "width = " << width();
}

void NumericColumn::ToValue(const std::string& key,
                            std::vector<double>::iterator first,
                            bool* use_def) const {
  *use_def = false;
  double v = def();
  if (key == "") {
    *use_def = true;
    (*first) = v;
    return;
  }
  try {
    v = std::stof(key);
    if (std::isnan(v) || std::isinf(v)) {
      v = def();
      *use_def = true;
    }
  } catch (const std::exception& e) {
    *use_def = true;
  }
  (*first) = v;
}

void NumericColumn::Serialize(xJson::Value* bc) const {
  BasicColumn::Serialize(bc);
}

void StringBaseColumn::Serialize(xJson::Value* bc) const {
  BasicColumn::Serialize(bc);
  (*bc)["min_count"] = min_count_;
  (*bc)["top_k"] = top_k_;
  (*bc)["oov_buckets"] = oov_buckets_;
  (*bc)["default_key"] = default_key_;
  (*bc)["scan_from"] = scan_from_;
  (*bc)["reverse"] = reverse_;
  xJson::Value& keys = (*bc)["keys"];
  keys = xJson::arrayValue;
  for (size_t i = 0; i < keys_.size(); ++i) {
    keys.append(keys_[i]);
  }
  (*bc)["size"] = keys_.size() + oov_buckets_;
}

void StringBaseColumn::PrintDebugInfo() const {
  BasicColumn::PrintDebugInfo();
  LOG(INFO) << "min_count = " << min_count_;
  LOG(INFO) << "top_k = " << top_k_;
  LOG(INFO) << "default key = " << default_key_;
  LOG(INFO) << "scan_from = " << scan_from_;
  LOG(INFO) << "reverse = " << reverse_;
  LOG(INFO) << "oov size = " << oov_buckets_;
  LOG(INFO) << "dict size = " << keys_.size();
  for (size_t i = 0; i < std::min<size_t>(10, keys_.size()); ++i) {
    LOG(INFO) << "#" << i << ": " << keys_[i];
  }
}

void StringBaseColumn::ReConstruct(const xJson::Value& bc) {
  BasicColumn::ReConstruct(bc);

  if (!checkJsonArgs(bc, "min_count", int32_tag, "top_k", int32_tag,
                     "oov_buckets", int32_tag, "keys", array_tag)) {
    throw std::logic_error("ReConstruct error");
  }
  keys_.clear();
  indexer_.clear();
  min_count_ = bc["min_count"].asInt();
  top_k_ = bc["top_k"].asInt();
  oov_buckets_ = bc["oov_buckets"].asInt();

  // 兼容模式
  if (!checkJsonArgs(bc, "default_key", string_tag)) {
    LOG(INFO) << "ReConstruct no field 'default_key', use default value ''";
    default_key_ = "";
  } else {
    default_key_ = bc["default_key"].asString();
  }
  if (!checkJsonArgs(bc, "scan_from", int32_tag)) {
    LOG(INFO) << "ReConstruct no field 'scan_from', use default value 'head'";
    scan_from_ = SCAN_TYPE::HEAD;
  } else {
    scan_from_ = static_cast<SCAN_TYPE>(bc["scan_from"].asInt());
  }
  if (!checkJsonArgs(bc, "reverse", bool_tag)) {
    LOG(INFO) << "ReConstruct no field 'reverse', use default value 'false'";
    reverse_ = false;
  } else {
    reverse_ = bc["reverse"].asBool();
  }

  const xJson::Value& keys = bc["keys"];
  for (xJson::ArrayIndex i = 0; i < keys.size(); ++i) {
    if (!keys[i].isString()) {
      throw std::logic_error("ReConstruct error");
    }
    std::string key = keys[i].asString();
    keys_.push_back(key);
    indexer_[key] = i;
  }
}

void StringColumn::ToValue(const std::string& key,
                           std::vector<double>::iterator first,
                           bool* use_def) const {
  *use_def = false;
  if (key.empty()) {
    // 特征缺失情况，返回默认值
    (*first) = def();
    return;
  }
  double v = -1.0;  // 特征不在词典返回 -1
  auto it = indexer_.find(key);
  if (it != indexer_.end()) {
    v = static_cast<double>(it->second);
  } else {
    if (oov_buckets_ > 0) {
      v = utils::MurMurHash3(key) % oov_buckets_ + indexer_.size();
    } else {
      *use_def = true;
    }
  }
  (*first) = v;
}

// 优先取在词典中的值，不在词典的值放在最后
void StringListColumn::ToValue(const std::string& key,
                               std::vector<double>::iterator first,
                               bool* use_def) const {
  *use_def = true;  // 当list全部为默认值时才为true
  // 初始都填 -1, def() 值可能不是 -1
  std::fill(first, first + width(), -1.0);
  if (key.empty()) {
    // 为空时特殊处理，默认1个元素
    *first = def();
    return;
  }
  auto tokens = utils::Split(key, ",");
  if (scan_from_ == SCAN_TYPE::TAIL) {
    std::reverse(tokens.begin(), tokens.end());
  }
  int cnt = 0;
  int hashcnt = 0;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (cnt >= width()) {
      break;
    }
    if (tokens[i].empty()) {
      continue;
    }
    std::string token = tokens[i].ToString();
    auto it = indexer_.find(token);
    if (it != indexer_.end()) {
      *(first + cnt) = it->second;
      ++cnt;
      *use_def = false;
    } else {
      if (oov_buckets_ > 0) {
        ++hashcnt;
        double v = utils::MurMurHash3(token) % oov_buckets_ + indexer_.size();
        *(first + cnt) = v;
        ++cnt;
        *use_def = false;
      }
    }
  }
  if (reverse_) {
    std::reverse(first, first + cnt);
  }
}

void FloatListColumn::ToValue(const std::string& key,
                              std::vector<double>::iterator first,
                              bool* use_def) const {
  *use_def = true;  // 当list全部为默认值时才为true
  std::fill(first, first + width(), def());
  auto tokens = utils::Split(key, ",");
  if (scan_from_ == SCAN_TYPE::TAIL) {
    std::reverse(tokens.begin(), tokens.end());
  }
  int cnt = 0;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (cnt >= width()) {
      break;
    }
    if (tokens[i] == "") {
      continue;
    }
    try {
      *(first + cnt) = std::stof(tokens[i].ToString());
      if (std::isnan(*(first + i)) || std::isinf(*(first + i))) {
        *(first + cnt) = def();
        ++cnt;
        continue;
      }
      ++cnt;
      *use_def = false;
    } catch (const std::exception& e) {
    }
  }
  if (reverse_) {
    std::reverse(first, first + cnt);
  }
}

void FloatListColumn::Serialize(xJson::Value* bc) const {
  BasicColumn::Serialize(bc);
}

// 优先取在词典中的值，不在词典的值放在最后
void WeightedStringListColumn::ToValue(const std::string& key,
                                       std::vector<double>::iterator first,
                                       bool* use_def) const {
  *use_def = true;  // 当list全部为默认值时才为true
  std::fill(first, first + width(), -1.0);
  // 权重必须默认为 0，当 -1 时，weight 必须为 0
  for (int i = width() / 2; i < width(); i++) {
    *(first + i) = 0.0;
  }
  if (key.empty()) {
    // 为空时特殊处理，默认1个元素
    *first = def();
    *(first + width() / 2) = (def() == -1 ? 0.0 : 1.0);
    return;
  }
  auto tokens = utils::Split(key, ",");
  if (scan_from_ == SCAN_TYPE::TAIL) {
    std::reverse(tokens.begin(), tokens.end());
  }
  std::vector<StringPiece> subtokens;
  std::string token;
  int cnt = 0;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (cnt >= width() / 2) {
      break;
    }
    subtokens.clear();
    utils::Split(tokens[i], ":", &subtokens);
    if (subtokens.size() != 2) {
      continue;
    }
    double weight = 0.0;
    try {
      weight = std::stof(subtokens[1].ToString());
    } catch (const std::exception& e) {
      continue;
    }
    if (weight < min_weight_) {
      continue;
    }
    if (subtokens[0].empty()) {
      continue;
    }

    if (subtokens[0].empty()) {
      continue;
    }
    token = subtokens[0].ToString();
    auto it = indexer_.find(token);
    if (it != indexer_.end()) {
      *(first + cnt) = it->second;
      *(first + cnt + width() / 2) = weight;
      ++cnt;
      *use_def = false;
    } else {
      if (oov_buckets_ > 0) {
        double v = utils::MurMurHash3(token) % oov_buckets_ + indexer_.size();
        *(first + cnt) = v;
        *(first + cnt + width() / 2) = weight;
        ++cnt;
        *use_def = false;
      }
    }
  }
  if (reverse_) {
    std::reverse(first, first + cnt);
    std::reverse(first + width() / 2, first + width() / 2 + cnt);
  }
}

void WeightedStringListColumn::Serialize(xJson::Value* bc) const {
  StringBaseColumn::Serialize(bc);
  (*bc)["min_weight"] = min_weight_;
}

void WeightedStringListColumn::ReConstruct(const xJson::Value& bc) {
  StringBaseColumn::ReConstruct(bc);
  if (!checkJsonArgs(bc, "min_weight", double_tag)) {
    throw std::logic_error("ReConstruct error");
  }
  min_weight_ = bc["min_weight"].asDouble();
}

void WeightedStringListColumn::PrintDebugInfo() const {
  StringBaseColumn::PrintDebugInfo();
  LOG(INFO) << "min_weight = " << min_weight_;
}

void TripleListColumn::ToValue(const std::string& key,
                               std::vector<double>::iterator first,
                               bool* use_def) const {
  *use_def = true;  // 当list全部为默认值时才为true
  std::fill(first, first + width(), -1.0);
  // 权重必须默认为 0，当 -1 时，weight 必须为 0
  for (int i = width() / 2; i < width(); i++) {
    *(first + i) = 0.0;
  }
  if (key.empty()) {
    // 为空时特殊处理，默认1个元素
    *first = def();
    *(first + width() / 2) = (def() == -1 ? 0.0 : 1.0);

    *(first + width_pos()) = def();
    *(first + width() / 2 + width_pos()) = (def() == -1 ? 0.0 : 1.0);
    return;
  }
  auto tokens = utils::Split(key, ",");
  if (scan_from_ == SCAN_TYPE::TAIL) {
    std::reverse(tokens.begin(), tokens.end());
  }

  int pos_idx = 0;
  int neg_idx = width_pos();
  std::vector<StringPiece> subtokens;
  std::string rowkey;
  int poscnt = 0;
  int negcnt = 0;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (poscnt >= width_pos() && negcnt >= width_neg()) {
      break;
    }
    subtokens.clear();
    // id:xx:xx
    utils::Split(tokens[i], ":", &subtokens);
    if (subtokens.size() != 3) {
      continue;
    }
    rowkey = subtokens[0].ToString();
    if (rowkey.empty()) {
      continue;
    }
    // get rowkey index
    auto it = indexer_.find(rowkey);
    double id = def();
    bool valid = false;
    if (it != indexer_.end()) {
      id = it->second;
      valid = true;
    } else {
      if (oov_buckets_ > 0) {
        id = utils::MurMurHash3(rowkey) % oov_buckets_ + indexer_.size();
        valid = true;
      }
    }
    if (!valid) {
      continue;
    }
    std::pair<int, double> p =
        filter_func_(subtokens[1], subtokens[2], params_);
    if (p.first == 1) {
      if (poscnt >= width_pos()) {
        continue;
      }
      ++poscnt;
      *(first + pos_idx) = id;
      *(first + width() / 2 + pos_idx) = p.second;
      ++pos_idx;
    } else if (p.first == 0) {
      if (negcnt >= width_neg()) {
        continue;
      }
      ++negcnt;
      *(first + neg_idx) = id;
      *(first + width() / 2 + neg_idx) = p.second;
      ++neg_idx;
    } else {
      continue;
    }
  }
  if (reverse_) {
    std::reverse(first, first + poscnt);
    std::reverse(first + width() / 2, first + width() / 2 + poscnt);

    std::reverse(first + width_pos(), first + width_pos() + negcnt);
    std::reverse(first + width() / 2 + width_pos(),
                 first + width() / 2 + width_pos() + negcnt);
  }
}

void TripleListColumn::Serialize(xJson::Value* bc) const {
  StringBaseColumn::Serialize(bc);
  (*bc)["width_pos"] = width_pos_;
  (*bc)["width_neg"] = width_neg_;
  (*bc)["filter_type"] = filter_type_;
  (*bc)["params"] = params_;
}

void TripleListColumn::PrintDebugInfo() const {
  StringBaseColumn::PrintDebugInfo();
  LOG(INFO) << "width_pos = " << width_pos_;
  LOG(INFO) << "width_neg = " << width_neg_;
  LOG(INFO) << "filter_type = " << filter_type_;
}

void TripleListColumn::ReConstruct(const xJson::Value& bc) {
  StringBaseColumn::ReConstruct(bc);
  if (!checkJsonArgs(bc, "width_pos", int32_tag, "width_neg", int32_tag,
                     "filter_type", string_tag, "params", object_tag)) {
    throw std::logic_error("ReConstruct error");
  }
  width_pos_ = bc["width_pos"].asInt();
  width_neg_ = bc["width_neg"].asInt();
  filter_type_ = bc["filter_type"].asString();
  params_ = bc["params"];

  if (global_filter_map.count(filter_type_) == 0) {
    LOG(ERROR) << "Undefined filter_type '" << filter_type_ << "', "
               << "global_filter_map.size = " << global_filter_map.size();
    throw std::logic_error("undefined filter_type");
  }
  filter_func_ = global_filter_map[filter_type_];
}

}  // namespace assembler
