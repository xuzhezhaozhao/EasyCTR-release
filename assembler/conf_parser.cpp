#include "assembler/conf_parser.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "deps/jsoncpp/json/json.h"

#include "assembler/column.h"
#include "assembler/file_reader.h"
#include "assembler/jsoncpp_helper.hpp"
#include "assembler/meta_manager.h"

namespace assembler {

using ::utils::jsoncpp_helper::checkJsonArgs;
using ::utils::jsoncpp_helper::string_tag;
using ::utils::jsoncpp_helper::int32_tag;
using ::utils::jsoncpp_helper::double_tag;
using ::utils::jsoncpp_helper::bool_tag;
using ::utils::jsoncpp_helper::array_tag;
using ::utils::jsoncpp_helper::object_tag;

bool ConfParser::Parse(const std::string& conf_path, bool load_dict) {
  xJson::Value root;
  xJson::CharReaderBuilder rbuilder;
  rbuilder["collectComments"] = false;
  FileReader file_reader(ctx_->env(), conf_path);
  if (!file_reader.Init()) {
    LOG(ERROR) << "Open file " << conf_path << " failed.";
    return false;
  }

  std::string conf;
  if (!file_reader.ReadAll(&conf)) {
    LOG(ERROR) << "Read file " << conf_path << " failed.";
    return false;
  }

  std::istringstream iss(conf);
  std::string err;
  bool parse_ok = xJson::parseFromStream(rbuilder, iss, &root, &err);
  if (!parse_ok) {
    LOG(ERROR) << err;
    return false;
  }

  if (!checkJsonArgs(root, "input_data", object_tag)) {
    LOG(ERROR) << "Conf parse error";
  }
  const xJson::Value& data = root["input_data"];
  if (!checkJsonArgs(data, "meta_file", string_tag)) {
    LOG(ERROR) << "Conf parse error";
    return false;
  }
  data_paths_.meta_file = data["meta_file"].asString();

  if (!checkJsonArgs(root, "assembler", array_tag)) {
    LOG(ERROR) << "Conf parse error";
  }

  for (xJson::ArrayIndex idx = 0; idx < root["assembler"].size(); ++idx) {
    if (!root["assembler"][idx].isObject()) {
      LOG(ERROR) << "column is not object.";
      return false;
    }
    xJson::Value& json_column = root["assembler"][idx];
    if (!checkJsonArgs(json_column, "iname", string_tag, "type", string_tag)) {
      LOG(ERROR) << "Conf parse error, column idx = " << idx;
      return false;
    }
    // 兼容老版本
    if (!json_column.isMember("alias")) {
      json_column["alias"] = json_column["iname"].asString();
    }
    if (!checkJsonArgs(json_column, "alias", string_tag)) {
      LOG(ERROR) << "Conf parse error, column idx = " << idx;
      return false;
    }

    std::shared_ptr<BasicColumn> column;
    bool create_column_ok = true;
    const std::string& ct = json_column["type"].asString();
    if (ct == "int") {
      create_column_ok =
          CreateNumericColumn(json_column, Field::Type::INT, &column);
    } else if (ct == "float") {
      create_column_ok =
          CreateNumericColumn(json_column, Field::Type::FLOAT, &column);
    } else if (ct == "string") {
      create_column_ok = CreateStringColumn(json_column, &column, load_dict);
    } else if (ct == "string_list") {
      create_column_ok =
          CreateStringListColumn(json_column, &column, load_dict);
    } else if (ct == "float_list") {
      create_column_ok = CreateFloatListColumn(json_column, &column);
    } else if (ct == "weighted_string_list") {
      create_column_ok =
          CreateWeightedStringListColumn(json_column, &column, load_dict);
    } else if (ct == "triple_list") {
      create_column_ok =
          CreateTripleListColumn(json_column, &column, load_dict);
    } else {
      LOG(ERROR) << "Unknown column::type '" << ct << "'";
      return false;
    }
    if (!create_column_ok) {
      return false;
    }
    columns_.push_back(column);
  }

  return true;
}

bool ConfParser::CreateNumericColumn(const xJson::Value& obj, Field::Type type,
                                     std::shared_ptr<BasicColumn>* column) {
  const std::string& iname = obj["iname"].asString();
  const std::string& alias = obj["alias"].asString();
  if (!checkJsonArgs(obj, "default", double_tag)) {
    LOG(ERROR) << "Conf parse error";
    return false;
  }
  double def = obj["default"].asDouble();
  (*column).reset(
      new NumericColumn(ctx_, iname, alias, type, Field::CID::CID_END, def));
  return true;
}

bool ConfParser::CreateStringColumn(const xJson::Value& obj,
                                    std::shared_ptr<BasicColumn>* column,
                                    bool load_dict) {
  const std::string& iname = obj["iname"].asString();
  const std::string& alias = obj["alias"].asString();
  if (!checkJsonArgs(obj, "min_count", int32_tag, "top_k", int32_tag,
                     "oov_buckets", int32_tag, "default", string_tag,
                     "dict_file", string_tag)) {
    LOG(ERROR) << "Conf parse error";
    return false;
  }
  int min_count = obj["min_count"].asInt();
  int top_k = obj["top_k"].asInt();
  int oov_buckets = obj["oov_buckets"].asInt();
  const std::string& default_key = obj["default"].asString();
  StringColumn* sc =
      new StringColumn(ctx_, iname, alias, Field::CID::CID_END, -1, min_count,
                       top_k, oov_buckets, default_key);
  const std::string& dict_file = obj["dict_file"].asString();
  if (load_dict && dict_file != "" && !sc->LoadFromDictFile(dict_file)) {
    LOG(ERROR) << "Load dict file '" << dict_file << "' error.";
    return false;
  }
  sc->UpdateDefault();

  (*column).reset(sc);
  return true;
}

bool ConfParser::CreateStringListColumn(const xJson::Value& obj,
                                        std::shared_ptr<BasicColumn>* column,
                                        bool load_dict) {
  const std::string& iname = obj["iname"].asString();
  const std::string& alias = obj["alias"].asString();
  if (!checkJsonArgs(obj, "min_count", int32_tag, "top_k", int32_tag,
                     "oov_buckets", int32_tag, "default", string_tag,
                     "dict_file", string_tag, "width", int32_tag, "scan_from",
                     string_tag, "reverse", bool_tag)) {
    LOG(ERROR) << "Conf parse error";
    return false;
  }

  int min_count = obj["min_count"].asInt();
  int top_k = obj["top_k"].asInt();
  int oov_buckets = obj["oov_buckets"].asInt();
  const std::string& default_key = obj["default"].asString();
  int width = obj["width"].asInt();
  const std::string& sf = obj["scan_from"].asString();
  bool reverse = obj["reverse"].asBool();
  SCAN_TYPE scan_from;
  if (sf == "head") {
    scan_from = SCAN_TYPE::HEAD;
  } else if (sf == "tail") {
    scan_from = SCAN_TYPE::TAIL;
  } else {
    LOG(ERROR) << "Conf parse error: scan_from";
    return false;
  }

  StringListColumn* sc = new StringListColumn(
      ctx_, iname, alias, Field::CID::CID_END, -1, min_count, top_k, width,
      oov_buckets, default_key, scan_from, reverse);

  const std::string& dict_file = obj["dict_file"].asString();
  if (load_dict && dict_file != "" && !sc->LoadFromDictFile(dict_file)) {
    LOG(ERROR) << "Load dict file '" << dict_file << "' error.";
    return false;
  }
  sc->UpdateDefault();
  (*column).reset(sc);

  return true;
}

bool ConfParser::CreateFloatListColumn(const xJson::Value& obj,
                                       std::shared_ptr<BasicColumn>* column) {
  const std::string& iname = obj["iname"].asString();
  const std::string& alias = obj["alias"].asString();
  if (!checkJsonArgs(obj, "width", int32_tag, "scan_from", string_tag,
                     "reverse", bool_tag)) {
    LOG(ERROR) << "Conf parse error";
    return false;
  }
  int width = obj["width"].asInt();
  const std::string& sf = obj["scan_from"].asString();
  bool reverse = obj["reverse"].asBool();
  SCAN_TYPE scan_from;
  if (sf == "head") {
    scan_from = SCAN_TYPE::HEAD;
  } else if (sf == "tail") {
    scan_from = SCAN_TYPE::TAIL;
  } else {
    LOG(ERROR) << "Conf parse error: scan_from";
    return false;
  }

  double def = 0.0;
  FloatListColumn* sc = new FloatListColumn(
      ctx_, iname, alias, Field::CID::CID_END, def, width, scan_from, reverse);
  (*column).reset(sc);

  return true;
}

bool ConfParser::CreateWeightedStringListColumn(
    const xJson::Value& obj, std::shared_ptr<BasicColumn>* column,
    bool load_dict) {
  const std::string& iname = obj["iname"].asString();
  const std::string& alias = obj["alias"].asString();
  if (!checkJsonArgs(obj, "min_count", int32_tag, "top_k", int32_tag,
                     "oov_buckets", int32_tag, "default", string_tag,
                     "dict_file", string_tag, "width", int32_tag, "min_weight",
                     double_tag, "scan_from", string_tag, "reverse",
                     bool_tag)) {
    LOG(ERROR) << "Conf parse error";
    return false;
  }
  int min_count = obj["min_count"].asInt();
  int top_k = obj["top_k"].asInt();
  int oov_buckets = obj["oov_buckets"].asInt();
  int width = obj["width"].asInt();
  double min_weight = obj["min_weight"].asDouble();
  const std::string& default_key = obj["default"].asString();
  const std::string& sf = obj["scan_from"].asString();
  bool reverse = obj["reverse"].asBool();
  SCAN_TYPE scan_from;
  if (sf == "head") {
    scan_from = SCAN_TYPE::HEAD;
  } else if (sf == "tail") {
    scan_from = SCAN_TYPE::TAIL;
  } else {
    LOG(ERROR) << "Conf parse error: scan_from";
    return false;
  }
  WeightedStringListColumn* sc = new WeightedStringListColumn(
      ctx_, iname, alias, Field::CID::CID_END, -1, min_count, top_k, width,
      min_weight, oov_buckets, default_key, scan_from, reverse);
  const std::string& dict_file = obj["dict_file"].asString();
  if (load_dict && dict_file != "" && !sc->LoadFromDictFile(dict_file)) {
    LOG(ERROR) << "Load dict file '" << dict_file << "' error.";
    return false;
  }
  sc->UpdateDefault();
  (*column).reset(sc);

  return true;
}

bool ConfParser::CreateTripleListColumn(
    const xJson::Value& obj, std::shared_ptr<BasicColumn>* column,
    bool load_dict) {
  const std::string& iname = obj["iname"].asString();
  const std::string& alias = obj["alias"].asString();
  if (!checkJsonArgs(obj, "min_count", int32_tag, "top_k", int32_tag,
                     "oov_buckets", int32_tag, "default", string_tag,
                     "dict_file", string_tag, "width_pos", int32_tag,
                     "width_neg", int32_tag, "filter_type", string_tag,
                     "params", object_tag, "scan_from", string_tag, "reverse",
                     bool_tag)) {
    LOG(ERROR) << "Conf parse error";
    return false;
  }
  int min_count = obj["min_count"].asInt();
  int top_k = obj["top_k"].asInt();
  int oov_buckets = obj["oov_buckets"].asInt();
  int width_pos = obj["width_pos"].asInt();
  int width_neg = obj["width_neg"].asInt();
  const std::string& default_key = obj["default"].asString();
  std::string filter_type = obj["filter_type"].asString();
  xJson::Value params = obj["params"];
  const std::string& sf = obj["scan_from"].asString();
  bool reverse = obj["reverse"].asBool();
  SCAN_TYPE scan_from;
  if (sf == "head") {
    scan_from = SCAN_TYPE::HEAD;
  } else if (sf == "tail") {
    scan_from = SCAN_TYPE::TAIL;
  } else {
    LOG(ERROR) << "Conf parse error: scan_from";
    return false;
  }

  TripleListColumn* sc = new TripleListColumn(
      ctx_, iname, alias, Field::CID::CID_END, -1, min_count, top_k, width_pos,
      width_neg, oov_buckets, default_key, filter_type, params, scan_from,
      reverse);

  const std::string& dict_file = obj["dict_file"].asString();
  if (load_dict && dict_file != "" && !sc->LoadFromDictFile(dict_file)) {
    LOG(ERROR) << "Load dict file '" << dict_file << "' error.";
    return false;
  }
  sc->UpdateDefault();
  (*column).reset(sc);

  return true;
}

void ConfParser::PrintDebugInfo() {
  LOG(INFO) << "--------------- ConfParser Info ---------------";
  LOG(INFO) << "#columns = " << columns_.size();
  for (auto& c : columns_) {
    LOG(INFO) << "iname = " << c->iname() << ", alias = " << c->alias()
              << ", type = " << Field::TYPE_STR(c->type())
              << ", def = " << c->def() << ", width = " << c->width();
  }
  LOG(INFO) << "meta_file = " << data_paths_.meta_file;
  LOG(INFO) << "------------- ConfParser Info End -------------";
}

}  // namespace assembler
