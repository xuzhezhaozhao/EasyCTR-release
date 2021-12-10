#ifndef ASSEMBLER_FEATURE_H_
#define ASSEMBLER_FEATURE_H_

#include <iostream>
#include <string>
#include <vector>

#include "assembler/jsoncpp_helper.hpp"
#include "assembler/utils.h"
#include "deps/jsoncpp/json/json.h"

namespace assembler {

using ::utils::jsoncpp_helper::checkJsonArgs;
using ::utils::jsoncpp_helper::int32_tag;
using ::utils::jsoncpp_helper::string_tag;

struct Field {
  enum Type {
    INT = 0,
    FLOAT,
    STRING,
    STRING_LIST,
    FLOAT_LIST,
    WEIGHTED_STRING_LIST,
    TRIPLE_LIST,
    TYPE_END,
  } type;
  static const char* TYPE_STR(Field::Type type) {
    static const char* type_str[] = {"int",
                                     "float",
                                     "string",
                                     "string_list",
                                     "float_list",
                                     "weighted_string_list",
                                     "triple_list"};
    return type_str[type];
  }

  enum CID { USER = 0, ITEM, CTX, EXTRA, TARGET, CID_END } cid;
  static const char* CID_STR(Field::CID cid) {
    static const char* cid_str[] = {"user", "item", "ctx", "extra", "target"};
    return cid_str[cid];
  }

  std::string id;
  void Serialize(xJson::Value* f) const {
    (*f)["type"] = type;
    (*f)["cid"] = cid;
    (*f)["id"] = id;
  }
  void Parse(const xJson::Value& f) {
    if (!checkJsonArgs(f, "type", int32_tag, "cid", int32_tag, "id",
                       string_tag)) {
      LOG(ERROR) << "parse field error";
      throw std::logic_error("Parse error");
    }
    type = static_cast<Field::Type>(f["type"].asInt());
    cid = static_cast<Field::CID>(f["cid"].asInt());
    id = f["id"].asString();
  }

  void PrintDebugInfo() {
    LOG(INFO) << "type = " << type;
    LOG(INFO) << "cid = " << cid;
    LOG(INFO) << "id = " << id;
  }
};

typedef std::vector<double> Feature;
struct Example {
  Feature feature;
  double label = 0.0;
  double weight = 0.0;

  Example() {}
  Example(const Feature& feature_, double label_, double weight_)
      : feature(feature_), label(label_), weight(weight_) {}
};

}  // namespace assembler

#endif  // ASSEMBLER_FEATURE_H_
