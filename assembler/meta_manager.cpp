
#include "assembler/meta_manager.h"

#include <string>
#include <vector>

#include "tensorflow/core/platform/file_system.h"

#include "assembler/file_reader.h"

namespace assembler {

MetaManager::MetaManager(::tensorflow::Env* env) : env_(env) {}

void MetaManager::Init(const std::string& meta_path) {
  FileReader file_reader(env_, meta_path);
  if (!file_reader.Init()) {
    LOG(ERROR) << "Open meta file '" << meta_path << "' failed.";
    throw std::logic_error("Open meta file failed");
  }

  std::string line;
  int lineindex = 0;
  while (file_reader.ReadLine(&line)) {
    ++lineindex;
    if (lineindex == 1 || line.empty()) {
      // skip header and empty line
      continue;
    }
    if (line[0] == '#') {
      continue;
    }

    // id, name, data_type, class_type
    std::vector<StringPiece> tokens;
    utils::Split(line, " \t;", &tokens);
    if (tokens.size() < 4) {
      LOG(ERROR) << "meta file format error in line " << lineindex << ": "
                 << "tokens' size is less than 4.";
      throw std::logic_error("meta error");
    }
    std::string id = tokens[0].ToString();
    std::string name = tokens[1].ToString();
    std::string data_type = tokens[2].ToString();
    std::string class_type = tokens[3].ToString();

    Field::Type type;
    if (data_type == "int" || data_type == "uint32") {
      type = Field::Type::INT;
    } else if (data_type == "string") {
      type = Field::Type::STRING;
    } else if (data_type == "float") {
      type = Field::Type::FLOAT;
    } else if (data_type == "string_list") {
      type = Field::Type::STRING_LIST;
    } else if (data_type == "float_list") {
      type = Field::Type::FLOAT_LIST;
    } else if (data_type == "weighted_string_list") {
      type = Field::Type::WEIGHTED_STRING_LIST;
    } else if (data_type == "triple_list") {
      type = Field::Type::TRIPLE_LIST;
    } else {
      LOG(ERROR) << "meta file format error in line " << lineindex
                 << ": Unknow data type '" << data_type << "'.";
      // DO NOT THROW EXCEPTION
      continue;
    }
    if (id == "") {
      LOG(ERROR) << "meta file format error in line " << lineindex
                 << ": field id is empty";
      throw std::logic_error("meta error");
    }
    if (meta_.count(id) > 0) {
      LOG(ERROR) << "meta file format error in line " << lineindex
                 << ": field id duplicated.";
      throw std::logic_error("meta error");
    }

    Field::CID cid;
    if (class_type == "user") {
      cid = Field::CID::USER;
    } else if (class_type == "item") {
      cid = Field::CID::ITEM;
    } else if (class_type == "ctx") {
      cid = Field::CID::CTX;
    } else if (class_type == "extra") {
      cid = Field::CID::EXTRA;
    } else if (class_type == "target") {
      cid = Field::CID::TARGET;
    } else {
      LOG(ERROR) << "meta file format error in line " << lineindex
                 << ": unknow class type = " << class_type;
      // DO NOT THROW EXCEPTION
      continue;
    }
    meta_[id] = {type, cid, id};
    name2id_[name] = id;
    id2name_[id] = name;
  }
}

}  // namespace assembler
