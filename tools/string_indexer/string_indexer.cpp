#include <algorithm>
#include <fstream>

#include "assembler/utils.h"
#include "tools/string_indexer/string_indexer.h"

namespace assembler {
namespace tools {

void StringIndexer::WriteToFiles() {
  for (const auto& item : indexer_) {
    std::string output = output_dir_ + "/" + item.first + ".dict";
    std::ofstream ofs(output);
    if (!ofs.is_open()) {
      LOG(FATAL) << "Open output data file '" << output
                 << "' failed. errmsg = " << strerror(errno);
    }

    std::vector<std::pair<int, std::string>> dict;
    for (const auto& p : item.second) {
      dict.push_back({p.second, p.first});
    }
    std::sort(dict.begin(), dict.end(),
              std::greater<std::pair<int, std::string>>());

    for (const auto& p : dict) {
      ofs.write(p.second.data(), p.second.size());
      ofs.write("|", 1);
      auto s = std::to_string(p.first);
      ofs.write(s.data(), s.size());
      ofs.write("\n", 1);
    }
    ofs.close();
  }

  for (const auto& p : stats_) {
    std::string output = output_dir_ + "/" + p.first + ".stat";
    std::ofstream ofs(output);
    if (!ofs.is_open()) {
      LOG(FATAL) << "Open output data file '" << output
                 << "' failed. errmsg = " << strerror(errno);
    }
    double mean = p.second.sum / p.second.cnt;
    double stddev = p.second.squares_sum / p.second.cnt -
        p.second.sum * p.second.sum / (p.second.cnt * p.second.cnt);
    stddev = std::pow(stddev, 0.5);

    double log_mean = p.second.log_sum / p.second.cnt;
    double log_stddev = p.second.log_squares_sum / p.second.cnt -
        p.second.log_sum * p.second.log_sum / (p.second.cnt * p.second.cnt);
    log_stddev = std::pow(log_stddev, 0.5);

    auto s = std::to_string(mean);
    ofs.write(s.data(), s.size());
    ofs.write(",", 1);
    s = std::to_string(stddev);
    ofs.write(s.data(), s.size());
    ofs.write(",", 1);
    s = std::to_string(log_mean);
    ofs.write(s.data(), s.size());
    ofs.write(",", 1);
    s = std::to_string(log_stddev);
    ofs.write(s.data(), s.size());
  }
}
void StringIndexer::Index(const std::string& data_path) {
  std::ifstream ifs(data_path);
  if (!ifs.is_open()) {
    LOG(FATAL) << "Open data file '" << data_path << "' failed.";
  }
  std::string line;
  int lineindex = 0;
  LOG(INFO) << "Index file " << data_path << " ...";
  while (!ifs.eof()) {
    ++lineindex;
    if (lineindex % 100000 == 0) {
      LOG(INFO) << "Parse " << lineindex / 10000 << "w lines ...";
    }
    std::getline(ifs, line);
    if (line.empty()) {
      continue;
    }
    std::vector<StringPiece> tokens;
    utils::Split(line, ";\t", &tokens);
    std::string id;
    std::string value;
    for (size_t i = 0; i < tokens.size(); i++) {
      if (tokens[i].empty()) {
        continue;
      }
      size_t x = tokens[i].find('|');
      if (x == StringPiece::npos) {
        // TODO(zhezhaoxu) 不 fatal, 兼容目前的脏数据
        LOG(ERROR) << "train data format error, no '|', lineindex = "
                   << lineindex << ", line = '" << line << "'";
        continue;
      }
      id = tokens[i].substr(0, x).ToString();
      if (id == "") {
        // TODO(zhezhaoxu) 不 fatal, 兼容目前的脏数据
        LOG(ERROR) << "empty id, lineindex = " << lineindex;
        continue;
      }
      // feature id
      auto it = mm_.meta().find(id);
      if (it == mm_.meta().end()) {
        continue;
      }
      std::string fname = mm_.id2name().at(id);
      value = tokens[i].substr(x + 1).ToString();
      Field field = it->second;
      if (field.type == Field::Type::STRING) {
        if (value != "") {
          ++indexer_[fname][value];
        } else {
          indexer_[fname]["</empty_token>"] = 0;
        }
      } else if (field.type == Field::Type::STRING_LIST) {
        auto sub_tokens = utils::Split(value, ",");
        for (const auto& sub_token : sub_tokens) {
          auto word = sub_token.ToString();
          if (word != "") {
            ++indexer_[fname][word];
          } else {
            indexer_[fname]["</empty_token>"] = 0;
          }
        }
      } else if (field.type == Field::Type::WEIGHTED_STRING_LIST) {
        auto sub_tokens = utils::Split(value, ",");
        for (const auto& sub_token : sub_tokens) {
          if (sub_token.empty()) {
            continue;
          }
          auto e = utils::Split(sub_token, ":");
          if (e.size() != 2) {
            LOG(ERROR) << fname << " format error, size is not 2, lineidx = "
                       << lineindex << ", subtoken = " << sub_token
                       << ", line = " << line;
            LOG(ERROR) << "token=" << value << ".";
            // TODO(zhezhaoxu) 不 fatal, 兼容目前的脏数据
            LOG(ERROR) << "sub_tokens.size = " << sub_tokens.size();
            continue;
          }
          auto word = e[0].ToString();
          if (word != "") {
            ++indexer_[fname][word];
          } else {
            indexer_[fname]["</empty_token>"] = 0;
          }
        }
      } else if (field.type == Field::Type::TRIPLE_LIST) {
        auto sub_tokens = utils::Split(value, ",");
        for (const auto& sub_token : sub_tokens) {
          if (sub_token.empty()) {
            continue;
          }
          auto e = utils::Split(sub_token, ":");
          if (e.size() != 3) {
            LOG(ERROR) << fname << " format error, size is not 3, lineidx = "
                       << lineindex << ", subtoken = " << sub_token
                       << ", line = " << line;
            LOG(ERROR) << "token=" << value << ".";
            // TODO(zhezhaoxu) 不 fatal, 兼容目前的脏数据
            LOG(ERROR) << "sub_tokens.size = " << sub_tokens.size();
            continue;
          }
          auto word = e[0].ToString();
          if (word != "") {
            ++indexer_[fname][word];
          } else {
            indexer_[fname]["</empty_token>"] = 0;
          }
        }
      } else if (field.type == Field::Type::INT ||
                 field.type == Field::Type::FLOAT) {
        if (value != "") {
          double x = 0;
          try {
            x = std::stod(value);
          } catch (const std::exception& e) {
            LOG(ERROR) << fname << " format error, to double failed, lineidx = "
                       << lineindex << ", line = " << line;
            continue;
          }
          stats_[fname].sum += x;
          stats_[fname].squares_sum += x * x;

          x = std::max(x + 1.0, 1.0);
          stats_[fname].log_sum += log(x);
          stats_[fname].log_squares_sum += log(x) * log(x);
          stats_[fname].cnt += 1;
        }
      }
    }
  }
}

}  // namespace tools
}  // namespace assembler
