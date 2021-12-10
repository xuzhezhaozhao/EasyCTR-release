#include <cassert>
#include <fstream>
#include <iostream>
#include <set>

#include "assembler/utils.h"

int main(int argc, char* argv[]) {
  if (argc != 7) {
    std::cerr << "Usage: <input_embedding> <skip-lines> <dict> "
                 "<min_count> <top_k> <output_embedding>"
              << std::endl;
    exit(-1);
  }

  std::ifstream data_ifs(argv[1]);
  int skip_lines = std::stoi(argv[2]);
  std::ifstream dict_ifs(argv[3]);
  int min_count = std::stoi(argv[4]);
  int top_k = std::stoi(argv[5]);
  std::ofstream output(argv[6]);
  if (!data_ifs.is_open() || !dict_ifs.is_open() || !output.is_open()) {
    std::cerr << "open file error" << std::endl;
    exit(-1);
  }

  // 构造词典
  std::vector<std::string> dict;
  std::string line;
  while (!dict_ifs.eof()) {
    std::getline(dict_ifs, line);
    if (line == "") {
      continue;
    }
    auto tokens = assembler::utils::Split(line, "|");
    int freq = std::stoi(tokens[1].ToString());
    if (freq < min_count) {
      continue;
    }
    dict.push_back(tokens[0].ToString());
    if (top_k > 0 && (int)dict.size() >= top_k) {
      break;
    }
  }
  dict_ifs.close();

  // 读取 pre-trained embedding
  std::map<std::string, std::string> embeds;
  int dim = -1;
  int lineindex = 0;
  while (!data_ifs.eof()) {
    std::getline(data_ifs, line);
    ++lineindex;
    if (lineindex % 100000 == 0) {
      std::cerr << "load " << lineindex / 10000 << "w lines ..." << std::endl;
    }
    if (lineindex <= skip_lines) {
      continue;
    }
    if (line == "") {
      continue;
    }
    while (line.back() == ' ' or line.back() == '\t') {
      line.pop_back();
    }
    auto tokens = assembler::utils::Split(line, " \t");
    if (tokens.size() < 2) {
      std::cerr << "embedding error, tokens size < 2";
      exit(-1);
    }

    if (dim != -1 && dim != static_cast<int>(tokens.size()) - 1) {
      std::cerr << "embedding dim error." << std::endl;
      exit(-1);
    }
    dim = tokens.size() - 1;
    embeds.insert({tokens[0].ToString(), line});
  }
  data_ifs.close();

  if (dim == -1) {
    std::cerr << " embedding error, no valid data";
    exit(-1);
  }
  // 输出到文件
  std::string s;
  s = std::to_string(dict.size());
  output.write(s.data(), s.size());
  output.write(" ", 1);
  s = std::to_string(dim);
  output.write(s.data(), s.size());
  output.write("\n", 1);

  int cnt = 0;
  for (const auto& key : dict) {
    if (embeds.count(key) > 0) {
      output.write(embeds[key].data(), embeds[key].size());
      output.write("\n", 1);
    } else {
      output.write(key.data(), key.size());
      output.write(" #", 2);   // 默认值
      output.write("\n", 1);
      ++cnt;
    }
  }
  std::cerr << "[crop_embedding] rowkey not found cnt = " << cnt << std::endl;
  output.close();
  return 0;
}
