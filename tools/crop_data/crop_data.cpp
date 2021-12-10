#include <cassert>
#include <fstream>
#include <iostream>
#include <set>

#include "assembler/utils.h"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage: <data> <dict>" << std::endl;
    exit(-1);
  }

  std::ifstream data_ifs(argv[1]);
  std::ifstream dict_ifs(argv[2]);
  assert(data_ifs.is_open());
  assert(dict_ifs.is_open());

  std::set<std::string> dict;
  std::string line;
  while (!dict_ifs.eof()) {
    std::getline(dict_ifs, line);
    if (line == "") {
      continue;
    }
    auto tokens = assembler::utils::Split(line, " \t");
    dict.insert(tokens[0].ToString());
  }

  while (!data_ifs.eof()) {
    std::getline(data_ifs, line);
    if (line == "") {
      continue;
    }
    auto tokens = assembler::utils::Split(line, " \t");
    if (dict.count(tokens[0].ToString()) > 0) {
      std::cout << line << std::endl;
    }
  }

  return 0;
}
