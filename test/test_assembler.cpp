#include <fstream>
#include <iostream>

#include "assembler/assembler.h"
#include "assembler/feature.h"
#include "assembler/utils.h"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "<Usage>: <conf> <input>" << std::endl;
    exit(-1);
  }

  // TODO(zhezhaoxu) Create real ctx
  assembler::Assembler assembler(NULL);
  assembler.Init(argv[1]);

  std::ifstream ifs(argv[2]);
  if (!ifs.is_open()) {
    std::cerr << "Open input file failed." << std::endl;
    exit(-1);
  }
  std::string line;
  int lineindex = 0;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    ++lineindex;
    if (line.empty()) {
      continue;
    }
    int dim = assembler.feature_size();
    assembler::Example example(assembler::Feature(dim), 0.0, 0.0);
    assembler.GetExample(line, true, &example);
    assembler::utils::PrintExample(example);
  }

  return 0;
}
