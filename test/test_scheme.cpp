#include <fstream>
#include <iostream>

#include "deps/jsoncpp/json/json.h"
#include "assembler/assembler.h"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "<Usage>: <conf>" << std::endl;
    exit(-1);
  }

  // TODO(zhezhaoxu) Create real ctx
  assembler::Assembler assembler(NULL);
  assembler.Init(argv[1]);

  xJson::Value scheme = assembler.GetFeatureScheme();
  std::cout << scheme.toStyledString() << std::endl;

  return 0;
}
