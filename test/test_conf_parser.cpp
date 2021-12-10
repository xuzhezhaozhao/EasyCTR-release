
#include <iostream>
#include <fstream>

#include "assembler/conf_parser.h"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "<Usage>: <conf>" << std::endl;
    exit(-1);
  }

  // TODO(zhezhaoxu) Create real
  assembler::ConfParser parser(NULL);
  std::ifstream ifs(argv[1]);
  std::ostringstream tmp;
  tmp << ifs.rdbuf();
  std::string c = tmp.str();
  bool ok = parser.Parse(c);
  std::cout << "Parse status: " << ok << std::endl;
  if (!ok) {
    exit(-1);
  }

  std::cout << "#column = " << parser.columns().size() << std::endl;
  for (auto p : parser.columns()) {
    std::cout << "iname = " << p->iname() << std::endl;
    std::cout << "-------" << std::endl;
  }

  return 0;
}
