
#include <iostream>

#include "tools/string_indexer/string_indexer.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cout << "Usage: <meta_path> <data_path> <output_dir>" << std::endl;
    exit(-1);
  }
  std::string meta_path(argv[1]);
  std::string data_path(argv[2]);
  std::string output_dir(argv[3]);
  ::tensorflow::Env* env = ::tensorflow::Env::Default();
  assembler::tools::StringIndexer indexer(env, meta_path, output_dir);
  indexer.Index(data_path);
  indexer.WriteToFiles();

  return 0;
}
