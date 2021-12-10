
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
  srand((unsigned)time(NULL));

  if (argc != 3) {
    std::cout << "Usage: <input> <output>" << std::endl;
    exit(-1);
  }
  std::string input(argv[1]);
  std::string output(argv[2]);
  std::vector<std::string> filenames;
  std::ifstream ifs(input);
  if (!ifs.is_open()) {
    std::cout << "Open input file error." << std::endl;
    exit(-1);
  }
  std::string line;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    if (line.empty()) {
      continue;
    }
    std::cout << "shuffle filename: " << line << std::endl;
    filenames.push_back(line);
  }
  ifs.close();

  std::vector<std::ifstream *> inputs;
  std::ofstream ofs(output);
  if (!ofs.is_open()) {
    std::cout << "Open output file error." << std::endl;
    exit(-1);
  }
  for (auto& f : filenames) {
    inputs.push_back(new std::ifstream(f));
    if (!inputs.back()->is_open()) {
      std::cout << "Open input file '" << f << "' error." << std::endl;
      exit(-1);
    }
  }
  while (!inputs.empty()) {
    int idx = rand() % inputs.size();
    if (inputs[idx]->eof()) {
      delete inputs[idx];
      inputs.erase(inputs.begin() + idx, inputs.begin() + idx +1);
      continue;
    }
    std::getline(*inputs[idx], line);
    if (line.empty() && inputs[idx]->eof()) {
      continue;
    }
    ofs.write(line.data(), line.size());
    ofs.write("\n", 1);
  }

  ofs.close();
  return 0;
}
