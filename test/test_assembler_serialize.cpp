
#include "../assembler/assembler.h"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "<Usage>: <conf>" << std::endl;
    exit(-1);
  }

  // TODO(zhezhaoxu) Construct real ctx
  assembler::Assembler ab(NULL);
  std::cout << "Init ..." << std::endl;
  ab.Init(argv[1]);
  std::cout << "Init done" << std::endl;
  std::string serialized;
  ab.Serialize(&serialized);
  std::cout << "-----------------" << std::endl;
  std::cout << "serialized" << std::endl;
  std::cout << serialized << std::endl;
  std::cout << "-----------------" << std::endl;
  std::cout << "origin: " << std::endl;
  ab.PrintDebugInfo();

  std::cout << "parsing ..." << std::endl;
  // TODO(zhezhaoxu) Construct real ctx
  assembler::Assembler reconstruct(NULL);
  reconstruct.ParseFromString(serialized);
  std::cout << "parsing done" << std::endl;

  std::cout << "reconstruct: " << std::endl;
  reconstruct.PrintDebugInfo();

  return 0;
}
