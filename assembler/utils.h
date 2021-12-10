#ifndef ASSEMBLER_UTILS_H_
#define ASSEMBLER_UTILS_H_

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/env.h"

#include "assembler/stringpiece.h"

namespace assembler {

struct Example;

namespace utils {
std::vector<StringPiece> Split(StringPiece text, const std::string& delim);
void Split(StringPiece text, const std::string& delim,
           std::vector<StringPiece>* result);
void PrintExample(const Example& example);
void WriteString(std::ostringstream& oss, const std::string& s);
void WriteInt(std::ostringstream& oss, int v);
void WriteSize(std::ostringstream& oss, size_t v);
void WriteFloat(std::ostringstream& oss, float v);
void ReadString(std::istringstream& iss, std::string* s);
void ReadInt(std::istringstream& iss, int* v);
void ReadSize(std::istringstream& iss, size_t* v);
void ReadFloat(std::istringstream& iss, float* v);
std::map<std::string, int> ParseMeta(const std::string& meta_str);
bool LoadFromDictFile(::tensorflow::Env* env, const std::string& dict_file,
                      int min_count, int top_k,
                      std::unordered_map<std::string, int>* indexer,
                      std::vector<std::string>* keys);

// also see: https://github.com/google/highwayhash
uint32_t MurMurHash3(const std::string& key);

}  // namespace utils
}  // namespace assembler

#endif  // ASSEMBLER_UTILS_H_
