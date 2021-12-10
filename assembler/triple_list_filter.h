#ifndef ASSEMBLER_TRIPLE_LIST_FILTER_H_
#define ASSEMBLER_TRIPLE_LIST_FILTER_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

#include "assembler/stringpiece.h"
#include "deps/jsoncpp/json/json.h"

namespace assembler {

extern std::unordered_map<std::string,
                          std::function<std::pair<int, double>(
                              StringPiece, StringPiece, const xJson::Value&)>>
    global_filter_map;

}  // namespace assembler

#endif  // ASSEMBLER_TRIPLE_LIST_FILTER_H_
