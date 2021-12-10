
#include <cmath>
#include <unordered_map>
#include <utility>

#include "assembler/triple_list_filter.h"

namespace assembler {

// 对于三元组类型的特征，可以通过 filter 函数筛选出正、负记录列表。
//
// 返回值:
//   第一个整数:
//     1: 正记录
//     0: 负记录
//     -1: 丢弃
//   第二个浮点数:
//     权重
// 参数:
//   @a: 三元组第二个参数
//   @b: 三元组第三个参数
//   @params: 自定义参数
static std::pair<int, double> default_filter(StringPiece a, StringPiece b,
                                             const xJson::Value& params) {
  return std::make_pair(1, 1.0);
}

std::unordered_map<std::string,
                   std::function<std::pair<int, double>(
                       StringPiece, StringPiece, const xJson::Value&)>>
    global_filter_map = {{"default_filter", default_filter}};

}  // namespace assembler
