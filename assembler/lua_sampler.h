#ifndef ASSEMBLER_LUA_SAMPLER_H_
#define ASSEMBLER_LUA_SAMPLER_H_

#include <string>
#include <vector>
#include <utility>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

extern "C" {
#include "deps/lua/src/lauxlib.h"
#include "deps/lua/src/lua.h"
#include "deps/lua/src/lualib.h"
}


namespace assembler {

class LuaSampler {
 public:
  LuaSampler() = default;
  ~LuaSampler();

  bool Init(const std::string& script);
  const std::vector<std::string>& features() const { return features_; }
  std::pair<double, double> get_label_and_weight(
      const std::vector<std::string>& params);

 private:
  // TODO(zhezhaoxu) use shared_ptr
  lua_State* L = NULL;
  std::vector<std::string> features_;
  ::tensorflow::mutex mu_;
};

}  // namespace assembler

#endif  // ASSEMBLER_LUA_SAMPLER_H_
