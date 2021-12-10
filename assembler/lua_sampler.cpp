
#include "assembler/lua_sampler.h"

#include "tensorflow/core/platform/default/logging.h"

#include "assembler/utils.h"

namespace assembler {

LuaSampler::~LuaSampler() {
  if (L) {
    lua_close(L);
  }
}

bool LuaSampler::Init(const std::string& script) {
  LOG(INFO) << "LuaSampler Init ...";
  L = luaL_newstate();
  if (!L) {
    LOG(ERROR) << "luaL_newstate error";
    return false;
  }
  luaL_openlibs(L);
  if (luaL_dofile(L, script.c_str()) != 0) {
    LOG(ERROR) << "luaL_dofile error, script = '" << script << "'";
    return false;
  }
  lua_getglobal(L, "features");
  if (!lua_isstring(L, -1)) {
    LOG(ERROR) << "lua sampler script error: 'features' is not string";
    return false;
  }
  std::string s = lua_tostring(L, -1);
  lua_pop(L, 1);

  auto tokens = utils::Split(s, ",; \t");
  features_.clear();
  for (auto& token : tokens) {
    if (token.empty()) {
      continue;
    }
    features_.push_back(token.ToString());
  }
  LOG(INFO) << "LuaSampler Init done, features size = " << features_.size();
  return true;
}

std::pair<double, double> LuaSampler::get_label_and_weight(
    const std::vector<std::string>& params) {
  std::pair<double, double> p;

  {
    // lock
    ::tensorflow::mutex_lock l(mu_);
    lua_getglobal(L, "get_label_and_weight");
    for (const auto& param : params) {
      lua_pushstring(L, param.data());
    }
    int err = lua_pcall(L, params.size(), 2, 0);
    if (err == 0) {
      p.first = lua_tonumber(L, -2);   // label
      p.second = lua_tonumber(L, -1);  // weight
    } else {
      LOG(ERROR) << "lua_pcall failed, err = " << err;
      throw std::logic_error("lua_pcall failed.");
    }
    lua_pop(L, 2);
  }

  return p;
}

}  // namespace assembler
