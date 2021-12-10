/*******************************************************
 @Author: zhezhaoxu
 @Created Time : 2017年08月31日 星期四 22时03分51秒
 @File Name: jsoncpp_helper.hpp
 @Description:
 ******************************************************/

/**
 * APIs:
 * 命名空间 utils::jsoncpp_helper;
 *
 * 1. 校验 json 字段
 * template <typename... Args>
 * bool checkJsonArgs(const Json::Value& value, Args... args);
 *
 * 参数类型为变长模板参数, 第一个参数是 Json::Value 类型, 为待校验的json数据;
 * 之后的参数为多个 (key, type) 对的形式. 例如：
 *
 *  using namespace utils::jsoncpp_helper;
 * 	Json::Value value;
 * 	value["id"] = 158;
 * 	value["qq"] = 123456;
 * 	value["wechat"] = "abc_abc";
 * 	value["null"] = Json::nullValue;
 * 	value["score"] = 95.5;
 * 	value["list"][0] = 1;
 * 	value["list"][1] = 2;
 * 	value["obj"]["home"] = "zhangshu";
 *
 * 	ASSERT_TRUE(checkJsonArgs(
 * 		value, "id", int32_tag, "qq", uint32_tag, "wechat", string_tag,
 *"null",
 * 		null_tag, "score", double_tag, "list", array_tag, "obj",
 *object_tag));
 *
 * (key, type)参数对中 key 的类型可以是 char* 和 std::string; type 有如下几种
 * 类型, * 对应 jsoncpp 支持的数据类型:
 * 	null_tag
 * 	bool_tag
 * 	int32_tag
 * 	int64_tag
 * 	uint32_tag
 * 	uint64_tag
 * 	integral_tag
 * 	double_tag
 * 	numeric_tag
 * 	string_tag
 * 	array_tag
 * 	object_tag
 *
 * 2. 设置 json 字段数据
 * template <typename... Args>
 * void setJsonValue(Json::Value& value, Args... args);
 *
 * 参数类型为变长模板参数,第一个参数是 Json::Value 类型, 为待设置的json数据;
 * 之后的参数为多个 (key, value) 对的形式. 例如：
 *
 *	using namespace utils::jsoncpp_helper;
 *	Json::Value value;
 *	setJsonValue(value, "name", "zhezhaoxu", "id", 158, "good", true,
 *"score",
 *				 95.5);
 *	ASSERT_TRUE(checkJsonArgs(value, "name", string_tag, "id", int32_tag,
 *							  "good", bool_tag,
 *"score",
 *double_tag));
 *	ASSERT_TRUE(value["name"] == "zhezhaoxu");
 *	ASSERT_TRUE(value["id"] == 158);
 *	ASSERT_TRUE(value["good"] == true);
 *	ASSERT_TRUE(value["score"] == 95.5);
 *
 * 3. 设置 json 数组数据
 * template <typename... Args>
 * void setJsonArrayValue(Json::Value& arrayValue, Args... args);
 *
 * 参数类型为变长模板参数, 第一个参数是 Json::Value 类型, 为待设置的json数据;
 * 之后的参数有两种格式：
 * (1) 多个 value 的形式. 例如:
 * using namespace utils::jsoncpp_helper;
 * 	Json::Value value;
 * 	setJsonArrayValue(value["array"], 158, 19930326, "zhezhao", true, 95.5);
 * 	ASSERT_TRUE(checkJsonArgs(value, "array", array_tag));
 * 	ASSERT_TRUE(checkJsonArgs(value["array"][0] == 158));
 * 	ASSERT_TRUE(checkJsonArgs(value["array"][1] == 19930326));
 * 	ASSERT_TRUE(checkJsonArgs(value["array"][2] == "zhezhao"));
 * 	ASSERT_TRUE(checkJsonArgs(value["array"][3] == true));
 * 	ASSERT_TRUE(checkJsonArgs(value["array"][4] == 95.5));
 *
 * 	(2) (intput, N) 的形式
 * 	input 为输入数组, N为数组大小; 例如:
 *	std::vector<int> tickets = {1, 2, 3};
 *	setJsonArrayValue(value["tickets"], tickets, 3);
 *	ASSERT_TRUE(checkJsonArgs(value, "tickets", array_tag));
 *	EXPECT_TRUE(value["tickets"][0] == 1);
 *	EXPECT_TRUE(value["tickets"][1] == 2);
 *	EXPECT_TRUE(value["tickets"][2] == 3);
 *
 */

#ifndef UTILS_JSONCPP_HELPER_H_
#define UTILS_JSONCPP_HELPER_H_

#include "deps/jsoncpp/json/json.h"
#include "tensorflow/core/platform/default/logging.h"

namespace utils {
namespace jsoncpp_helper {

struct null_tag_t {};
struct bool_tag_t {};
struct int32_tag_t {};
struct int64_tag_t {};
struct uint32_tag_t {};
struct uint64_tag_t {};
struct integral_tag_t {};
struct double_tag_t {};
struct numeric_tag_t {};
struct string_tag_t {};
struct array_tag_t {};
struct object_tag_t {};

// initialize to suppress compile error for gcc 4.4
static const struct null_tag_t null_tag = {};
static const struct bool_tag_t bool_tag = {};
static const struct int32_tag_t int32_tag = {};
static const struct int64_tag_t int64_tag = {};
static const struct uint32_tag_t uint32_tag = {};
static const struct uint64_tag_t uint64_tag = {};
static const struct integral_tag_t integral_tag = {};
static const struct double_tag_t double_tag = {};
static const struct numeric_tag_t numeric_tag = {};
static const struct string_tag_t string_tag = {};
static const struct array_tag_t array_tag = {};
static const struct object_tag_t object_tag = {};

// base
inline bool __checkJsonArgs(const xJson::Value&) { return true; }

#define DEFINE_checkJsonArgs(type1, type2)                             \
  template <typename T, typename... Args>                              \
  bool __checkJsonArgs(const xJson::Value& value, const T& key, type1, \
                       Args... args) {                                 \
    if (!(value.isMember(key) && value[key].type2())) {                \
      LOG(ERROR) << "[checkJsonArgs] Error: key = '" << key << "'";    \
      return false;                                                    \
    }                                                                  \
    return __checkJsonArgs(value, args...);                            \
  }

DEFINE_checkJsonArgs(null_tag_t, isNull);
DEFINE_checkJsonArgs(bool_tag_t, isBool);
DEFINE_checkJsonArgs(int32_tag_t, isInt);
DEFINE_checkJsonArgs(int64_tag_t, isInt64);
DEFINE_checkJsonArgs(uint32_tag_t, isUInt);
DEFINE_checkJsonArgs(uint64_tag_t, isUInt64);
DEFINE_checkJsonArgs(integral_tag_t, isIntegral);
DEFINE_checkJsonArgs(double_tag_t, isDouble);
DEFINE_checkJsonArgs(numeric_tag_t, isNumeric);
DEFINE_checkJsonArgs(string_tag_t, isString);
DEFINE_checkJsonArgs(array_tag_t, isArray);
DEFINE_checkJsonArgs(object_tag_t, isObject);

#undef DEFINE_checkJsonArgs

template <typename T>
struct is_jsontype {
  enum { value = 0 };
};
template <>
struct is_jsontype<bool> {
  enum { value = 1 };
};
template <>
struct is_jsontype<int32_t> {
  enum { value = 1 };
};
template <>
struct is_jsontype<uint32_t> {
  enum { value = 1 };
};
template <>
struct is_jsontype<int64_t> {
  enum { value = 1 };
};
template <>
struct is_jsontype<uint64_t> {
  enum { value = 1 };
};
template <>
struct is_jsontype<double> {
  enum { value = 1 };
};
template <>
struct is_jsontype<const char*> {
  enum { value = 1 };
};
template <>
struct is_jsontype<std::string> {
  enum { value = 1 };
};
template <>
struct is_jsontype<xJson::Value> {
  enum { value = 1 };
};

// base
inline void __setJsonValue(xJson::Value&) {}

template <typename T, typename... Args,
          typename = typename std::enable_if<is_jsontype<
              typename std::remove_reference<T>::type>::value>::type>
void __setJsonValue(xJson::Value& value, const char* key, T&& val,
                    Args... args) {
  value[key] = std::forward<T>(val);
  __setJsonValue(value, args...);
}

// base
inline void __setJsonArrayValue(xJson::Value&) {}

template <typename T, typename... Args,
          typename = typename std::enable_if<is_jsontype<
              typename std::remove_reference<T>::type>::value>::type>
void __setJsonArrayValue(xJson::Value& arrayValue, T&& val, Args... args) {
  arrayValue[arrayValue.size()] = std::forward<T>(val);
  __setJsonArrayValue(arrayValue, args...);
}

// 利用 SFINAE
template <typename T>
void __setJsonArrayValue(xJson::Value& arrayValue, T input, size_t N) {
  for (xJson::ArrayIndex index = 0; index < N; ++index) {
    arrayValue[index] = input[index];
  }
}

/********* API ****************************/

// 检查value中字段类型, 使用方法如下
// checkJsonArgs(value, key1, int32_tag, key2, string_tag);
template <typename... Args>
bool checkJsonArgs(const xJson::Value& value, Args... args) {
  return __checkJsonArgs(value, args...);
}

// 设置 value 中的字段,使用方法如下：
// setJsonValue(value, key1, val1, key2, val2, key3, val3, ...);
// key* 为字符串类型, val* 为 jsoncpp 支持的任意类型
template <typename... Args>
void setJsonValue(xJson::Value& value, Args... args) {
  __setJsonValue(value, args...);
}

// 设置数组类型的值,使用方法如下：
// 有两种形式：
// 1. setJsonArrayValue(arrayValue, val1, val2, val3, ...);
//  val* 为 jsoncpp 支持的任意类型
// 2. setJsonArrayValue(arrayValue, input, N);
//  input 类型只要重载了 [] 操作符就行,N 为数组大小
template <typename... Args>
void setJsonArrayValue(xJson::Value& arrayValue, Args... args) {
  __setJsonArrayValue(arrayValue, args...);
}

/******** end API ***********************/

}  // namespace jsoncpp_helper
}  // namespace utils

#endif
