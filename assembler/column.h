#ifndef ASSEMBLER_COLUMN_H_
#define ASSEMBLER_COLUMN_H_

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "assembler/feature.h"
#include "assembler/stringpiece.h"
#include "assembler/triple_list_filter.h"
#include "deps/jsoncpp/json/json.h"

static const char* EXTRA_KEY = "_extra_";

namespace assembler {

enum SCAN_TYPE {
  HEAD = 0,
  TAIL,
  END,
};

class BasicColumn {
 public:
  explicit BasicColumn(::tensorflow::OpKernelConstruction* ctx) : ctx_(ctx) {}
  BasicColumn(::tensorflow::OpKernelConstruction* ctx, const std::string& iname,
              const std::string& alias, Field::Type type, Field::CID cid,
              double def, int width)
      : ctx_(ctx),
        iname_(iname),
        alias_(alias),
        type_(type),
        cid_(cid),
        def_(def),
        width_(width) {}
  virtual ~BasicColumn() {}
  virtual void ToValue(const std::string&, std::vector<double>::iterator,
                       bool*) const = 0;
  virtual void Serialize(xJson::Value* bc) const;
  virtual void ReConstruct(const xJson::Value& bc);

  static std::shared_ptr<BasicColumn> Parse(
      ::tensorflow::OpKernelConstruction* ctx, const xJson::Value& bc);

  ::tensorflow::OpKernelConstruction* ctx() { return ctx_; }
  const std::string& iname() const { return iname_; }
  const std::string& alias() const { return alias_; }
  Field::Type type() const { return type_; }
  Field::CID cid() const { return cid_; }
  void set_cid(Field::CID cid) { cid_ = cid; }
  double def() const { return def_; }
  void set_def(double def) { def_ = def; }
  int width() const { return width_; }

  virtual void PrintDebugInfo() const;

 protected:
  BasicColumn(const BasicColumn&) = delete;
  void operator=(const BasicColumn&) = delete;

  ::tensorflow::OpKernelConstruction* ctx_;
  std::string iname_;
  std::string alias_;
  Field::Type type_;
  Field::CID cid_;
  double def_;
  int width_;
};

class NumericColumn : public BasicColumn {
 public:
  explicit NumericColumn(::tensorflow::OpKernelConstruction* ctx)
      : BasicColumn(ctx) {}
  NumericColumn(::tensorflow::OpKernelConstruction* ctx,
                const std::string& iname, const std::string& alias,
                Field::Type type, Field::CID cid, double def)
      : BasicColumn(ctx, iname, alias, type, cid, def, 1) {}
  NumericColumn(const NumericColumn&) = default;
  NumericColumn& operator=(const NumericColumn&) = default;
  void ToValue(const std::string& key, std::vector<double>::iterator,
               bool*) const override;
  void Serialize(xJson::Value* bc) const override;
};

class FloatListColumn : public BasicColumn {
 public:
  explicit FloatListColumn(::tensorflow::OpKernelConstruction* ctx)
      : BasicColumn(ctx) {}
  FloatListColumn(::tensorflow::OpKernelConstruction* ctx,
                  const std::string& iname, const std::string& alias,
                  Field::CID cid, double d, int w, SCAN_TYPE st, bool reverse)
      : BasicColumn(ctx, iname, alias, Field::Type::FLOAT_LIST, cid, d, w),
        scan_from_(st),
        reverse_(reverse) {}
  FloatListColumn(const FloatListColumn&) = default;
  FloatListColumn& operator=(const FloatListColumn&) = default;
  void ToValue(const std::string&, std::vector<double>::iterator,
               bool*) const override;
  void Serialize(xJson::Value* bc) const override;

 private:
  SCAN_TYPE scan_from_;
  bool reverse_;
};

class StringBaseColumn : public BasicColumn {
 public:
  explicit StringBaseColumn(::tensorflow::OpKernelConstruction* ctx)
      : BasicColumn(ctx) {}
  StringBaseColumn(::tensorflow::OpKernelConstruction* ctx,
                   const std::string& iname, const std::string& alias,
                   Field::Type type, int width, Field::CID cid, double def,
                   int min_count, int top_k, int oov_buckets,
                   const std::string& default_key, SCAN_TYPE st, bool reverse)
      : BasicColumn(ctx, iname, alias, type, cid, def, width),
        min_count_(min_count),
        top_k_(top_k),
        oov_buckets_(oov_buckets),
        default_key_(default_key),
        scan_from_(st),
        reverse_(reverse) {}
  StringBaseColumn(const StringBaseColumn&) = default;
  StringBaseColumn& operator=(const StringBaseColumn&) = default;

  int min_count() const { return min_count_; }
  int top_k() const { return top_k_; }
  int oov_buckets() const { return oov_buckets_; }
  const std::unordered_map<std::string, int>& indexer() const {
    return indexer_;
  }
  SCAN_TYPE scan_from() const { return scan_from_; }
  bool reverse() const { return reverse_; }

  void Serialize(xJson::Value* bc) const override;
  void ReConstruct(const xJson::Value& bc) override;
  void PrintDebugInfo() const override;

  virtual bool LoadFromDictFile(const std::string& dict_file) {
    return utils::LoadFromDictFile(ctx()->env(), dict_file, min_count_, top_k_,
                                   &indexer_, &keys_);
  }
  virtual void UpdateDefault() {
    if (default_key_ == EXTRA_KEY) {
      indexer_[EXTRA_KEY] = keys_.size();
      keys_.push_back(EXTRA_KEY);
    }
    bool b;
    std::vector<double> v(width());
    ToValue(default_key_, v.begin(), &b);
    set_def(v[0]);
  }

 protected:
  std::unordered_map<std::string, int> indexer_;
  std::vector<std::string> keys_;
  int min_count_;
  int top_k_;
  int oov_buckets_;
  std::string default_key_;
  SCAN_TYPE scan_from_;
  bool reverse_;
};

class StringColumn : public StringBaseColumn {
 public:
  explicit StringColumn(::tensorflow::OpKernelConstruction* ctx)
      : StringBaseColumn(ctx) {}
  StringColumn(::tensorflow::OpKernelConstruction* ctx,
               const std::string& iname, const std::string& alias,
               Field::CID cid, double def, int min_count, int top_k,
               int oov_buckets, const std::string& default_key)
      : StringBaseColumn(ctx, iname, alias, Field::Type::STRING, 1, cid, def,
                         min_count, top_k, oov_buckets, default_key,
                         SCAN_TYPE::HEAD, false) {}
  StringColumn(const StringColumn&) = default;
  StringColumn& operator=(const StringColumn&) = default;

  void ToValue(const std::string&, std::vector<double>::iterator,
               bool*) const override;
};

class StringListColumn : public StringBaseColumn {
 public:
  explicit StringListColumn(::tensorflow::OpKernelConstruction* ctx)
      : StringBaseColumn(ctx) {}
  StringListColumn(::tensorflow::OpKernelConstruction* ctx,
                   const std::string& iname, const std::string& alias,
                   Field::CID cid, double d, int min_count, int top_k, int w,
                   int oov_buckets, const std::string& default_key,
                   SCAN_TYPE st, bool reverse)
      : StringBaseColumn(ctx, iname, alias, Field::Type::STRING_LIST, w, cid, d,
                         min_count, top_k, oov_buckets, default_key, st,
                         reverse) {}
  StringListColumn(const StringListColumn&) = default;
  StringListColumn& operator=(const StringListColumn&) = default;
  void ToValue(const std::string&, std::vector<double>::iterator,
               bool*) const override;
};

class WeightedStringListColumn : public StringBaseColumn {
 public:
  explicit WeightedStringListColumn(::tensorflow::OpKernelConstruction* ctx)
      : StringBaseColumn(ctx) {}
  WeightedStringListColumn(::tensorflow::OpKernelConstruction* ctx,
                           const std::string& iname, const std::string& alias,
                           Field::CID cid, double d, int min_count, int top_k,
                           int w, double min_weight, int oov_buckets,
                           const std::string& default_key, SCAN_TYPE st,
                           bool reverse)
      : StringBaseColumn(ctx, iname, alias, Field::Type::WEIGHTED_STRING_LIST,
                         w * 2, cid, d, min_count, top_k, oov_buckets,
                         default_key, st, reverse),
        min_weight_(min_weight) {}
  WeightedStringListColumn(const WeightedStringListColumn&) = default;
  WeightedStringListColumn& operator=(const WeightedStringListColumn&) =
      default;
  // 前 width 个元素为 id 值，后 width 个元素为 weight 值
  void ToValue(const std::string&, std::vector<double>::iterator,
               bool*) const override;

  void Serialize(xJson::Value* bc) const override;
  void PrintDebugInfo() const override;
  void ReConstruct(const xJson::Value& bc) override;

  double min_weight() const { return min_weight_; }

 private:
  double min_weight_;
};

// 用户视频播放历史这类特征
// 产出正、负播放历史及对应的权重
class TripleListColumn : public StringBaseColumn {
 public:
  explicit TripleListColumn(::tensorflow::OpKernelConstruction* ctx)
      : StringBaseColumn(ctx) {}
  TripleListColumn(::tensorflow::OpKernelConstruction* ctx,
                   const std::string& iname, const std::string& alias,
                   Field::CID cid, double d, int min_count, int top_k,
                   int w_pos, int w_neg, int oov_buckets,
                   const std::string& default_key,
                   const std::string& filter_type, const xJson::Value& params,
                   SCAN_TYPE st, bool reverse)
      : StringBaseColumn(ctx, iname, alias, Field::Type::TRIPLE_LIST,
                         (w_pos + w_neg) * 2, cid, d, min_count, top_k,
                         oov_buckets, default_key, st, reverse),
        width_pos_(w_pos),
        width_neg_(w_neg),
        filter_type_(filter_type),
        params_(params) {
    if (global_filter_map.count(filter_type) == 0) {
      LOG(ERROR) << "Undefined filter_type '" << filter_type << "', "
                 << "global_filter_map.size = " << global_filter_map.size();
      throw std::logic_error("undefined filter_type");
    }
    filter_func_ = global_filter_map[filter_type];
  }

  TripleListColumn(const TripleListColumn&) = default;
  TripleListColumn& operator=(const TripleListColumn&) = default;

  // 返回格式为: width_pos 个正播放ID, w_neg 个负播放ID,
  // width_pos 个正播放ID权重, w_neg 个负播放ID权重
  void ToValue(const std::string&, std::vector<double>::iterator,
               bool*) const override;

  void Serialize(xJson::Value* bc) const override;
  void PrintDebugInfo() const override;
  void ReConstruct(const xJson::Value& bc) override;

  const std::string& filter_type() const { return filter_type_; }
  int width_pos() const { return width_pos_; }
  int width_neg() const { return width_neg_; }

 private:
  int width_pos_;
  int width_neg_;
  std::string filter_type_;
  xJson::Value params_;
  std::function<std::pair<int, double>(StringPiece, StringPiece,
                                       const xJson::Value&)>
      filter_func_;
};

}  // namespace assembler

#endif  // ASSEMBLER_COLUMN_H_
