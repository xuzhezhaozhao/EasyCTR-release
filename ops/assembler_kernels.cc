#include <time.h>
#include <memory>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "assembler/assembler.h"
#include "assembler/utils.h"
#include "deps/jsoncpp/json/json.h"

using assembler::Assembler;
using assembler::Example;
using assembler::Feature;

namespace tensorflow {
class AssemblerOp : public OpKernel {
 public:
  explicit AssemblerOp(OpKernelConstruction* ctx) : OpKernel(ctx), cnt_(0) {
    LOG(INFO) << " ------------------------ ";
    LOG(INFO) << "Init AssemblerOp ...";
    std::string conf_path;
    bool use_lua_sampler = false;
    std::string lua_sampler_script;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("conf_path", &conf_path));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_lua_sampler", &use_lua_sampler));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("lua_sampler_script", &lua_sampler_script));
    assembler_.reset(new Assembler(ctx));
    assembler_->Init(conf_path, use_lua_sampler, lua_sampler_script, true,
                     true);
    LOG(INFO) << "Init AssemblerOp OK.";
    LOG(INFO) << " ------------------------ ";
  }

  void Compute(OpKernelContext* ctx) override {
    auto flat_input = ctx->input(0).flat<std::string>();
    OP_REQUIRES(ctx, flat_input.size() == 1,
                errors::InvalidArgument("input size is not 1"));
    auto is_predict_tensor = ctx->input(1).flat<bool>();
    OP_REQUIRES(ctx, is_predict_tensor.size() == 1,
                errors::InvalidArgument("is_predict tensor size is not 1"));
    bool is_predict = is_predict_tensor(0);
    const std::string& input = flat_input(0);
    int dim = assembler_->feature_size();
    Example example(Feature(dim), 0.0, 0.0);
    int total = 1;
    if (!assembler_->GetExample(input, is_predict, &example)) {
      total = 0;
    }
    if (example.weight <= 0.0) {
      total = 0;
    }

    if (total == 1) {
      if (example.label == 1) {
      } else if (example.label == 0) {
      }
    }

    const int every_step = 10000000;
    if (cnt_ % every_step == 0) {
      mutex_lock l(mu_);
      if (cnt_ % every_step == 0) {
        LOG(INFO) << "input: ";
        LOG(INFO) << flat_input(0);
        assembler_->PrintExample(example);
      }
      ++cnt_;
    } else {
      ++cnt_;  // do not use mutex for performence
    }
    // Create output tensors
    Tensor* feature_tensor = nullptr;
    Tensor* label_tensor = nullptr;
    Tensor* weight_tensor = nullptr;
    TensorShape feature_shape;
    feature_shape.AddDim(total);
    feature_shape.AddDim(dim);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, feature_shape, &feature_tensor));

    TensorShape label_shape;
    label_shape.AddDim(total);
    label_shape.AddDim(1);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, label_shape, &label_tensor));

    TensorShape weight_shape;
    weight_shape.AddDim(total);
    weight_shape.AddDim(1);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, weight_shape, &weight_tensor));

    if (total > 0) {
      // Fill output tensors
      auto flat_feature = feature_tensor->flat<double>();
      auto flat_label = label_tensor->flat<double>();
      auto flat_weight = weight_tensor->flat<double>();
      flat_label(0) = example.label;
      flat_weight(0) = example.weight;
      for (size_t fi = 0; fi < example.feature.size(); ++fi) {
        flat_feature(fi) = example.feature[fi];
      }
    }
  }

 private:
  std::shared_ptr<Assembler> assembler_;
  size_t cnt_;

  mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("Assembler").Device(DEVICE_CPU), AssemblerOp);

class AssemblerSchemeOp : public OpKernel {
 public:
  explicit AssemblerSchemeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    LOG(INFO) << "Init AssemblerSchemeOp ...";
    std::string conf_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("conf_path", &conf_path));
    assembler_.reset(new Assembler(ctx));
    assembler_->Init(conf_path);
    scheme_ = assembler_->GetFeatureScheme().toStyledString();
    LOG(INFO) << "Init AssemblerSchemeOp done";
  }

  void Compute(OpKernelContext* ctx) override {
    // Create output tensors
    Tensor* tensor = nullptr;
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &tensor));
    auto scalar = tensor->scalar<std::string>();
    scalar(0) = scheme_;
  }

 private:
  std::shared_ptr<Assembler> assembler_;
  std::string scheme_;
};

REGISTER_KERNEL_BUILDER(Name("AssemblerScheme").Device(DEVICE_CPU),
                        AssemblerSchemeOp);

class AssemblerSerializeOp : public OpKernel {
 public:
  explicit AssemblerSerializeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    LOG(INFO) << "\n\n";
    LOG(INFO) << "Init AssemblerSerializeOp ...";
    std::string conf_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("conf_path", &conf_path));
    assembler_.reset(new Assembler(ctx));
    assembler_->Init(conf_path);
    assembler_->Serialize(&serialized_);
    LOG(INFO) << "Init AssemblerSerializeOp done";
    LOG(INFO) << "\n\n";
  }

  void Compute(OpKernelContext* ctx) override {
    // Create output tensors
    Tensor* tensor = nullptr;
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &tensor));
    auto scalar = tensor->scalar<std::string>();
    scalar(0) = serialized_;
  }

 private:
  std::shared_ptr<Assembler> assembler_;
  std::string serialized_;
};

REGISTER_KERNEL_BUILDER(Name("AssemblerSerialize").Device(DEVICE_CPU),
                        AssemblerSerializeOp);

class AssemblerServingOp : public OpKernel {
 public:
  explicit AssemblerServingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    model_uptime_ = time(NULL);
    LOG(INFO) << "\n\n";
    LOG(INFO) << " ------------------------ ";
    LOG(INFO) << "Init AssemblerServingOp ...";
    std::string serialized;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("serialized", &serialized));
    assembler_.reset(new Assembler(ctx));
    OP_REQUIRES(ctx, assembler_->ParseFromString(serialized),
                errors::Internal("assembler::ParseFromString error"));
    LOG(INFO) << "Assembler columns size = " << assembler_->columns().size();
    assembler_->PrintDebugInfo();
    LOG(INFO) << "Init AssemblerServingOp done";
    LOG(INFO) << " ------------------------ ";
  }

  void Compute(OpKernelContext* ctx) override {
    time_t now = time(NULL);
    if (now - model_uptime_ > 120 * 60 * 60) {
    } else if (now - model_uptime_ > 96 * 60 * 60) {
    } else if (now - model_uptime_ > 72 * 60 * 60) {
    } else if (now - model_uptime_ > 48 * 60 * 60) {
    } else if (now - model_uptime_ > 24 * 60 * 60) {
    } else if (now - model_uptime_ > 12 * 60 * 60) {
    } else if (now - model_uptime_ > 6 * 60 * 60) {
    } else {
    }

    const Tensor& user_feature_tensor = ctx->input(0);
    const Tensor& item_feature_tensor = ctx->input(1);
    std::string user_feature = user_feature_tensor.flat<string>()(0);
    std::vector<std::string> items;
    auto flat_items = item_feature_tensor.flat<std::string>();
    for (int i = 0; i < flat_items.size(); ++i) {
      items.push_back(flat_items(i));
    }

    std::vector<std::vector<double>> features;
    bool is_recall = false;  // 该 op 为 ctr 模式
    assembler_->GetServingInputs(user_feature, items, is_recall, &features);
    // Create output tensors
    Tensor* output = NULL;
    size_t sz = assembler_->feature_size();
    TensorShape shape;
    shape.AddDim(features.size());
    shape.AddDim(sz);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    auto matrix_output = output->matrix<double>();
    for (size_t i = 0; i < features.size(); ++i) {
      for (size_t j = 0; j < features[i].size(); ++j) {
        OP_REQUIRES(
            ctx, sz == features[i].size(),
            errors::Internal("Internal error, features.size not matched"));
        matrix_output(i, j) = features[i][j];
      }
    }
  }

 private:
  std::shared_ptr<Assembler> assembler_;
  time_t model_uptime_;
};

REGISTER_KERNEL_BUILDER(Name("AssemblerServing").Device(DEVICE_CPU),
                        AssemblerServingOp);

class AssemblerDssmServingOp : public OpKernel {
 public:
  explicit AssemblerDssmServingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    model_uptime_ = time(NULL);
    LOG(INFO) << "\n\n";
    LOG(INFO) << " ------------------------ ";
    LOG(INFO) << "Init AssemblerDssmServingOp ...";
    std::string serialized;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("serialized", &serialized));
    assembler_.reset(new Assembler(ctx));
    OP_REQUIRES(ctx, assembler_->ParseFromString(serialized),
                errors::Internal("assembler::ParseFromString error"));
    LOG(INFO) << "Assembler columns size = " << assembler_->columns().size();
    assembler_->PrintDebugInfo();
    LOG(INFO) << "Init AssemblerDssmServingOp done";
    LOG(INFO) << " ------------------------ ";
  }

  void Compute(OpKernelContext* ctx) override {
    time_t now = time(NULL);
    if (now - model_uptime_ > 120 * 60 * 60) {
    } else if (now - model_uptime_ > 96 * 60 * 60) {
    } else if (now - model_uptime_ > 72 * 60 * 60) {
    } else if (now - model_uptime_ > 48 * 60 * 60) {
    } else if (now - model_uptime_ > 24 * 60 * 60) {
    } else if (now - model_uptime_ > 12 * 60 * 60) {
    } else if (now - model_uptime_ > 6 * 60 * 60) {
    } else {
    }

    const Tensor& user_feature_tensor = ctx->input(0);
    std::string user_feature = user_feature_tensor.flat<string>()(0);
    std::vector<std::vector<double>> features;
    bool is_recall = true;           // dssm recall 模式
    std::vector<std::string> items;  // recall 模式下为空
    assembler_->GetServingInputs(user_feature, items, is_recall, &features);
    // Create output tensors
    Tensor* output = NULL;
    size_t sz = assembler_->feature_size();
    TensorShape shape;
    shape.AddDim(features.size());
    shape.AddDim(sz);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    auto matrix_output = output->matrix<double>();
    for (size_t i = 0; i < features.size(); ++i) {
      for (size_t j = 0; j < features[i].size(); ++j) {
        OP_REQUIRES(
            ctx, sz == features[i].size(),
            errors::Internal("Internal error, features.size not matched"));
        matrix_output(i, j) = features[i][j];
      }
    }
  }

 private:
  std::shared_ptr<Assembler> assembler_;
  time_t model_uptime_;
};

REGISTER_KERNEL_BUILDER(Name("AssemblerDssmServing").Device(DEVICE_CPU),
                        AssemblerDssmServingOp);

class AssemblerWithNegativeSamplingOp : public OpKernel {
 public:
  explicit AssemblerWithNegativeSamplingOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), cnt_(0) {
    LOG(INFO) << " ------------------------ ";
    LOG(INFO) << "Init AssemblerWithNegativeSamplingOp ...";
    std::string conf_path;
    std::string nce_items_path;
    int nce_items_min_count, nce_items_top_k;
    bool use_lua_sampler = false;
    std::string lua_sampler_script;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("conf_path", &conf_path));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("nce_items_path", &nce_items_path));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("nce_count", &nce_count_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("nce_items_min_count", &nce_items_min_count));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("nce_items_top_k", &nce_items_top_k));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_lua_sampler", &use_lua_sampler));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("lua_sampler_script", &lua_sampler_script));
    assembler_.reset(new Assembler(ctx));
    assembler_->Init(conf_path, use_lua_sampler, lua_sampler_script, true,
                     true);
    int seed = 1234;  // TODO(zhezhaoxu) add seed
    assembler_->InitNegativeSampling(nce_items_path, nce_items_min_count,
                                     nce_items_top_k, seed);
    LOG(INFO) << "Init AssemblerWithNegativeSamplingOp OK.";
    LOG(INFO) << " ------------------------ ";
  }

  void Compute(OpKernelContext* ctx) override {
    auto flat_input = ctx->input(0).flat<std::string>();
    OP_REQUIRES(ctx, flat_input.size() == 1,
                errors::InvalidArgument("input size is not 1"));
    const std::string& input = flat_input(0);
    int dim = assembler_->feature_size();
    size_t total = 1 + nce_count_;
    std::vector<Example> examples(total, Example(Feature(dim), 0.0, 0.0));
    if (!assembler_->GetExamplesWithNegativeSampling(input, &examples)) {
      total = 0;
    }
    if (total > 0 && examples[0].label == 0) {
      // Not do negative sampling for real negative sample
      total = 1;
    }
    if (total > 0 && (examples[0].label < 0 || examples[0].weight <= 0)) {
      total = 0;
    }
    if (total > 0) {
      if (examples[0].label == 1) {
      } else if (examples[0].label == 0) {
      }
    }

    const int every_step = 10000000;
    if (cnt_ % every_step == 0) {
      mutex_lock l(mu_);
      if (cnt_ % every_step == 0) {
        assembler_->PrintExample(examples[0]);
        if (nce_count_ > 0) {
          LOG(ERROR) << " --- negative sampling --- ";
          assembler_->PrintExample(examples[1]);
        }
      }
      ++cnt_;
    } else {
      ++cnt_;  // do not use mutex for performence
    }
    // Create output tensors
    Tensor* feature_tensor = nullptr;
    Tensor* label_tensor = nullptr;
    Tensor* weight_tensor = nullptr;
    TensorShape feature_shape;
    feature_shape.AddDim(total);
    feature_shape.AddDim(dim);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, feature_shape, &feature_tensor));

    TensorShape label_shape;
    label_shape.AddDim(total);
    label_shape.AddDim(1);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, label_shape, &label_tensor));

    TensorShape weight_shape;
    weight_shape.AddDim(total);
    weight_shape.AddDim(1);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, weight_shape, &weight_tensor));

    // Fill output tensors
    auto flat_feature = feature_tensor->flat<double>();
    auto flat_label = label_tensor->flat<double>();
    auto flat_weight = weight_tensor->flat<double>();

    int idx = 0;
    for (size_t i = 0; i < total; ++i) {
      flat_label(i) = examples[i].label;
      flat_weight(i) = examples[i].weight;
      for (int j = 0; j < dim; ++j) {
        flat_feature(idx) = examples[i].feature[j];
        ++idx;
      }
    }
  }

 private:
  std::shared_ptr<assembler::Assembler> assembler_;
  size_t cnt_;
  int nce_count_;

  mutex mu_;
};

REGISTER_KERNEL_BUILDER(
    Name("AssemblerWithNegativeSampling").Device(DEVICE_CPU),
    AssemblerWithNegativeSamplingOp);

}  // namespace tensorflow
