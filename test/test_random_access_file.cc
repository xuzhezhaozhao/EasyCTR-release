
#include <iostream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

REGISTER_OP("TestRandomAccessFile").Attr("filename: string = ''").Doc(R"doc(
)doc");

class TestRandomAccessFileOp : public OpKernel {
 public:
  explicit TestRandomAccessFileOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename_));
    std::cout << "filename = " << filename_ << std::endl;
    OP_REQUIRES_OK(ctx, ctx->env()->NewRandomAccessFile(filename_, &file_));
  }

  void Compute(OpKernelContext* ctx) override {
    StringPiece result;
    char buf[40965];
    file_->Read(0, 4096, &result, buf);
    std::cout << "read size = " << result.size() << std::endl;
    std::cout << "begin ..." << std::endl;
    std::cout << std::string(result) << std::endl;
    std::cout << "end" << std::endl;
  }

 private:
  std::string filename_;
  std::unique_ptr<RandomAccessFile> file_;
};

REGISTER_KERNEL_BUILDER(Name("TestRandomAccessFile").Device(DEVICE_CPU),
                        TestRandomAccessFileOp);

}  // namespace tensorflow
