#include "op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
namespace tensorflow {
namespace {
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename dataTP>
class FusedTileGemmOp : public OpKernel {
 public:
  explicit FusedTileGemmOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
  }

  void Compute(OpKernelContext* context) override {
    int rank = context->input(0).dims();

    const int batch = context->input(0).shape().dim_size(0);
    const int head_sz = context->input(1).shape().dim_size(2) / head_num_;
    const int seq = context->input(0).shape().dim_size(2);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, {batch, 1, context->input(1).shape().dim_size(2)},
                       &output_tensor));

    OP_REQUIRES_OK(
        context,
        functor::FusedTileGemmFunctor<Device, dataTP>::Compute(
            context->eigen_device<Device>(),
            reinterpret_cast<const void*>(
                context->input(0).flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                context->input(1).flat<dataTP>().data()),
            reinterpret_cast<void*>(output_tensor->flat<dataTP>().data()),
            batch, seq, head_sz, head_num_));
  }

 private:
  int head_num_;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(dataTP)                                     \
  REGISTER_KERNEL_BUILDER(Name("FusedTileGemm")                  \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<dataTP>("dataTP"), \
                          FusedTileGemmOp<GPUDevice, dataTP>)

REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU
#endif
}  // namespace
}  // namespace tensorflow
