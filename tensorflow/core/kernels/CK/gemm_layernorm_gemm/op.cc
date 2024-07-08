#include "op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
namespace tensorflow {
namespace {
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename dataTP>
class GemmLayernormGemmOp : public OpKernel {
 public:
  explicit GemmLayernormGemmOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
    OP_REQUIRES_OK(context, context->GetAttr("lrelu_alpha", &lrelu_alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("do_layer_norm", &do_layer_norm_));
    OP_REQUIRES_OK(context, context->GetAttr("do_leaky_relu", &do_leaky_relu_));
  }

  void Compute(OpKernelContext* context) override {
    int M = context->input(0).shape().dim_size(0);
    int N0 = context->input(1).shape().dim_size(0);
    int K = context->input(0).shape().dim_size(2);
    int N1 = context->input(5).shape().dim_size(1);

    Tensor* output_tensor = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, {M, head_num_, N1},
                                                     &output_tensor));

    OP_REQUIRES_OK(
        context,
        functor::GemmLayernormGemmFunctor<Device, dataTP>::Compute(
            context->eigen_device<Device>(),
            reinterpret_cast<const void*>(
                context->input(0).flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                context->input(1).flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                context->input(2).flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                context->input(3).flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                context->input(4).flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                context->input(5).flat<dataTP>().data()),
            reinterpret_cast<void*>(output_tensor->flat<dataTP>().data()), K, M,
            N0, N1, head_num_, lrelu_alpha_, do_layer_norm_, do_leaky_relu_));
  }

 private:
  int head_num_;
  float lrelu_alpha_;
  bool do_layer_norm_;
  bool do_leaky_relu_;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(dataTP)                                     \
  REGISTER_KERNEL_BUILDER(Name("GemmLayernormGemm")              \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<dataTP>("dataTP"), \
                          GemmLayernormGemmOp<GPUDevice, dataTP>)

REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU
#endif
}  // namespace
}  // namespace tensorflow
