#include "op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
namespace tensorflow {
namespace {
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename dataTP>
class GemmLayernormGemmSoftmaxGemmOp : public OpKernel {
 public:
  explicit GemmLayernormGemmSoftmaxGemmOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
    OP_REQUIRES_OK(context, context->GetAttr("lrelu_alpha", &lrelu_alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("do_layer_norm", &do_layer_norm_));
    OP_REQUIRES_OK(context, context->GetAttr("do_leaky_relu", &do_leaky_relu_));
    OP_REQUIRES_OK(context, context->GetAttr("do_query_mask", &do_query_mask_));
  }

  void Compute(OpKernelContext* context) override {
    int M = context->input(0).shape().dim_size(0);
    int N0 = context->input(1).shape().dim_size(0);
    int K = context->input(0).shape().dim_size(2);
    int B_kv = context->input(5).shape().dim_size(0);
    int N1 = context->input(5).shape().dim_size(1);
    int long_seq = context->input(6).shape().dim_size(1);
    int N2 = context->input(7).shape().dim_size(2);

    Tensor* output_tensor = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, {B_kv * M, 1, N2},
                                                     &output_tensor));

    OP_REQUIRES_OK(
        context,
        functor::GemmLayernormGemmSoftmaxGemmFunctor<Device, dataTP>::Compute(
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
            reinterpret_cast<const void*>(
                context->input(6).flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                context->input(7).flat<dataTP>().data()),
            reinterpret_cast<void*>(output_tensor->flat<dataTP>().data()), K, M,
            N0, N1, long_seq, N2, B_kv, head_num_, lrelu_alpha_, do_layer_norm_,
            do_leaky_relu_, do_query_mask_));
  }

 private:
  int head_num_;
  float lrelu_alpha_;
  bool do_layer_norm_;
  bool do_leaky_relu_;
  bool do_query_mask_;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(dataTP)                                     \
  REGISTER_KERNEL_BUILDER(Name("GemmLayernormGemmSoftmaxGemm")   \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<dataTP>("dataTP"), \
                          GemmLayernormGemmSoftmaxGemmOp<GPUDevice, dataTP>)

REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU
#endif
}  // namespace
}  // namespace tensorflow
