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
    // Get the stamp token.
    const Tensor* matrix_a0;
    OP_REQUIRES_OK(context, context->input("matrix_a0", &matrix_a0));
    const Tensor* matrix_b0;
    OP_REQUIRES_OK(context, context->input("matrix_b0", &matrix_b0));
    const Tensor* matrix_c0;
    OP_REQUIRES_OK(context, context->input("matrix_c0", &matrix_c0));
    const Tensor* layernorm_beta;
    OP_REQUIRES_OK(context, context->input("layernorm_beta", &layernorm_beta));
    const Tensor* layernorm_gamma;
    OP_REQUIRES_OK(context, context->input("layernorm_gamma", &layernorm_gamma));
    const Tensor* matrix_b1;
    OP_REQUIRES_OK(context, context->input("matrix_b1", &matrix_b1));
    const Tensor* softmaxmask;
    OP_REQUIRES_OK(context, context->input("softmaxmask", &softmaxmask));
    const Tensor* matrix_b2;
    OP_REQUIRES_OK(context, context->input("matrix_b2", &matrix_b2));

    int M = matrix_a0->shape().dim_size(0);
    int N0 = matrix_b0->shape().dim_size(0);
    int K = matrix_a0->shape().dim_size(2);
    int B_kv = matrix_b1->shape().dim_size(0);
    int N1 = matrix_b1->shape().dim_size(1);
    int long_seq = softmaxmask->shape().dim_size(1);
    int N2 = matrix_b2->shape().dim_size(2);

    Tensor* output_tensor = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, {B_kv * M, 1, N2},
                                                     &output_tensor));

    OP_REQUIRES_OK(
        context,
        functor::GemmLayernormGemmSoftmaxGemmFunctor<Device, dataTP>::Compute(
            context->eigen_device<Device>(),
            reinterpret_cast<const void*>(
                matrix_a0->flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                matrix_b0->flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                matrix_c0->flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                matrix_b1->flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                matrix_b2->flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                layernorm_gamma->flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                layernorm_beta->flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                softmaxmask->flat<dataTP>().data()),
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
