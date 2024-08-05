#include "op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
namespace tensorflow {
namespace {
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename dataTP>
class GemmLayernormOp : public OpKernel {
 public:
  explicit GemmLayernormOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* matrix_a0;
    OP_REQUIRES_OK(context, context->input("matrix_a0", &matrix_a0));
    const Tensor* matrix_b0;
    OP_REQUIRES_OK(context, context->input("matrix_b0", &matrix_b0));
    const Tensor* matrix_c0;
    OP_REQUIRES_OK(context, context->input("matrix_c0", &matrix_c0));
    const Tensor* layernorm_beta;
    OP_REQUIRES_OK(context, context->input("beta", &layernorm_beta));
    const Tensor* layernorm_gamma;
    OP_REQUIRES_OK(context,
                   context->input("gamma", &layernorm_gamma));

    int d0 = matrix_a0->shape().dim_size(0);
    int d1 = matrix_a0->shape().dim_size(1);
    int N = matrix_b0->shape().dim_size(0);
    int K = matrix_b0->shape().dim_size(1);
    int M = d0 * d1;
    Tensor* output_tensor = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, {d0, d1, N},
                                                     &output_tensor));
    if (output_tensor->NumElements() == 0)
      return;

    OP_REQUIRES_OK(
        context,
        functor::GemmLayernormFunctor<Device, dataTP>::Compute(
            context->eigen_device<Device>(),
            reinterpret_cast<const void*>(matrix_a0->flat<dataTP>().data()),
            reinterpret_cast<const void*>(matrix_b0->flat<dataTP>().data()),
            reinterpret_cast<const void*>(matrix_c0->flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                layernorm_gamma->flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                layernorm_beta->flat<dataTP>().data()),
            reinterpret_cast<void*>(output_tensor->flat<dataTP>().data()), K, M,
            N, head_num_));
  }

 private:
  int head_num_;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(dataTP)                                     \
  REGISTER_KERNEL_BUILDER(Name("GemmLayernorm")              \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<dataTP>("dataTP"), \
                          GemmLayernormOp<GPUDevice, dataTP>)

REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU
#endif
}  // namespace
}  // namespace tensorflow
