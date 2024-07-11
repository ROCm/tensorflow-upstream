#include "op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
namespace tensorflow {
namespace {
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename dataTP>
class FusedGemmBiasAddOp : public OpKernel {
 public:
  explicit FusedGemmBiasAddOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    int rank = context->input(0).dims();

    int M = context->input(0).shape().dim_size(0);
    int N = context->input(1).shape().dim_size(0);
    int K = context->input(0).shape().dim_size(1);

    Tensor* output_tensor = nullptr;

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {M, N}, &output_tensor));
    Eigen::array<int, 2> bcast({M, N});
    Tensor bias_tmp;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Eigen::half>::value,
                                          TensorShape({M, N}), &bias_tmp));
    bias_tmp.flat<dataTP>() = context->input(2).flat<dataTP>().broadcast(bcast);
    OP_REQUIRES_OK(
        context,
        functor::FusedGemmBiasAddFunctor<Device, dataTP>::Compute(
            context->eigen_device<Device>(), M, N, K,
            reinterpret_cast<const void*>(
                context->input(0).flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                context->input(1).flat<dataTP>().data()),
            reinterpret_cast<const void*>(bias_tmp.flat<dataTP>().data()),
            reinterpret_cast<void*>(output_tensor->flat<dataTP>().data())));
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(dataTP)                                     \
  REGISTER_KERNEL_BUILDER(Name("FusedGemmBiasAdd")               \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<dataTP>("dataTP"), \
                          FusedGemmBiasAddOp<GPUDevice, dataTP>)

REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU
#endif
}  // namespace
}  // namespace tensorflow
