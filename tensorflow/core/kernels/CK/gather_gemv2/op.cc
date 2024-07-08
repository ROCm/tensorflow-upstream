#include "op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
namespace tensorflow {
namespace {
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename dataTP>
class GatherGemv2Op : public OpKernel {
 public:
  explicit GatherGemv2Op(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
  }

  void Compute(OpKernelContext* context) override {
    int head_sz = context->input(1).shape().dim_size(2) / head_num_;
    int seq = context->input(1).shape().dim_size(1);
    int B = context->input(1).shape().dim_size(0);
    int index = context->input(0).shape().dim_size(0);

    Tensor* output_tensor = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, {index, 1, head_num_*head_sz},
                                                     &output_tensor));

    OP_REQUIRES_OK(
        context,
        functor::GatherGemv2Functor<Device, dataTP>::Compute(
            context->eigen_device<Device>(),
            reinterpret_cast<const void*>(
                context->input(0).flat<dataTP>().data()),
            reinterpret_cast<const void*>(
                context->input(1).flat<dataTP>().data()),
            reinterpret_cast<const int*>(
                context->input(2).flat<int>().data()),
            reinterpret_cast<void*>(output_tensor->flat<dataTP>().data()),
            head_sz, seq, B, index, head_num_));
  }

 private:
  int head_num_;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(dataTP)                                                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("GatherGemv2").Device(DEVICE_GPU).TypeConstraint<dataTP>("dataTP"), \
      GatherGemv2Op<GPUDevice, dataTP>)

REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU
#endif
}  // namespace
}  // namespace tensorflow
