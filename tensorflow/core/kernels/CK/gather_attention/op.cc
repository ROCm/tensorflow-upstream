#include "op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
namespace tensorflow {
namespace {
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename dataTP>
class GatherAttentionOp : public OpKernel {
 public:
  explicit GatherAttentionOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* matrix_a0;
    OP_REQUIRES_OK(context, context->input("matrix_a0", &matrix_a0));
    const Tensor* matrix_b0;
    OP_REQUIRES_OK(context, context->input("matrix_b0", &matrix_b0));
    const Tensor* matrix_key;
    OP_REQUIRES_OK(context, context->input("matrix_key", &matrix_key));
    const Tensor* indices;
    OP_REQUIRES_OK(context, context->input("indices", &indices));
    const Tensor* matrix_b1;
    OP_REQUIRES_OK(context, context->input("matrix_b1", &matrix_b1));
    int head_sz = matrix_a0->shape().dim_size(2) / head_num_;
    int seq = matrix_b0->shape().dim_size(1);
    int B = matrix_b0->shape().dim_size(0);
    int index = matrix_a0->shape().dim_size(0);

    Tensor* output_tensor = nullptr;

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {index, 1, head_sz * head_num_},
                                            &output_tensor));

    OP_REQUIRES_OK(
        context,
        functor::GatherAttentionFunctor<Device, dataTP>::Compute(
            context->eigen_device<Device>(),
            reinterpret_cast<const void*>(matrix_a0->flat<dataTP>().data()),
            reinterpret_cast<const void*>(matrix_b0->flat<dataTP>().data()),
            reinterpret_cast<const void*>(matrix_key->flat<dataTP>().data()),
            reinterpret_cast<const int*>(indices->flat<int>().data()),
            reinterpret_cast<const void*>(matrix_b1->flat<dataTP>().data()),
            reinterpret_cast<void*>(output_tensor->flat<dataTP>().data()),
            head_sz, seq, B, index, head_num_));
  }

 private:
  int head_num_;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(dataTP)                                     \
  REGISTER_KERNEL_BUILDER(Name("GatherAttention")                \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<dataTP>("dataTP"), \
                          GatherAttentionOp<GPUDevice, dataTP>)

REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU
#endif
}  // namespace
}  // namespace tensorflow
