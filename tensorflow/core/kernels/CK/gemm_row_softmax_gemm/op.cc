#include "op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
namespace tensorflow {
namespace {
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename dataTP>
class GemmRowSoftmaxGemmOp : public OpKernel {
 public:
  explicit GemmRowSoftmaxGemmOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* matrix_a0;
    OP_REQUIRES_OK(context, context->input("matrix_a0", &matrix_a0));
    const Tensor* matrix_b0;
    OP_REQUIRES_OK(context, context->input("matrix_b0", &matrix_b0));
    const Tensor* matrix_a1;
    OP_REQUIRES_OK(context, context->input("matrix_a1", &matrix_a1));
    const Tensor* kmask;
    OP_REQUIRES_OK(context, context->input("kmask", &kmask));

    int batch = matrix_b0->shape().dim_size(0);
    int head_num = matrix_b0->shape().dim_size(1);
    int seq = matrix_b0->shape().dim_size(2);
    int new_head = matrix_a0->shape().dim_size(0);

    Tensor* output_tensor = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, {batch, head_num, seq},
                                                     &output_tensor));
    Tensor matrix_b1;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Eigen::half>::value,
                                          TensorShape({batch, new_head, seq}),
                                          &matrix_b1));
    OP_REQUIRES_OK(
        context,
        functor::GemmRowSoftmaxGemmFunctor<Device, dataTP>::Compute(
            context->eigen_device<Device>(),
            reinterpret_cast<const void*>(matrix_b0->flat<dataTP>().data()),
            reinterpret_cast<const void*>(matrix_a0->flat<dataTP>().data()),
            reinterpret_cast<const void*>(matrix_a1->flat<dataTP>().data()),
            reinterpret_cast<const void*>(kmask->flat<dataTP>().data()),
            reinterpret_cast<void*>(matrix_b1.flat<dataTP>().data()),
            reinterpret_cast<void*>(output_tensor->flat<dataTP>().data()),
            batch, seq, head_num, new_head));
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(dataTP)                                     \
  REGISTER_KERNEL_BUILDER(Name("GemmRowSoftmaxGemm")             \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<dataTP>("dataTP"), \
                          GemmRowSoftmaxGemmOp<GPUDevice, dataTP>)

REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU
#endif
}  // namespace
}  // namespace tensorflow
