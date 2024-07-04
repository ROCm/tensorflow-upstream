#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
namespace tensorflow {
namespace  {
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename dataTP>
class FusedGemmBiasAddOp : public OpKernel {
public:
  explicit FusedGemmBiasAddOp(OpKernelConstruction* context): OpKernel(context) {
}

void Compute(OpKernelContext* context) override {
    int rank = context->input(0).dims();
    int d0 = 1;
    int d1 = 1;
    int M = 1;
    int N = 1;
    int K = 1;
    if (rank == 2) {
         M = context->input(0).shape().dim_size(0);
         N = context->input(1).shape().dim_size(0);
         K = context->input(0).shape().dim_size(1);
    } else if(rank == 3) {
         d0 = context->input(0).shape().dim_size(0);
         d1 = context->input(0).shape().dim_size(1);
         M = d0 * d1;
         N = context->input(1).shape().dim_size(0);
         K = context->input(0).shape().dim_size(2);
    }
    Tensor* output_tensor = nullptr;
    if (rank == 2) {
        OP_REQUIRES_OK(context, context->allocate_output(0, {M, N}, &output_tensor));
    } else if(rank ==3 ){
        OP_REQUIRES_OK(context, context->allocate_output(0, {d0, d1, N}, &output_tensor));
    }
    OP_REQUIRES_OK(context, functor::Fused_Gemm_Bias_Add_Functor<Device, dataTP>::Compute(context->eigen_device<Device>(), M, N, K, 1, reinterpret_cast<const void*>(context->input(0).flat<dataTP>().data()), reinterpret_cast<const void*>(context->input(1).flat<dataTP>().data()), reinterpret_cast<const void*>(context->input(2).flat<dataTP>().data()), reinterpret_cast<void*>(output_tensor->flat<dataTP>().data())));
    }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(dataTP)\
    REGISTER_KERNEL_BUILDER(\
        Name("FusedGemmBiasAdd")\
        .Device(DEVICE_GPU)\
        .TypeConstraint<dataTP>("dataTP"),\
    FusedGemmBiasAddOp<GPUDevice, dataTP>)

    REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU
#endif
} // namespace 
} // namespace tensorflow

