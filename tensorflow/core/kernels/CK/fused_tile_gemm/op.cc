#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
namespace tensorflow {
namespace  {
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename dataTP>
class FusedTileGemmOp : public OpKernel {
public:
  explicit FusedTileGemmOp(OpKernelConstruction* context): OpKernel(context) {}

void Compute(OpKernelContext* context) override {
    std::cout << "COmpute!" <<std::endl;
    const Tensor& a_tensor = context->input(0);
    const Tensor& b_tensor = context->input(1);

    // Rank of input tensors.
    int rank = context->input(0).dims();

    int M = 1, N = 1, K = 1;
    int StrideA = 1, StrideB = 1, StrideC = 1;
    int KBatch = 1;
    int d0 = 1, d1 = 1;
    
    if (rank == 1) {
        M = a_tensor.shape().dim_size(0);
        N = b_tensor.shape().dim_size(0);
    }
    if (rank == 2) {
        M = a_tensor.shape().dim_size(0);
        N = b_tensor.shape().dim_size(1);
        K = a_tensor.shape().dim_size(1);
        StrideA = K;
        StrideB = N;
        StrideC = N;
    } else if(rank == 3) {
        d0 = a_tensor.dim_size(0);
        d1 = a_tensor.dim_size(1);
        M = d0 * d1;
        N = b_tensor.dim_size(2);
        K = a_tensor.dim_size(2);
        StrideA = K;
        StrideB = N;
        StrideC = N;
        KBatch = d0;
    } else {
      context->CtxFailure(errors::InvalidArgument("Unsupported input tensor rank: ", rank));
      return;
    }

    Tensor* output_tensor = nullptr;
    if (rank == 1) {
        OP_REQUIRES_OK(context, context->allocate_output(0, {M}, &output_tensor));
    }
    if (rank == 2) {
        OP_REQUIRES_OK(context, context->allocate_output(0, {M, N}, &output_tensor));
    } else if(rank ==3 ){
        OP_REQUIRES_OK(context, context->allocate_output(0, {d0, d1, N}, &output_tensor));
    }

    Param arguments = {
        M,
        N,
        K,
        KBatch,
        StrideA,
        StrideB,
        StrideC,
        a_tensor.flat<dataTP>().data(),
        b_tensor.flat<dataTP>().data(),
        output_tensor->flat<dataTP>().data()};
    OP_REQUIRES_OK(context, functor::Fused_Tile_Gemm_Functor<Device, dataTP>::Compute(context->eigen_device<Device>(), arguments));
    }
};

#define REGISTER_GPU(dataTP)\
    REGISTER_KERNEL_BUILDER(\
        Name("FusedTileGemm")\
        .Device(DEVICE_GPU)\
        .TypeConstraint<dataTP>("dataTP"),\
    FusedTileGemmOp<GPUDevice, dataTP>)

    // Register for more types.
    REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU
} // namespace 
} // namespace tensorflow
