#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "api.h"
#include "op.h"
#include "cutlass_irrelevant.h"
#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
namespace functor {
template <typename dataTP_ >
struct Fused_Gemm_Bias_Add_Functor<GPUDevice, dataTP_>{
static Status Compute(const GPUDevice&d, int sm, const Param & param) {
    const cudaStream_t &stream = d.stream();
    if (sm == 70) {
        if (param.N <= 32)
            FusedGemmBiasAdd_volta_impl<32>(param.M, param.N, param.K, param.Batch, const_cast<void*>(param.A0), const_cast<void*>(param.B0), const_cast<void*>(param.C0), param.D0, stream);
        else if (param.N > 32 && param.N <= 64)
            FusedGemmBiasAdd_volta_impl<64>(param.M, param.N, param.K, param.Batch, const_cast<void*>(param.A0), const_cast<void*>(param.B0), const_cast<void*>(param.C0), param.D0, stream);
        else if (param.N > 64 )
            FusedGemmBiasAdd_volta_impl<128>(param.M, param.N, param.K, param.Batch, const_cast<void*>(param.A0), const_cast<void*>(param.B0), const_cast<void*>(param.C0), param.D0, stream);
    }
    else if(sm >= 75) {
        if (param.N <= 32)
            FusedGemmBiasAdd_turing_impl<32>(param.M, param.N, param.K, param.Batch, const_cast<void*>(param.A0), const_cast<void*>(param.B0), const_cast<void*>(param.C0), param.D0, stream);
        else if (param.N > 32 && param.N <= 64)
            FusedGemmBiasAdd_turing_impl<64>(param.M, param.N, param.K, param.Batch, const_cast<void*>(param.A0), const_cast<void*>(param.B0), const_cast<void*>(param.C0), param.D0, stream);
        else if (param.N > 64)
            FusedGemmBiasAdd_turing_impl<128>(param.M, param.N, param.K, param.Batch, const_cast<void*>(param.A0), const_cast<void*>(param.B0), const_cast<void*>(param.C0), param.D0, stream);
    }
    else assert(0);
    return Status::OK();
}
}; // struct Fused_Gemm_Bias_Add_Functor
} // namespace functor
template struct functor::Fused_Gemm_Bias_Add_Functor<GPUDevice, Eigen::half>;
} // namespace tensorflow
#endif
