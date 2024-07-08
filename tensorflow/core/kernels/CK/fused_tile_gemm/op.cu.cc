#define EIGEN_USE_GPU
#include "api.h"
#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
namespace functor {
template <typename dataTP_ >
struct Fused_Tile_Gemm_Functor<GPUDevice, dataTP_> {
static Status Compute(const GPUDevice& d, const Param & param) {
    std::cout << "op.cu.cc!" <<std::endl;
    const hipStream_t &stream = d.stream();
    FusedTileGemm<256>(param.M, param.N, param.K, param.KBatch,
                        param.StrideA, param.StrideB, param.StrideC,
                        static_cast<const ADataType*>(param.A),
                        static_cast<const ADataType*>(param.B),
                        static_cast<ADataType*>(param.C),
                        stream);
    return Status::OK();
}
}; // struct Fused_Tile_Gemm_Functor
// Explicit instantiation for Eigen::half.
template struct Fused_Tile_Gemm_Functor<GPUDevice, Eigen::half>;
} // namespace functor
} // namespace tensorflow