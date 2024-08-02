#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include "ck_tile/06_gemm_ln/gemm_ln.hpp"
#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

extern float gemm_ln_fp16(gemm_ln_args& param, ck_tile::stream_config stream);
namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

template <typename T>
Status ComputeInternal(const GPUDevice& d, const void* mat_A0,
                       const void* mat_B0, const void* mat_C,
                       const void* Gamma, const void* Beta,
                       void* mat_D, int K, int M, int N, int head_num) {
using TypeConfig = GemmLnTypeConfig<T>;

using QDataType     = typename TypeConfig::QDataType;
using WDataType     = typename TypeConfig::WDataType;
using S0accDataType = typename TypeConfig::S0accDataType;
using BiasDataType  = typename TypeConfig::BiasDataType;
using GammaDataType = typename TypeConfig::GammaDataType;
using BetaDataType  = typename TypeConfig::BetaDataType;
using RDataType     = typename TypeConfig::RDataType;
using ODataType     = typename TypeConfig::ODataType;

  const hipStream_t stream = d.stream();

  ck_tile::stream_config stream_config{stream, false, 0, 20, 50, true};


    auto gemm_ln_args_device = gemm_ln_args{mat_A0,
                                            mat_B0,
                                            mat_C,
                                            Gamma,
                                            Beta,
                                            mat_D,
                                            K,
                                            M,
                                            N,
                                            head_num,
                                            0.,
                                            false,
                                            false};

    gemm_ln_fp16(gemm_ln_args_device, stream_config);
    return Status::OK();
}
namespace functor {
template <typename T>
struct GemmLayernormGemmFunctor<GPUDevice, T> {
 public:
  static Status Compute(const GPUDevice& d, const void* mat_A0,
                        const void* mat_B0, const void* mat_C,
                        const void* Gamma, const void* Beta,
                        void* mat_D, int K, int M, int N, int head_num) {
    if constexpr (std::is_same_v<T, Eigen::half>) {
      return ComputeInternal<ck_tile::fp16_t>(
          d, mat_A0, mat_B0, mat_C, Gamma, Beta, mat_D, K, M, N, head_num);
    }

    return Status::OK();
  }
};  // struct Fused_Gemm_Bias_Add_Functor
}  // namespace functor
template struct functor::GemmLayernormGemmFunctor<GPUDevice, Eigen::half>;
}  // namespace tensorflow
#endif
