#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
// #include "api.h"
#include "gemm_ln_gemm.h"
#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using gemm_ln_gemm_dtype_0 = ck_tile::fp16_t;

// HeadDimQK <= 32
// 4-wave version
using gemm_ln_gemm_block_tile_0 = ck_tile::sequence<64, 32, 32, 64, 32>;
using gemm_ln_gemm_block_warps_0 = ck_tile::sequence<4, 1, 1>;
using gemm_ln_gemm_warp_tile_0 = ck_tile::sequence<16, 16, 32>;
using gemm_ln_gemm_warp_tile_1 = ck_tile::sequence<16, 16, 16>;

using gemm_ln_gemm_shape_0 = ck_tile::TileGemmLnGemmShape<
    gemm_ln_gemm_block_tile_0, gemm_ln_gemm_block_warps_0,
    gemm_ln_gemm_warp_tile_0, gemm_ln_gemm_block_warps_0,
    gemm_ln_gemm_warp_tile_1>;

using gemm_ln_gemm_trait_0 =
    ck_tile::TileGemmLnGemmTraits<true, true, true, true, -1>;

using gemm_ln_gemm_pipeline_problem_0 = ck_tile::BlockGemmLnGemmPipelineProblem<
    typename GemmLnGemmTypeConfig<gemm_ln_gemm_dtype_0>::QDataType,
    typename GemmLnGemmTypeConfig<gemm_ln_gemm_dtype_0>::WDataType,
    typename GemmLnGemmTypeConfig<gemm_ln_gemm_dtype_0>::S0accDataType,
    typename GemmLnGemmTypeConfig<gemm_ln_gemm_dtype_0>::BiasDataType,
    typename GemmLnGemmTypeConfig<gemm_ln_gemm_dtype_0>::GammaDataType,
    typename GemmLnGemmTypeConfig<gemm_ln_gemm_dtype_0>::BetaDataType,

    typename GemmLnGemmTypeConfig<gemm_ln_gemm_dtype_0>::RDataType,
    typename GemmLnGemmTypeConfig<gemm_ln_gemm_dtype_0>::KDataType,
    typename GemmLnGemmTypeConfig<gemm_ln_gemm_dtype_0>::OaccDataType,
    typename GemmLnGemmTypeConfig<gemm_ln_gemm_dtype_0>::ODataType,
    gemm_ln_gemm_shape_0, gemm_ln_gemm_trait_0>;

using gemm_ln_gemm_pipeline_0 =
    ck_tile::BlockGemmLnGemmPipelineQWKlds<gemm_ln_gemm_pipeline_problem_0>;

using gemm_ln_gemm_epilogue_0 =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
        typename GemmLnGemmTypeConfig<ck_tile::fp16_t>::OaccDataType,
        typename GemmLnGemmTypeConfig<ck_tile::fp16_t>::ODataType, true, true>>;

using gemm_ln_gemm_kernel_0 = ck_tile::GemmLnGemmKernel<
    ck_tile::GemmLnGemmTilePartitioner<gemm_ln_gemm_shape_0>,
    gemm_ln_gemm_pipeline_0, gemm_ln_gemm_epilogue_0>;

using trait_0 =
    gemm_ln_gemm_traits_<16, ck_tile::fp16_t, 64, 32, 32, 64, 32,
                         ck_tile::BlockGemmLnGemmPipelineEnum::QWK_LDS, true,
                         true, true, true>;
using k_ = gemm_ln_gemm_kernel_0;
namespace functor {
template <typename dataTP_>
struct GemmLayernormGemmFunctor<GPUDevice, dataTP_> {
 public:
  static Status Compute(const GPUDevice& d, const void* mat_A0,
                        const void* mat_B0, const void* mat_C,
                        const void* mat_B1, const void* Gamma, const void* Beta,
                        void* mat_D, int K, int M, int N0, int N1, int head_num,
                        float lrelu_alpha, bool do_layer_norm,
                        bool do_leaky_relu) {
    const auto& stream = d.stream();

    ck_tile::stream_config stream_config{stream, false, 0, 20, 50, true};

    auto gemm_ln_gemm_args_device = gemm_ln_gemm_args{mat_A0,
                                                      mat_B0,
                                                      mat_C,
                                                      mat_B1,
                                                      Gamma,
                                                      Beta,
                                                      mat_D,
                                                      K,
                                                      M,
                                                      N0,
                                                      N1,
                                                      head_num,
                                                      lrelu_alpha,
                                                      do_layer_norm,
                                                      do_leaky_relu};

    auto [kargs, grids] =
        gemm_ln_gemm_create_kargs_and_grids<k_>(gemm_ln_gemm_args_device);
    constexpr dim3 blocks = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::launch_kernel(stream_config,
                           ck_tile::make_kernel<blocks.x, kBlockPerCu>(
                               k_{}, grids, blocks, 0, kargs));

    return Status::OK();
  }
};  // struct Fused_Gemm_Bias_Add_Functor
}  // namespace functor
template struct functor::GemmLayernormGemmFunctor<GPUDevice, Eigen::half>;
}  // namespace tensorflow
#endif
