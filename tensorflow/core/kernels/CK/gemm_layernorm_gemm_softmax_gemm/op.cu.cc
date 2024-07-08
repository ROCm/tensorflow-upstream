#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
// #include "api.h"
#include "gemm_ln_attn.h"
#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
namespace tensorflow {
    using GPUDevice = Eigen::GpuDevice;
using gemm_ln_attn_dtype_0 = ck_tile::fp16_t;

// HeadDimQK <= 16
// HeadDimV  <= 32
// 4-wave version
using gemm_ln_attn_block_tile_0 = ck_tile::sequence<64, 16,
                                                    64,  // Gemm0 M, N, K
                                                    64,
                                                    16,  // Gemm1  , N, K
                                                    32,
                                                    64>;  // Gemm2  , N, K
using gemm_ln_attn_block_warps_0 = ck_tile::sequence<4, 1, 1>;
using gemm_ln_attn_warp_tile_0 = ck_tile::sequence<16, 16, 32>;
using gemm_ln_attn_warp_tile_1 = ck_tile::sequence<16, 16, 16>;

using gemm_ln_attn_shape_0 = ck_tile::TileGemmLnAttnShape<
    gemm_ln_attn_block_tile_0, gemm_ln_attn_block_warps_0,
    gemm_ln_attn_warp_tile_0, gemm_ln_attn_block_warps_0,
    gemm_ln_attn_warp_tile_1, gemm_ln_attn_block_warps_0,
    gemm_ln_attn_warp_tile_1>;
using gemm_ln_attn_trait_0 =
    ck_tile::TileGemmLnAttnTraits<true, true, true, true, true, -1>;

using gemm_ln_attn_pipeline_problem_0 = ck_tile::BlockGemmLnAttnPipelineProblem<
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::QDataType,
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::WDataType,
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::S0accDataType,
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::BiasDataType,
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::GammaDataType,
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::BetaDataType,

    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::RDataType,
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::KDataType,
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::S1accDataType,
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::KeyMaskDataType,
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::SMPLComputeDataType,

    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::PDataType,
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::VDataType,
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::OaccDataType,
    typename GemmLnAttnTypeConfig<gemm_ln_attn_dtype_0>::ODataType,
    gemm_ln_attn_shape_0, gemm_ln_attn_trait_0>;

using gemm_ln_attn_pipeline_0 =
    ck_tile::BlockGemmLnAttnPipelineQWKVlds<gemm_ln_attn_pipeline_problem_0>;

using gemm_ln_attn_epilogue_0 =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
        typename GemmLnAttnTypeConfig<ck_tile::fp16_t>::OaccDataType,
        typename GemmLnAttnTypeConfig<ck_tile::fp16_t>::ODataType, true, true>>;

using gemm_ln_attn_kernel_0 = ck_tile::GemmLnAttnKernel<
    ck_tile::GemmLnAttnTilePartitioner<gemm_ln_attn_shape_0>,
    gemm_ln_attn_pipeline_0, gemm_ln_attn_epilogue_0>;

using trait_0 =
    gemm_ln_attn_traits_<16, 32, ck_tile::fp16_t, 64, 16, 64, 64, 16, 32, 64,
                         ck_tile::BlockGemmLnAttnPipelineEnum::QWKV_LDS, true,
                         true, true, true, true>;
using k_ = gemm_ln_attn_kernel_0;

namespace functor {
template <typename dataTP_>
struct GemmLayernormGemmSoftmaxGemmFunctor<GPUDevice, dataTP_> {
 public:
  static Status Compute(const GPUDevice& d, const void* mat_A0,
                        const void* mat_B0, const void* mat_C,
                        const void* mat_B1, const void* mat_B2,
                        const void* Gamma, const void* Beta,
                        const void* Keymask, void* mat_D, int K, int M, int N0,
                        int N1, int long_seq, int N2, int B_kv, int head_num,
                        float lrelu_alpha, bool do_layer_norm,
                        bool do_leaky_relu, bool do_query_mask) {
    const bool time_kernel = std::getenv("TF_CK_TIME_KERNEL") != nullptr;
    const auto& stream = d.stream();

    ck_tile::stream_config stream_config{stream, time_kernel, 0, 20, 50, true};

    auto gemm_ln_attn_args_device = gemm_ln_attn_args{mat_A0,
                                                      mat_B0,
                                                      mat_C,
                                                      mat_B1,
                                                      mat_B2,
                                                      Gamma,
                                                      Beta,
                                                      Keymask,
                                                      mat_D,
                                                      K,
                                                      M,
                                                      N0,
                                                      long_seq,
                                                      N2,
                                                      B_kv,
                                                      head_num,
                                                      lrelu_alpha,
                                                      do_layer_norm,
                                                      do_leaky_relu,
                                                      do_query_mask};

    auto [kargs, grids] = gemm_ln_attn_create_kargs_and_grids<k_>(gemm_ln_attn_args_device);
    constexpr dim3 blocks = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    float ave_time = ck_tile::launch_kernel(
        stream_config, ck_tile::make_kernel<blocks.x, kBlockPerCu>(
                           k_{}, grids, blocks, 0, kargs));
    if (time_kernel) {
    //   std::size_t flop = std::size_t(2) * M * N * K;
    //   std::size_t num_btype = sizeof(A0DataType) * M * K +
    //                           sizeof(B0DataType) * K * N +
    //                           sizeof(EDataType) * M * N;

    //   float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    //   float gb_per_sec = num_btype / 1.E6 / ave_time;
    //   LOG(INFO) << "Running time: " << ave_time << " ms, " << tflops
    //             << " TFlops, " << gb_per_sec << " GB/s";
    }
    return Status::OK();
  }
};  // struct Fused_Gemm_Bias_Add_Functor
}  // namespace functor
template struct functor::GemmLayernormGemmSoftmaxGemmFunctor<GPUDevice,
                                                             Eigen::half>;
}  // namespace tensorflow
#endif
