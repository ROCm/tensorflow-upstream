#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#include "fmha_fwd.h"
#include "mask.h"
#include "utils.h"
#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using TypeConfig = FmhaFwdTypeConfig<ck_tile::half_t>;

using QDataType             = typename TypeConfig::QDataType;
using KDataType             = typename TypeConfig::KDataType;
using VDataType             = typename TypeConfig::VDataType;
using BiasDataType          = typename TypeConfig::BiasDataType;
using RandValOutputDataType = typename TypeConfig::RandValOutputDataType;
using LSEDataType           = typename TypeConfig::LSEDataType;
using SaccDataType          = typename TypeConfig::SaccDataType;
using SMPLComputeDataType   = typename TypeConfig::SMPLComputeDataType;
using PDataType             = typename TypeConfig::PDataType;
using OaccDataType          = typename TypeConfig::OaccDataType;
using ODataType             = typename TypeConfig::ODataType;

namespace functor {
template <typename dataTP_>
struct GemvSoftmaxGemvFunctor<GPUDevice, dataTP_> {
 public:
  static Status Compute(const GPUDevice& d, const void* mat_A,
                              const void* mat_B0,
                              const void* keymask,
                              const void* mat_B1,
                              void* mat_D,
                              int head_sz,
                              int seq,
                              int batch) {
    const bool time_kernel = std::getenv("TF_CK_TIME_KERNEL") != nullptr;
    const auto& stream = d.stream();
    // const auto mask = ck_tile::GenericAttentionMask<false>{seq, seq};
    ck_tile::stream_config stream_config{stream, time_kernel, 0, 20, 50, true};
    const bool is_v_rowmajor = true;
    const bool i_perm = true;
    const bool o_perm = true;
    const int nhead = 1;
    const int nhead_k = 1;
    const int hdim_q = head_sz;
    const int hdim_K = head_sz;
    const int hdim_v = head_sz;
    const int max_seqlen_q = seq;
    const int max_seqlen_k = seq;
    const int shape_seqlen_k = seq;
    const int shape_seqlen_q = seq;
    auto fmha_traits = fmha_fwd_traits{head_sz,
                                       head_sz,
                                       "fp16",
                                       false,
                                       is_v_rowmajor,
                                       mask_enum::no_mask,
                                       bias_enum::elementwise_bias,
                                       false,
                                       false,
                                       false};

    auto fmha_args = [&]() {

        /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
        ///       seqlen_k] in this example, hence both the 'batch_stride_bias' &
        ///       'nhead_stride_bias' are 0.
        // setup stride_* arguments
        const ck_tile::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_k = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_v = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return i_perm ? shape_seqlen_k : nhead_k * shape_seqlen_k;
        }();
        const ck_tile::index_t stride_bias    = (i_perm ? shape_seqlen_k : 1 * shape_seqlen_k);
        const ck_tile::index_t stride_randval = (max_seqlen_k);
        const ck_tile::index_t stride_o_acc   = hdim_v;
        const ck_tile::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k = (i_perm ? shape_seqlen_k * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v = [&]() {
            if(is_v_rowmajor)
                return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
        }();
        const ck_tile::index_t nhead_stride_bias =
            (i_perm ? 0 * shape_seqlen_q * shape_seqlen_k : 0 * shape_seqlen_k);
        const ck_tile::index_t nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t nhead_stride_lse     = max_seqlen_q;
        const ck_tile::index_t nhead_stride_lse_acc = max_seqlen_q;
        const ck_tile::index_t nhead_stride_o_acc   = (max_seqlen_q * hdim_v);
        const ck_tile::index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q       = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k       = (nhead_k * shape_seqlen_k * hdim_q);
        const ck_tile::index_t batch_stride_v       = (nhead_k * hdim_v * shape_seqlen_k);
        const ck_tile::index_t batch_stride_bias    = (0 * nhead * shape_seqlen_q * shape_seqlen_k);
        const ck_tile::index_t batch_stride_randval = (nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_lse     = (nhead * max_seqlen_q);
        const ck_tile::index_t batch_stride_lse_acc = (nhead * max_seqlen_q);
        const ck_tile::index_t batch_stride_o_acc   = (nhead * max_seqlen_q * hdim_v);
        const ck_tile::index_t batch_stride_o       = (nhead * shape_seqlen_q * hdim_v);
        // setup split_stride_* arguments (only used in split-kv kernel)
        const ck_tile::index_t split_stride_lse_acc = (batch * nhead * max_seqlen_q);
        const ck_tile::index_t split_stride_o_acc   = (batch * nhead * max_seqlen_q * hdim_v);

        return fmha_fwd_args{mat_A,
                             mat_B0,
                             mat_B1,
                             keymask,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             mat_D,
                             nullptr,
                             nullptr,
                             nullptr,
                             shape_seqlen_q,
                             shape_seqlen_k,
                             batch,
                             max_seqlen_q,
                             hdim_q,
                             hdim_v,
                             nhead,
                             nhead_k,
                             1,
                             static_cast<float>(1.0 / ck_tile::sqrt(static_cast<float>(head_sz))),
                             1.f,
                             1.f,
                             stride_q,
                             stride_k,
                             stride_v,
                             stride_bias,
                             stride_randval,
                             stride_o_acc,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_randval,
                             nhead_stride_lse,
                             nhead_stride_lse_acc,
                             nhead_stride_o_acc,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_randval,
                             batch_stride_lse,
                             batch_stride_lse_acc,
                             batch_stride_o_acc,
                             batch_stride_o,
                             split_stride_lse_acc,
                             split_stride_o_acc,
                             0,
                             0,
                             static_cast<ck_tile::index_t>(mask_enum::no_mask),
                             0,
                             false,
                             {0, 0}};
    }();
    float ave_time = fmha_fwd(fmha_traits, fmha_args, stream_config);
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
template struct functor::GemvSoftmaxGemvFunctor<GPUDevice, Eigen::half>;
}  // namespace tensorflow
#endif
