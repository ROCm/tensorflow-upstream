#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include <ck_tile/core.hpp>

#include "gather_attention_dispatch.hpp"
#include "gather_attention_headdim_switch.hpp"
#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// clang-format off
extern template void run_gather_attention<ck_tile::fp16_t, ck_tile::fp16_t,  32>(const GatherAttentionParams& param, hipStream_t stream);
extern template void run_gather_attention<ck_tile::fp16_t, ck_tile::fp16_t,  64>(const GatherAttentionParams& param, hipStream_t stream);
extern template void run_gather_attention<ck_tile::fp16_t, ck_tile::fp16_t,  128>(const GatherAttentionParams& param, hipStream_t stream);
extern template void run_gather_attention<ck_tile::fp16_t, ck_tile::fp16_t,  256>(const GatherAttentionParams& param, hipStream_t stream);
// clang-format on

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

void gather_attention_fp16(GatherAttentionParams& param, hipStream_t stream) {
  HEAD_SZ_SWITCH(param.head_sz, kMaxK, [&] {
    run_gather_attention<ck_tile::fp16_t, ck_tile::fp16_t, kMaxK>(param,
                                                                  stream);
  });
};

namespace functor {
template <typename dataTP_>
struct GatherAttentionFunctor<GPUDevice, dataTP_> {
 public:
  static Status Compute(const GPUDevice& d, const void* mat_A,
                        const void* mat_B0, const void* keymask,
                        const int* indices, const void* mat_B1, void* mat_D,
                        int head_sz, int seq, int B, int index, int head_num) {
    const bool time_kernel = std::getenv("TF_CK_TIME_KERNEL") != nullptr;
    const hipStream_t stream = d.stream();

    GatherAttentionParams params;

    params.a_ptr = mat_A;
    params.b0_ptr = mat_B0;
    params.mask_ptr = keymask;
    params.indices_ptr = const_cast<int*>(indices);
    params.b1_ptr = mat_B1;  // workspace
    params.d_ptr = mat_D;
    params.num_batch = B;
    params.num_index = index;
    params.seqlen = seq;
    params.num_head = head_num;
    params.head_sz = head_sz;
    params.a_batch_stride = head_sz * head_num;
    params.a_nhead_stride = head_sz;
    params.b0_batch_stride = head_sz * head_num * seq;
    params.b0_seq_stride = head_sz * head_num;
    params.b0_nhead_stride = head_sz;
    params.mask_batch_stride = seq;
    params.b1_batch_stride = head_sz * head_num * seq;
    params.b1_seq_stride = head_sz * head_num;
    params.b1_nhead_stride = head_sz;
    params.d_batch_stride = head_sz * head_num;
    params.d_nhead_stride = head_sz;

    gather_attention_fp16(params, stream);
    // if (time_kernel) {
    //   std::size_t flop = std::size_t(2) * M * N * K;
    //   std::size_t num_btype = sizeof(A0DataType) * M * K +
    //                           sizeof(B0DataType) * K * N +
    //                           sizeof(EDataType) * M * N;

    //   float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    //   float gb_per_sec = num_btype / 1.E6 / ave_time;
    //   LOG(INFO) << "Running time: " << ave_time << " ms, " << tflops
    //             << " TFlops, " << gb_per_sec << " GB/s";
    // }
    // hipFree(mat_B1);
    return Status::OK();
  }
};  // struct Fused_Gemm_Bias_Add_Functor
}  // namespace functor
template struct functor::GatherAttentionFunctor<GPUDevice, Eigen::half>;
}  // namespace tensorflow
#endif
