#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include <ck_tile/core.hpp>

#include "ck_tile/05_gemm_row_softmax_gemm/gemm_row_softmax_gemm_dispatch.hpp"
#include "ck_tile/05_gemm_row_softmax_gemm/gemm_row_softmax_gemm_headdim_switch.hpp"
#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

extern template void
run_gemm_row_softmax_gemm<ck_tile::fp16_t, ck_tile::fp16_t, 8, 16>(
    const GemmRowSoftmaxGemmParams& param, hipStream_t stream);
extern template void
run_gemm_row_softmax_gemm<ck_tile::fp16_t, ck_tile::fp16_t, 8, 32>(
    const GemmRowSoftmaxGemmParams& param, hipStream_t stream);
extern template void
run_gemm_row_softmax_gemm<ck_tile::fp16_t, ck_tile::fp16_t, 8, 64>(
    const GemmRowSoftmaxGemmParams& param, hipStream_t stream);
extern template void
run_gemm_row_softmax_gemm<ck_tile::fp16_t, ck_tile::fp16_t, 16, 16>(
    const GemmRowSoftmaxGemmParams& param, hipStream_t stream);
extern template void
run_gemm_row_softmax_gemm<ck_tile::fp16_t, ck_tile::fp16_t, 16, 32>(
    const GemmRowSoftmaxGemmParams& param, hipStream_t stream);
extern template void
run_gemm_row_softmax_gemm<ck_tile::fp16_t, ck_tile::fp16_t, 16, 64>(
    const GemmRowSoftmaxGemmParams& param, hipStream_t stream);
extern template void
run_gemm_row_softmax_gemm<ck_tile::fp16_t, ck_tile::fp16_t, 32, 32>(
    const GemmRowSoftmaxGemmParams& param, hipStream_t stream);
extern template void
run_gemm_row_softmax_gemm<ck_tile::fp16_t, ck_tile::fp16_t, 32, 64>(
    const GemmRowSoftmaxGemmParams& param, hipStream_t stream);

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

void gemm_row_softmax_gemm_fp16(GemmRowSoftmaxGemmParams& param,
                                hipStream_t stream) {
  HEAD_SZ_SWITCH(param.b0_head_sz, Gemm0MaxK, [&] {
    NEW_HEAD_SZ_SWITCH(param.b1_head_sz, Gemm1MaxK, [&] {
      if constexpr (Gemm0MaxK <= Gemm1MaxK) {
        run_gemm_row_softmax_gemm<ck_tile::fp16_t, ck_tile::fp16_t, Gemm0MaxK,
                                  Gemm1MaxK>(param, stream);
      };
    });
  });
};

namespace functor {
template <typename T>
struct GemmRowSoftmaxGemmFunctor<GPUDevice, T> {
 public:
  static Status Compute(const GPUDevice& d, const void* mat_B0,
                        const void* mat_A0, const void* mat_A1,
                        const void* Keymask, void* mat_B1, void* mat_D,
                        int batch, int seq, int head_num, int new_head) {
    const hipStream_t stream = d.stream();

    GemmRowSoftmaxGemmParams params;

    params.b0_ptr = mat_B0;
    params.a0_ptr = mat_A0;
    params.mask_ptr = Keymask;
    params.b1_ptr = mat_B1;  // workspace
    params.a1_ptr = mat_A1;
    params.d_ptr = mat_D;
    params.num_batch = batch;
    params.seqlen = seq;
    params.b0_head_sz = head_num;
    params.b1_head_sz = new_head;
    params.b0_batch_stride = head_num * seq;
    params.b0_head_stride = seq;
    params.a0_ld_sz = head_num;
    params.mask_batch_stride = seq;
    params.b1_batch_stride = new_head * seq;
    params.b1_head_stride = seq;
    params.a1_ld_sz = new_head;
    params.d_batch_stride = head_num * seq;
    params.d_head_stride = seq;
    if constexpr (std::is_same_v<T, Eigen::half>) {
      gemm_row_softmax_gemm_fp16(params, stream);
    }
    return Status::OK();
  }
};  // struct Fused_Gemm_Bias_Add_Functor
}  // namespace functor
template struct functor::GemmRowSoftmaxGemmFunctor<GPUDevice, Eigen::half>;
}  // namespace tensorflow
#endif
