#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include <ck_tile/core.hpp>

#include "gemm_row_softmax_gemm_dispatch.h"
#include "gemm_row_softmax_gemm_headdim_switch.h"
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
      if (Gemm0MaxK <= Gemm1MaxK) {
        run_gemm_row_softmax_gemm<ck_tile::fp16_t, ck_tile::fp16_t, Gemm0MaxK,
                                  Gemm1MaxK>(param, stream);
      };
    });
  });
};

namespace functor {
template <typename dataTP_>
struct GemmRowSoftmaxGemmFunctor<GPUDevice, dataTP_> {
 public:
  static Status Compute(const GPUDevice& d, const void* mat_B0,
                        const void* mat_A0, const void* mat_A1,
                        const void* Keymask, void* mat_B1, void* mat_D,
                        int batch, int seq, int head_num, int new_head) {
    const bool time_kernel = std::getenv("TF_CK_TIME_KERNEL") != nullptr;
    const auto& stream = d.stream();

    // void* mat_B1 = nullptr;
    // hipMalloc(&mat_B1, batch*new_head*seq*sizeof(dataTP_));
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

    gemm_row_softmax_gemm_fp16(params, stream);
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
template struct functor::GemmRowSoftmaxGemmFunctor<GPUDevice, Eigen::half>;
}  // namespace tensorflow
#endif
