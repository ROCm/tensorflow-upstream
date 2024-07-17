#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#include "ck_tile/08_gemv_softmax_gemv/gemv_softmax_gemv_dispatch.hpp"
#include "ck_tile/08_gemv_softmax_gemv/gemv_softmax_gemv_headdim_switch.hpp"
#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

extern template void
run_gemv_softmax_gemv<ck_tile::fp16_t, ck_tile::fp16_t, 64>(
    const GemvSoftmaxGemvParams& param, hipStream_t stream);
extern template void
run_gemv_softmax_gemv<ck_tile::fp16_t, ck_tile::fp16_t, 128>(
    const GemvSoftmaxGemvParams& param, hipStream_t stream);
extern template void
run_gemv_softmax_gemv<ck_tile::fp16_t, ck_tile::fp16_t, 256>(
    const GemvSoftmaxGemvParams& param, hipStream_t stream);

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

void gemv_softmax_gemv_fp16(GemvSoftmaxGemvParams& param, hipStream_t stream) {
  HEAD_SZ_SWITCH(param.head_sz, kMaxK, [&] {
    run_gemv_softmax_gemv<ck_tile::fp16_t, ck_tile::fp16_t, kMaxK>(param,
                                                                   stream);
  });
};

namespace functor {
template <typename T>
struct GemvSoftmaxGemvFunctor<GPUDevice, T> {
 public:
  static Status Compute(const GPUDevice& d, const void* mat_A,
                        const void* mat_B0, const void* keymask,
                        const void* mat_B1, void* mat_D, int head_sz, int seq,
                        int batch) {
    const hipStream_t stream = d.stream();
    GemvSoftmaxGemvParams params;

    params.a_ptr = mat_A;
    params.b0_ptr = mat_B0;
    params.mask_ptr = keymask;
    params.b1_ptr = mat_B1;  // workspace
    params.d_ptr = mat_D;
    params.num_batch = batch;
    params.seqlen = seq;
    params.num_head = 1;  // num_head > 1 is supported
    params.head_sz = head_sz;
    params.a_batch_stride = head_sz;
    params.a_nhead_stride = head_sz;
    params.b0_batch_stride = seq * head_sz;
    params.b0_seq_stride = head_sz;
    params.b0_nhead_stride = head_sz;
    params.mask_batch_stride = seq;
    params.b1_batch_stride = seq * head_sz;
    params.b1_seq_stride = head_sz;
    params.b1_nhead_stride = head_sz;
    params.d_batch_stride = head_sz;
    params.d_nhead_stride = head_sz;
    if constexpr (std::is_same_v<T, Eigen::half>) {
      gemv_softmax_gemv_fp16(params, stream);
    }
    return Status::OK();
  }
};  // struct Fused_Gemm_Bias_Add_Functor
}  // namespace functor
template struct functor::GemvSoftmaxGemvFunctor<GPUDevice, Eigen::half>;
}  // namespace tensorflow
#endif
