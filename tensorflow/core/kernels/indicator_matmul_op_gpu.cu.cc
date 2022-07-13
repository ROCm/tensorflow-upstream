//
// Created by qiaoxj on 2019-12-10.
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/indicator_matmul_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

namespace {
template <typename T>
inline se::DeviceMemory<T> AsDeviceMemory(const T* gpu_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(gpu_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace

template <typename T>
struct HalfAsFloat {
  typedef T type;
};

template <>
struct HalfAsFloat<Eigen::half> {
  typedef float type;
};

template <typename Scalar, typename TIndex>
struct IMatmulParam {
  Scalar* A;
  Scalar* B;
  Scalar* C;
  Scalar** As;
  Scalar** Bs;
  Scalar** Cs;
  TIndex* indicators;
  int m, n, k;
  int batch_a, batch_b;
};

template <typename Scalar, typename TIndex>
__global__ void ComputePtrsKernel(IMatmulParam<Scalar, TIndex> param) {
  int m = param.m, n = param.n, k = param.k;
  int batch_a = param.batch_a, batch_b = param.batch_b;
  Scalar* A = param.A + blockIdx.x * batch_a * m * k;
  Scalar* B = param.B + blockIdx.x * batch_b * k * n;
  Scalar* C = param.C + blockIdx.x * batch_b * m * n;
  for (int i = threadIdx.x; i < batch_b; i += blockDim.x) {
    int64 offset = blockIdx.x * batch_b + i;
    int64 ind = (int64)param.indicators[i];
    if (ind < 0 || ind >= batch_a) {
      //printf("Indicator ERROR for indicator_matmul, indicator: %d.\n", ind);
      ind = 0;
    }
    param.As[offset] = &A[ind * m * k];
    param.Bs[offset] = &B[i * k * n];
    param.Cs[offset] = &C[i * m * n];
  }
}

template <typename Scalar>
void RunGemmStridedBatched(OpKernelContext* context, bool trans_a, bool trans_b,
                           int64 m, int64 n, int64 k, Scalar alpha,
                           const se::DeviceMemory<Scalar>& a, int64 stride_a,
                           const se::DeviceMemory<Scalar>& b, int64 stride_b,
                           Scalar beta, se::DeviceMemory<Scalar>* c,
                           int64 stride_c, int64 batch_count) {
  typedef typename HalfAsFloat<Scalar>::type CUDA_T;
  int lda = trans_a ? m : k;
  int ldb = trans_b ? k : n;
  int ldc = n;
  auto trans_a_tf = trans_a ? se::blas::Transpose::kTranspose
                            : se::blas::Transpose::kNoTranspose;
  auto trans_b_tf = trans_b ? se::blas::Transpose::kTranspose
                            : se::blas::Transpose::kNoTranspose;
  auto* stream = context->op_device_context()->stream();
  bool blas_launch_status =
      stream
          ->ThenBlasGemmStridedBatched(
              trans_b_tf, trans_a_tf, n, m, k, static_cast<CUDA_T>(alpha), b,
              ldb, stride_b, a, lda, stride_a, static_cast<CUDA_T>(beta), c,
              ldc, stride_c, batch_count)
          .ok();
  if (!blas_launch_status) {
    context->SetStatus(errors::Internal(
        "Blas GemmStridedBatched launch failed : m=", m, ", n=", n, ", k=", k));
  }
}

template <typename Scalar>
void RunGemmBatched(OpKernelContext* context, bool trans_a, bool trans_b,
                    int64 m, int64 n, int64 k, Scalar alpha, Scalar** a_ptrs,
                    Scalar** b_ptrs, Scalar beta, Scalar** c_ptrs,
                    int64 batch_count) {
  typedef typename HalfAsFloat<Scalar>::type CUDA_T;
  int lda = trans_a ? m : k;
  int ldb = trans_b ? k : n;
  int ldc = n;
  auto trans_a_tf = trans_a ? se::blas::Transpose::kTranspose
                            : se::blas::Transpose::kNoTranspose;
  auto trans_b_tf = trans_b ? se::blas::Transpose::kTranspose
                            : se::blas::Transpose::kNoTranspose;
  auto* stream = context->op_device_context()->stream();
  bool blas_launch_status =
      stream
          ->ThenBlasGemmBatched(
              trans_b_tf, trans_a_tf, n, m, k, static_cast<CUDA_T>(alpha),
              const_cast<const Scalar**>(b_ptrs), ldb,
              const_cast<const Scalar**>(a_ptrs), lda,
              static_cast<CUDA_T>(beta), c_ptrs, ldc, batch_count)
          .ok();

  if (!blas_launch_status) {
    context->SetStatus(errors::Internal(
        "Blas GemmBatched launch failed : m=", m, ", n=", n, ", k=", k));
  }
}

template <typename Scalar, typename TIndex>
void LaunchIndicatorMatmul<GPUDevice, Scalar, TIndex>::operator()(
    OpKernelContext* context, bool trans_a, bool trans_b, int64 m, int64 n,
    int64 k, const Tensor& in_a, const Tensor& in_b, const Tensor& indicator,
    Tensor* out, int64 batch_a, int64 batch_b, int64 paralle_num) {
  if (paralle_num == 1 && batch_a == 1) {
    auto a_ptr = AsDeviceMemory(in_a.template flat<Scalar>().data());
    auto b_ptr = AsDeviceMemory(in_b.template flat<Scalar>().data());
    auto out_ptr = AsDeviceMemory(out->template flat<Scalar>().data());
    RunGemmStridedBatched<Scalar>(context, trans_a, trans_b, m, n, k,
                                  Scalar(1.0), a_ptr, 0, b_ptr, k * n,
                                  Scalar(0.0), &out_ptr, m * n, batch_b);
    return;
  }
  auto a_base_ptr = in_a.template flat<Scalar>().data();
  auto b_base_ptr = in_b.template flat<Scalar>().data();
  auto c_base_ptr = out->template flat<Scalar>().data();
  IMatmulParam<Scalar, TIndex> param;
  param.A = const_cast<Scalar*>(a_base_ptr);
  param.B = const_cast<Scalar*>(b_base_ptr);
  param.C = c_base_ptr;
  param.indicators =
      const_cast<TIndex*>(indicator.template flat<TIndex>().data());
  param.m = m, param.n = n, param.k = k;
  param.batch_a = batch_a, param.batch_b = batch_b;
  const int64 size = paralle_num * batch_b;
  Tensor a_ptrs, b_ptrs, c_ptrs;
  OP_REQUIRES_OK(
      context, context->allocate_temp(DT_UINT64, TensorShape({size}), &a_ptrs));
  OP_REQUIRES_OK(
      context, context->allocate_temp(DT_UINT64, TensorShape({size}), &b_ptrs));
  OP_REQUIRES_OK(
      context, context->allocate_temp(DT_UINT64, TensorShape({size}), &c_ptrs));
  param.As = reinterpret_cast<Scalar**>(a_ptrs.flat<uint64>().data());
  param.Bs = reinterpret_cast<Scalar**>(b_ptrs.flat<uint64>().data());
  param.Cs = reinterpret_cast<Scalar**>(c_ptrs.flat<uint64>().data());
  //  BlasScratchAllocator scratch_allocator(context);
  const auto& d = context->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(param.batch_b, d);
  TF_CHECK_OK(GpuLaunchKernel(ComputePtrsKernel<Scalar, TIndex>, paralle_num,
                              config.thread_per_block, 0, d.stream(), param));
  RunGemmBatched<Scalar>(context, trans_a, trans_b, m, n, k, Scalar(1.0),
                         param.As, param.Bs, Scalar(0.0), param.Cs,
                         batch_b * paralle_num);
}

template struct LaunchIndicatorMatmul<GPUDevice, float, int32>;
template struct LaunchIndicatorMatmul<GPUDevice, double, int32>;
template struct LaunchIndicatorMatmul<GPUDevice, Eigen::half, int32>;
template struct LaunchIndicatorMatmul<GPUDevice, float, int64>;
template struct LaunchIndicatorMatmul<GPUDevice, double, int64>;
template struct LaunchIndicatorMatmul<GPUDevice, Eigen::half, int64>;
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
