//
// Created by qiaoxj on 2019-12-10.
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/indicator_matmul_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_blas_lt.h"
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

class BlasScratchAllocator : public se::ScratchAllocator {
 public:
  using Stream = se::Stream;
  using DeviceMemoryBytes = se::DeviceMemory<uint8>;

  BlasScratchAllocator(OpKernelContext* context) : context_(context) {}

  int64 GetMemoryLimitInBytes() override { return -1; }

  se::port::StatusOr<DeviceMemoryBytes> AllocateBytes(
      int64 byte_size) override {
    Tensor temporary_memory;

    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory));
    if (!allocation_status.ok()) {
      return se::port::StatusOr<DeviceMemoryBytes>(
          DeviceMemoryBytes::MakeFromByteSize(nullptr, 0));
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    return se::port::StatusOr<DeviceMemoryBytes>(
        DeviceMemoryBytes::MakeFromByteSize(
            temporary_memory.flat<uint8>().data(),
            temporary_memory.flat<uint8>().size()));
  }

 private:
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};

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

struct StridedCopyParams {
  uint32_t m, n, k;
  uint32_t batch_a, batch_b;
  uint32_t bytes_per_scalar;
};

template <typename Scalar>
void RunGemmStridedBatched(OpKernelContext* context, bool trans_a, bool trans_b,
                           int64 m, int64 n, int64 k, Scalar alpha,
                           const se::DeviceMemory<Scalar>& a, int64 stride_a,
                           const se::DeviceMemory<Scalar>& b, int64 stride_b,
                           Scalar beta, se::DeviceMemory<Scalar>* c,
                           int64 stride_c, int64 batch_count,
                           se::ScratchAllocator* allocator) {
	VLOG(1) << "Running RunGemmStridedBatched";
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
              ldc, stride_c, batch_count, allocator)
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
                    int64 batch_count, se::ScratchAllocator* allocator) {
	VLOG(1) << "Running RunGemmBatched";
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
              static_cast<CUDA_T>(beta), c_ptrs, ldc, batch_count,
              allocator)
          .ok();

  if (!blas_launch_status) {
    context->SetStatus(errors::Internal(
        "Blas GemmBatched launch failed : m=", m, ", n=", n, ", k=", k));
  }
}


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

#if TENSORFLOW_USE_ROCM
#if 0
#define LOAD(addr) __builtin_nontemporal_load(addr)
#else
#define LOAD(addr) (addr)[0]
#endif
#if 1
#define STORE(x, addr) __builtin_nontemporal_store((x), (addr))
#else
#define STORE(x, addr) (addr)[0] = (x)
#endif

template <uint32_t BlockSz, typename TIndex>
__global__ void StridedCopyKernel(const uint8_t *a_in, 
      const TIndex* indicators, uint8_t *a_out, StridedCopyParams p) {

  uint32_t pnum = blockIdx.x, batch = blockIdx.y, tid = threadIdx.x;
  __shared__ TIndex shidx;
  if(tid == 0) shidx = indicators[batch];
  __syncthreads();

  using Word = uint64_t;
  using Short = uint16_t;
  constexpr uint32_t chunk_sz = sizeof(Word)*2;

  uint32_t bytes = p.m*p.k*p.bytes_per_scalar;
  auto src = (const Word *)(a_in + (pnum * p.batch_a + shidx) * bytes);
  auto dst = (Word *)(a_out + (pnum * p.batch_b + batch) * bytes);

  uint32_t nwords = bytes / chunk_sz;
  for(uint32_t ofs = tid; ofs < nwords; ofs += BlockSz) {
    Word r1 = LOAD(src + ofs*2),
         r2 = LOAD(src + ofs*2 + 1);
    STORE(r1, dst + ofs*2);
    STORE(r2, dst + ofs*2 + 1);
  }

  const uint32_t bytes_mod = bytes % chunk_sz;
  if(tid < bytes_mod / sizeof(Short)) {
    uint32_t ofs = (bytes & ~(chunk_sz-1))/sizeof(Short) + tid;
    auto r1 = LOAD((const Short *)src + ofs);
    STORE(r1, (Short *)dst + ofs);
    // if(batch == 0) printf("bytes mod: %d, ofs: %d, src: %p dest: %p\n", 
    //         bytes_mod, ofs, (const Short *)src + ofs, (Short *)dst + ofs);
  }
}
#endif // TENSORFLOW_USE_ROCM


template <typename Scalar, typename TIndex>
void LaunchIndicatorMatmul<GPUDevice, Scalar, TIndex>::operator()(
    OpKernelContext* context, bool trans_a, bool trans_b, int64 m, int64 n,
    int64 k, const Tensor& in_a, const Tensor& in_b, const Tensor& indicator,
    Tensor* out, int64 batch_a, int64 batch_b, int64 parallel_num) {

  auto a_base_ptr = in_a.template flat<Scalar>().data();
  auto b_base_ptr = in_b.template flat<Scalar>().data();
  auto c_base_ptr = out->template flat<Scalar>().data();
  BlasScratchAllocator scratch_allocator(context);

  auto b_dev_ptr = AsDeviceMemory(b_base_ptr);
  auto out_dev_ptr = AsDeviceMemory(c_base_ptr);
  if (parallel_num == 1 && batch_a == 1) {
    auto a_dev_ptr = AsDeviceMemory(a_base_ptr);
    RunGemmStridedBatched<Scalar>(context, trans_a, trans_b, m, n, k,
                                  Scalar(1.0), a_dev_ptr, 0, b_dev_ptr, k * n,
                                  Scalar(0.0), &out_dev_ptr, m * n, batch_b,
                                  &scratch_allocator);
    return;
  }
#if TENSORFLOW_USE_ROCM
  // we use StridedBatched variant if hipblaslt is enabled (GroupedGemm not yet ready)
  if(se::gpu::GpuBlasLtEnabled()) { 
    auto ind_ptr = indicator.template flat<TIndex>().data();
    const int64 size = parallel_num * batch_b * m * k;
    //VLOG(0) << "Allocating " << (size * sizeof(Scalar)) << " bytes..";
    Tensor a_strided;
    OP_REQUIRES_OK(
       context, context->allocate_temp(DataTypeToEnum<Scalar>::v(), 
                                    TensorShape({size}), &a_strided));
  
    auto a_strided_ptr = a_strided.flat<Scalar>().data();
    auto* stream = context->op_device_context()->stream();

    dim3 grid(parallel_num, batch_b, 1);
    constexpr uint32_t BlockSz = 256;
    StridedCopyParams params{ (uint32_t)m, (uint32_t)n, (uint32_t)k, 
        (uint32_t)batch_a, (uint32_t)batch_b, (uint32_t)sizeof(Scalar)};

    TF_CHECK_OK(GpuLaunchKernel(StridedCopyKernel<BlockSz, TIndex>, grid,
                        BlockSz, 0, se::gpu::AsGpuStreamValue(stream), 
                  reinterpret_cast<const uint8_t *>(a_base_ptr), ind_ptr, 
                  reinterpret_cast<uint8_t *>(a_strided_ptr), params));

    auto a_dev_ptr = AsDeviceMemory(a_strided_ptr);
    auto B = parallel_num * batch_b;
    return RunGemmStridedBatched<Scalar>(context, trans_a, trans_b, m, n, k,
                            Scalar(1.0), a_dev_ptr, m * k, b_dev_ptr, k * n,
                            Scalar(0.0), &out_dev_ptr, m * n, B,
                            &scratch_allocator);
  } 
#endif // TENSORFLOW_USE_ROCM 
  IMatmulParam<Scalar, TIndex> param;
  param.A = const_cast<Scalar*>(a_base_ptr);
  param.B = const_cast<Scalar*>(b_base_ptr);
  param.C = c_base_ptr;
  param.indicators =
      const_cast<TIndex*>(indicator.template flat<TIndex>().data());
  param.m = m, param.n = n, param.k = k;
  param.batch_a = batch_a, param.batch_b = batch_b;
  const int64 size = parallel_num * batch_b;
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
  
  const auto& d = context->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(param.batch_b, d);
  TF_CHECK_OK(GpuLaunchKernel(ComputePtrsKernel<Scalar, TIndex>, parallel_num,
                              config.thread_per_block, 0, d.stream(), param));
  auto* stream = context->op_device_context()->stream();
  TF_CHECK_OK(stream->BlockHostUntilDone());
  RunGemmBatched<Scalar>(context, trans_a, trans_b, m, n, k, Scalar(1.0),
                         param.As, param.Bs, Scalar(0.0), param.Cs,
                         batch_b * parallel_num, &scratch_allocator);
} // LaunchIndicatorMatmul


template struct LaunchIndicatorMatmul<GPUDevice, float, int32>;
template struct LaunchIndicatorMatmul<GPUDevice, double, int32>;
template struct LaunchIndicatorMatmul<GPUDevice, Eigen::half, int32>;
template struct LaunchIndicatorMatmul<GPUDevice, float, int64>;
template struct LaunchIndicatorMatmul<GPUDevice, double, int64>;
template struct LaunchIndicatorMatmul<GPUDevice, Eigen::half, int64>;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}  // namespace tensorflow
