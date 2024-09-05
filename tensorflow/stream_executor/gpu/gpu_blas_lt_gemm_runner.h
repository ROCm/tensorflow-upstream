/* Copyright 2023 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_GEMM_RUNNER_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_GEMM_RUNNER_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/stream_executor/gpu/gpu_blas_lt.h"
#include "tensorflow/stream_executor/scratch_allocator.h"

namespace xla {
namespace gpu {
class AutotuneConfig;

}
}

namespace stream_executor {
  
namespace  gpu {

struct StridedGemmConfig {
  int64 m, n, k, batch_count;
  blas::Transpose trans_a, trans_b;
  xla::complex128 alpha;
  double beta;
  blas::DataType type_a, type_b, type_c, type_d;
  int64 lda, ldb, ldc, ldd;
  int64 stride_a, stride_b, stride_c, stride_d;
  blas::ComputationType compute_type;
};

namespace {

auto AsTuple(const GroupedGemmConfig& p) {
    // NOTE: alpha, beta and data pointers are not included in cache !!
  return std::make_tuple(p.m, p.n, p.k, p.batch_count,
                         p.trans_a, p.trans_b,
                         p.type_a, p.type_b, p.type_c, p.type_d,
                         p.lda, p.ldb, p.ldc, p.ldd, 
                         p.compute_type);
}

auto AsTuple(const StridedGemmConfig& p) {
  return std::make_tuple(p.m, p.n, p.k, p.batch_count,
                         p.trans_a, p.trans_b, p.alpha.real(), p.alpha.imag(), p.beta,
                         p.type_a, p.type_b, p.type_c, p.type_d,
                         p.lda, p.ldb, p.ldc, p.ldd, 
                         p.stride_a, p.stride_b, p.stride_c, p.stride_d,
                         p.compute_type);
}

} // namespace

bool operator ==(const GroupedGemmConfig& rhs, const GroupedGemmConfig& lhs);
bool operator ==(const StridedGemmConfig& rhs, const StridedGemmConfig& lhs);

template <typename H>
H AbslHashValue(H h, const GroupedGemmConfig& params) {
  return H::combine(std::move(h), AsTuple(params));
}

template <typename H>
H AbslHashValue(H h, const StridedGemmConfig& params) {
  return H::combine(std::move(h), AsTuple(params));
}

struct BlasLtGemmRunner {

  static BlasLtGemmRunner& i(const Stream *stream);
  const static size_t MaxSolutions = 384;

  template < class Scalar >
  xla::complex128 Convert(Scalar x) {
    
    if constexpr(std::is_same<Scalar, xla::complex64>::value ||
                 std::is_same<Scalar, xla::complex128>::value) {
      return static_cast< xla::complex128 >(x);
    } else {
      return static_cast< double >(x);
    }
  }

  template < class Scalar, class T >
  xla::Status Run(Stream& stream, blas::Transpose trans_a, 
    blas::Transpose trans_b, int64 m, int64 n, int64 k, 
    Scalar alpha, const DeviceMemory<T>& a, int64 lda, 
    const DeviceMemory<T>& b, int64 ldb, 
    Scalar beta, DeviceMemory<T> *c, int64 ldc, 
    ScratchAllocator* allocator) {
    
    auto type = dnn::ToDataType<T>::value;
    
    return RunStridedBatchedImpl(stream, trans_a, trans_b, m, n, k, 
      Convert(alpha), 
      type, a, lda, 0, type, b, ldb, 0, 
      Convert(beta).real(), // only real betas are supported!!
      type, c, ldc, 0, 1, allocator);
  }

  template < class Scalar, class T >
  xla::Status RunStridedBatched(Stream& stream, blas::Transpose trans_a, 
    blas::Transpose trans_b, int64 m, int64 n, int64 k, 
    Scalar alpha, const DeviceMemory<T>& a, int64 lda, int64 stride_a, 
    const DeviceMemory<T>& b, int64 ldb, int64 stride_b,
    Scalar beta, DeviceMemory<T> *c, int64 ldc, int64 stride_c,
    int64 batch_count, ScratchAllocator* allocator) {
    
    auto type = dnn::ToDataType<T>::value;
    return RunStridedBatchedImpl(stream, trans_a, trans_b, m, n, k, 
      Convert(alpha), 
      type, a, lda, stride_a, type, b, ldb, stride_b, 
      Convert(beta).real(), // only real betas are supported!!
      type, c, ldc, stride_c, batch_count, allocator);
  }

  template < class Scalar, class T >
  xla::Status RunBatched(Stream& stream, blas::Transpose trans_a, 
      blas::Transpose trans_b, int64 m, int64 n, int64 k, Scalar alpha, 
      const port::ArraySlice<DeviceMemory<T> *> &a, int64 lda, 
      const port::ArraySlice<DeviceMemory<T> *> &b, int64 ldb, Scalar beta,
      const port::ArraySlice<DeviceMemory<T> *> &c, int64 ldc, 
      int64 batch_count, ScratchAllocator* allocator) {
    
    // NOTE: Scalar types shall be verified for correctness vs T!!
    auto type = dnn::ToDataType<T>::value;
    auto cvt = [](auto x){
      using TT = port::ArraySlice<DeviceMemoryBase *>;
      return reinterpret_cast<const TT&>(x);
    };

    auto res = ContiguousStrides(cvt(a), cvt(b), cvt(c), batch_count);
    if (res.ok()) {
      auto strides = std::move(res.ValueOrDie());
      return RunStridedBatchedImpl(stream, trans_a, trans_b, m, n, k, 
          Convert(alpha), 
          type, *a[0], lda, strides[0] / sizeof(T), 
          type, *b[0], ldb, strides[1] / sizeof(T), 
          Convert(beta).real(), // only real betas are supported!!
          type, c[0], ldc, strides[2] / sizeof(T), batch_count, allocator);
    }
    return port::InternalError("RunBatched: port::ArraySlice NYI!");
  }

  template < class Scalar, class T >
  xla::Status RunBatched(Stream& stream, blas::Transpose trans_a, 
      blas::Transpose trans_b, int64 m, int64 n, int64 k, 
      Scalar alpha, const T** a, int64 lda, 
      const T** b, int64 ldb, Scalar beta,
      T** c, int64 ldc, int64 batch_count, ScratchAllocator* allocator) {

    // NOTE: Scalar types shall be verified for correctness vs T!!
    auto type = dnn::ToDataType<T>::value;
    return RunBatchedImpl(stream, trans_a, trans_b, m, n, k, 
      &alpha, type, reinterpret_cast< const void **>(a), lda, 
      type, reinterpret_cast< const void **>(b), ldb, &beta, 
      type, reinterpret_cast< void **>(c), ldc, batch_count, allocator);
  }

private:
  explicit BlasLtGemmRunner(StreamExecutor *parent);

  template < class TuneFunc >
  xla::StatusOr< gpu::BlasLt::MatmulAlgorithm > Autotune(
            const std::vector< gpu::BlasLt::MatmulAlgorithm >& algorithms,
                                TuneFunc&& benchmark_func);

  xla::Status RunBatchedImpl(Stream& stream, blas::Transpose trans_a, 
      blas::Transpose trans_b, int64 m, int64 n, int64 k, 
      const void *alpha, blas::DataType type_a, const void** a, int64 lda, 
      blas::DataType type_b, const void** b, int64 ldb, const void *beta,
      blas::DataType type_c, void** c, int64 ldc, int64 batch_count,
      ScratchAllocator* allocator);

  xla::Status RunStridedBatchedImpl(Stream& stream, blas::Transpose trans_a, 
      blas::Transpose trans_b, int64 m, int64 n, int64 k, 
      xla::complex128 alpha, 
      blas::DataType type_a, const DeviceMemoryBase& a, int64 lda, int64 stride_a,
      blas::DataType type_b, const DeviceMemoryBase& b, int64 ldb, int64 stride_b,
      double beta,
      blas::DataType type_c, DeviceMemoryBase *c, int64 ldc, int64 stride_c, 
      int64 batch_count, ScratchAllocator* allocator);

  xla::StatusOr< std::array< uint64_t, 3 >> ContiguousStrides(
      const port::ArraySlice<DeviceMemoryBase *>& a, 
      const port::ArraySlice<DeviceMemoryBase *>& b, 
      const port::ArraySlice<DeviceMemoryBase *>& c, int64 batch_count);

  std::unique_ptr< absl::Mutex > mutex_;
  // std::unique_ptr< xla::gpu::AutotuneConfig > config_;
  absl::flat_hash_map<GroupedGemmConfig, BlasLt::GroupedMatmulPlanPtr> grouped_gemm_map_;
  absl::flat_hash_map<StridedGemmConfig, BlasLt::MatmulPlanPtr> strided_gemm_map_;
};

} // namespace gpu

} // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_GEMM_RUNNER_H_
