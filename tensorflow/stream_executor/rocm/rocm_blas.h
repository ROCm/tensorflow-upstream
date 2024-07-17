/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// ROCM-specific support for BLAS functionality -- this wraps the rocBLAS
// library capabilities, and is only included into ROCM implementation code --
// it will not introduce rocm headers into other code.

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_BLAS_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_BLAS_H_

#include "absl/synchronization/mutex.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/temporary_device_memory.h"

namespace stream_executor {

class Stream;

namespace gpu {

// Type conversion helper that helps to map non-rocblas types to rocblas types
// Right now, it only converts the Eigen::half type to rocblas_half type
template <typename T>
struct RocBlasTypeConversionHelper {
  using mapped_type = T;
};

template <>
struct RocBlasTypeConversionHelper<Eigen::half> {
  using mapped_type = rocblas_half;
};

// Opaque and unique identifier for the rocBLAS plugin.
extern const PluginId kRocBlasPlugin;

class GpuExecutor;

// BLAS plugin for ROCM platform via rocBLAS library.
//
// This satisfies the platform-agnostic BlasSupport interface.
//
// Note that the rocBLAS handle that this encapsulates is implicitly tied to the
// context (and, as a result, the device) that the parent GpuExecutor is tied
// to. This simply happens as an artifact of creating the rocBLAS handle when a
// ROCM context is active.
//
// Thread-safe post-initialization.
class ROCMBlas : public blas::BlasSupport {
 public:
  explicit ROCMBlas(GpuExecutor *parent);

  // Allocates a rocBLAS handle.
  bool Init();

  // Releases the rocBLAS handle, if present.
  ~ROCMBlas() override;

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

 private:
  // Tells rocBLAS to enqueue the BLAS operation onto a particular Stream.
  //
  // rocBLAS is stateful, and only be associated with one stream (in order to
  // enqueue dispatch) at a given time. As a result, this generally must be
  // invoked before calling into rocBLAS.
  bool SetStream(Stream *stream) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // A helper function that calls the real rocBLAS function together with error
  // handling.
  //
  // rocblas_func:       rocBLAS function pointer.
  // rocblas_name:       rocBLAS function name.
  // stream:             Stream to enqueue the BLAS operation onto.
  // pointer_mode_host:  Indicate if the pointer to a scalar value is from host
  //                     (true) or device (false).
  // err_on_failure:     Whether to print an error if the rocBLAS function
  // fails. args:               Arguments of rocBLAS function.
  template <typename FuncT, typename... Args>
  bool DoBlasInternalImpl(FuncT rocblas_func, Stream *stream,
                          bool pointer_mode_host, bool err_on_failure,
                          Args... args);

  // Convenience functions that call DoBlasInternalImpl with different values
  // for err_on_failure.
  template <typename FuncT, typename... Args>
  bool DoBlasInternal(FuncT rocblas_func, Stream *stream,
                      bool pointer_mode_host, Args... args) {
    return DoBlasInternalImpl(rocblas_func, stream, pointer_mode_host,
                              /*err_on_failure=*/true, args...);
  }
  template <typename FuncT, typename... Args>
  bool DoBlasInternalFailureOK(FuncT rocblas_func, Stream *stream,
                               bool pointer_mode_host, Args... args) {
    return DoBlasInternalImpl(rocblas_func, stream, pointer_mode_host,
                              /*err_on_failure=*/false, args...);
  }

  // A helper allocation funciton to convert raw pointers memory layout to
  // strided flavor
  template <typename T, typename U>
  port::Status AllocateStridedBuffer(
    const std::vector<U*> &raw_ptrs,
    int batch_count, uint64_t batch_stride, ScratchAllocator *scratch_allocator,
    Stream *stream,
    std::unique_ptr<TemporaryDeviceMemory<U> > *temp_memory,
    DeviceMemory<U> *device_memory);

  // A helper function to implement DoBlasGemmBatched interfaces for generic
  // types.
  //
  // Note: This function is implemented using gemm_strided_batched interface,
  // NOT gemm_batched interface, because rocblas do not support it. As a
  // result, if the passed in batch matrix are not allocated in strided batched
  // format, it might end up in non-trivial amount of memory allocation and
  // copy. To avoid this, always prioritize to use DoBlasGemmStridedBatched
  // interface.
  //
  // In most use cases, batch matrix do get allocated in strided manner, making
  // calling this interface equivalent with DoBlasGemmStridedBatched. The only
  // use case we see so far that violates this observation is when batch
  // matrix is created by broadcasting from a smaller matrix. When it happens,
  // It will take advantage of the AllocateStridedBuffer subroutine to
  // reallocate the memory layout to be strided batched.
  template <class T, class V>
  bool DoBlasGemmBatchedImpl(Stream *stream, blas::BatchedGemmCallContext<T> ctx, V strided_fun);

	template <class T, class V, class Alpha_T>
	bool DoBlasGemmBatchedImpl( V rocblas_func, Stream *stream, blas::Transpose transa,
			blas::Transpose transb, uint64 m, uint64 n, uint64 k, Alpha_T alpha,
			const T **a_array, int lda, const T **b_array, int ldb, Alpha_T beta,
			T **c_array, int ldc, int batch_count);

  template <typename T, typename FuncT, typename NativeT = T>
  port::Status DoBlasGemmBatchedInternal(
      FuncT rocblas_func, Stream* stream, blas::Transpose transa,
      blas::Transpose transb, uint64 m, uint64 n, uint64 k, T alpha,
      const T** a_array, int lda, const T** b_array, int ldb, T beta,
      T** c_array, int ldc, int batch_count);

  // Helper function for implementing DoBlasGemvWithProfiling.
  template <typename T>
  bool DoBlasGemvWithProfilingImpl(Stream *stream, blas::Transpose trans,
                                   uint64 m, uint64 n, const T &alpha,
                                   const DeviceMemory<T> &a, int lda,
                                   const DeviceMemory<T> &x, int incx,
                                   const T &beta, DeviceMemory<T> *y, int incy,
                                   blas::ProfileResult *output_profile_result);

  bool DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<Eigen::half> &a,
                          int lda, const DeviceMemory<Eigen::half> &b, int ldb,
                          float beta, DeviceMemory<Eigen::half> *c, int ldc,
                          blas::CallContext context);
  bool DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc);

  template <class T, class V, class W>
  bool DoBlasGemmImpl(Stream *stream, blas::GemmCallContext<T> ctx, V strided_fun, W fun);


  // mutex that guards the rocBLAS handle for this device.
  absl::Mutex mu_;

  // GpuExecutor which instantiated this ROCMBlas.
  // Immutable post-initialization.
  GpuExecutor *parent_;

  // rocBLAS library handle on the device.
  rocblas_handle blas_ GUARDED_BY(mu_);

  SE_DISALLOW_COPY_AND_ASSIGN(ROCMBlas);
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_BLAS_H_
