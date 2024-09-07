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

#include <algorithm>
#include <climits>
#include <memory>
#include <sstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rocm/rocm_config.h"
#include "rocm/include/rocblas/rocblas.h"
#include "rocm/include/hipblaslt/hipblaslt-ext.hpp"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/env_var.h"

#include "tensorflow/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/stream_executor/gpu/gpu_helpers.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/stream_executor/rocm/hip_blas_lt.h"
#include "tensorflow/stream_executor/rocm/rocm_blas.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream.h"

#define SET_ATTR(setter, handle, attr, value) \
  ToStatus(setter(handle, attr, &value, sizeof(decltype(value))), #setter)

// hipblasLtMatmulDescGetAttribute does not allow nullptr for the last
// argument (size_t* sizeWritten)
#define GET_ATTR(getter, handle, attr, ValueT)                          \
  [&]() -> xla::StatusOr<ValueT> {                                     \
    ValueT value;                                                       \
    size_t size;                                                        \
    TF_RETURN_IF_ERROR(ToStatus(                                        \
        getter(handle, attr, &value, sizeof(ValueT), &size), #getter)); \
    return std::move(value);                                            \
  }()

namespace stream_executor {

namespace gpu {
void rocm_null_gpu_job(void* stream);
void launch_notify(const char* name, void* stream);
void launch_notify_finish(const char* name, void* stream);
int launch_notify2(const char* name, void* stream);
void launch_notify_finish2(const char* name, void* stream, int t);
}

namespace rocm {

using ::xla::complex128;
using ::xla::complex64;
using tensorflow::errors::InvalidArgument;
using tensorflow::bfloat16;
using namespace hipblaslt_ext;

// void GroupGemmUpdateArgs(hipStream_t stream, 
//         UserArguments *dev_args,
//         const gpu::GroupedGemmConfig& cfg);

void GroupGemmUpdateArgs(hipStream_t stream, 
        UserArguments *dev_args,
        const void **a, const void **b, const void **c, void **d,
      uint32_t num_gemms);

namespace {

typedef struct __attribute__((packed, aligned(8))) _rocblaslt_matmul_algo {
    uint8_t data[8] = {0};
    bool fallback = false;
    size_t max_workspace_bytes = 0;
} rocblaslt_matmul_algo;

template <typename T>
xla::Status SetAttr(hipblasLtMatrixLayout_t handle,
                     hipblasLtMatrixLayoutAttribute_t attr, T value) {
  return SET_ATTR(hipblasLtMatrixLayoutSetAttribute, handle, attr, value);
}

template <typename T>
xla::StatusOr<T> GetAttr(hipblasLtMatrixLayout_t handle,
                          hipblasLtMatrixLayoutAttribute_t attr) {
  return GET_ATTR(hipblasLtMatrixLayoutGetAttribute, handle, attr, T);
}

template <typename T>
xla::Status SetAttr(hipblasLtMatmulDesc_t handle,
                     hipblasLtMatmulDescAttributes_t attr, T value) {
  return SET_ATTR(hipblasLtMatmulDescSetAttribute, handle, attr, value);
}

template <typename T>
xla::StatusOr<T> GetAttr(hipblasLtMatmulDesc_t handle,
                          hipblasLtMatmulDescAttributes_t attr) {
  return GET_ATTR(hipblasLtMatmulDescGetAttribute, handle, attr, T);
}

template <typename T>
xla::Status SetAttr(hipblasLtMatmulPreference_t handle,
                     hipblasLtMatmulPreferenceAttributes_t attr, T value) {
  return SET_ATTR(hipblasLtMatmulPreferenceSetAttribute, handle, attr,
                  value);
}

xla::StatusOr<hipblasLtEpilogue_t> AsHipblasLtEpilogue(
    gpu::BlasLt::Epilogue epilogue) {
  switch (epilogue) {
    case gpu::BlasLt::Epilogue::kDefault:
      return HIPBLASLT_EPILOGUE_DEFAULT;
    case gpu::BlasLt::Epilogue::kReLU:
      return HIPBLASLT_EPILOGUE_RELU;
    case gpu::BlasLt::Epilogue::kBias:
      return HIPBLASLT_EPILOGUE_BIAS;
    case gpu::BlasLt::Epilogue::kBiasThenReLU:
      return HIPBLASLT_EPILOGUE_RELU_BIAS;
    case gpu::BlasLt::Epilogue::kGELU:
      return HIPBLASLT_EPILOGUE_GELU;
#if TF_ROCM_VERSION >= 60000
    case gpu::BlasLt::Epilogue::kGELUWithAux:
      return HIPBLASLT_EPILOGUE_GELU_AUX;
    case gpu::BlasLt::Epilogue::kBiasThenGELU:
      return HIPBLASLT_EPILOGUE_GELU_BIAS;
    case gpu::BlasLt::Epilogue::kBiasThenGELUWithAux:
      return HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
#endif
    default:
      return xla::InternalError("Unsupported epilogue: %d",
                                 static_cast<int>(epilogue));
  }
}

}  // namespace



template <typename T>
uint32_t checksum(const T* p, int n, int& n_inf)
{
  const uint32_t* pp = reinterpret_cast<const uint32_t*>(p);
  n *= sizeof(T);
  n /= 4;
  n_inf = 0;
  uint32_t s = 0;
  for(int i=0; i<n; i++) {
    s ^= pp[i]*(i*3789597+1);
    if(sizeof(T)==4)
      n_inf += !isfinite(p[i]);
    else
      n_inf += (isfinite(float(p[2*i]))?0:1) + (isfinite(float(p[2*i+1]))?0:1);
  }
  return s;
}


xla::Status BlasLt::Init() {
  hipblasLtHandle_t blas_lt;
  SE_HIPBLAS_RETURN_IF_ERROR(hipblasLtCreate(&blas_lt));
  absl::MutexLock lock(&mu_);
  blas_lt_.reset(blas_lt);
  return xla::Status::OK();
}

/*static*/ xla::StatusOr<BlasLt::MatrixLayout> BlasLt::MatrixLayout::Create(
    const gpu::MatrixLayout& m) {

  auto hipblas_data_type_ = AsHipblasDataType(m.dtype);
  hipblasLtMatrixLayout_t hip_layout;
  SE_HIPBLAS_RETURN_IF_ERROR(hipblasLtMatrixLayoutCreate(
      &hip_layout, hipblas_data_type_, m.num_rows, m.num_cols,
      m.leading_dim_stride));
  // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
  BlasLt::MatrixLayout layout(hip_layout, hipblas_data_type_, m);
  if (m.order != gpu::MatrixLayout::Order::kColumnMajor) {
    return xla::InternalError("HipblasLT does not support row-major matrices");
  }
  TF_RETURN_IF_ERROR(SetAttr(hip_layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                             static_cast<int32_t>(m.batch_size)));

  VLOG(2) << "BlasLt::MatrixLayout::Create type: " << (int)m.dtype
          << " rows: " << m.num_rows << " cols: " << m.num_cols
          << " batch_size: " << m.batch_size
          << " leading_dim_stride: " << m.leading_dim_stride
          << " batch_stride: " << m.batch_stride;

  TF_RETURN_IF_ERROR(SetAttr(hip_layout,
                             HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                             m.batch_stride));
  return std::move(layout);
}

/*static*/ xla::StatusOr<BlasLt::MatmulDesc> BlasLt::MatmulDesc::Create(
    blas::ComputationType compute_type, blas::DataType scale_type,
    blas::Transpose trans_a, blas::Transpose trans_b, Epilogue epilogue,
    PointerMode pointer_mode) {
  hipblasLtMatmulDesc_t hip_desc;
  VLOG(2) << "BlasLt::MatmulDesc::Create compute_type: " << int(compute_type)
          << " scale_type: " << int(scale_type)
          << " epilogue: " << int(epilogue) << " trans_a: " << int(trans_a)
          << " trans_b: " << int(trans_b) << " pointer_mode "
          << int(pointer_mode);
  auto hip_scale_type = AsHipblasDataType(scale_type);
  auto hip_compute_type = AsHipblasComputeType(compute_type);
  SE_HIPBLAS_RETURN_IF_ERROR(hipblasLtMatmulDescCreate(
      &hip_desc, hip_compute_type, hip_scale_type));

  int32_t bias_flag =
      static_cast<int32_t>(epilogue) & static_cast<int32_t>(Epilogue::kBias);
  // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
  BlasLt::MatmulDesc desc(hip_desc, hip_compute_type, hip_scale_type,
                          bias_flag != 0);
  if (pointer_mode != PointerMode::kHost) {
    return xla::InternalError("hipblaslt does not support device pointers");
  }

  TF_RETURN_IF_ERROR(SetAttr(hip_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                             AsHipblasOperation(trans_a)));
  TF_RETURN_IF_ERROR(SetAttr(hip_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                             AsHipblasOperation(trans_b)));
  TF_ASSIGN_OR_RETURN(hipblasLtEpilogue_t epi, AsHipblasLtEpilogue(epilogue));
  TF_RETURN_IF_ERROR(SetAttr(hip_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, epi));
  return std::move(desc);
}

auto BlasLt::MatmulPlan::GetAlgorithms(size_t max_algorithm_count,
                                       size_t max_workspace_size) const
    -> xla::StatusOr<std::vector<MatmulAlgorithm>> {
  max_algorithm_count = std::min(max_algorithm_count, size_t{INT_MAX});
  std::vector<hipblasLtMatmulHeuristicResult_t> results(max_algorithm_count);

  {
    absl::MutexLock lock(&blas_lt_ref_.mu_);
    TF_RET_CHECK(blas_lt_ref_.blas_lt_ != nullptr);

    hipblasLtMatmulPreference_t hip_preference;
    SE_HIPBLAS_RETURN_IF_ERROR(
        hipblasLtMatmulPreferenceCreate(&hip_preference));

    // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
    Owned<hipblasLtMatmulPreference_t> preference(
        hip_preference, hipblasLtMatmulPreferenceDestroy);

    TF_RETURN_IF_ERROR(SetAttr<uint64_t>(
        hip_preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        max_workspace_size));

    gpu::ScopedActivateExecutorContext sac{blas_lt_ref_.parent_};

    // hipBlasLt requires setting the bias pointer (even a dummy one), otherwise
    // no algorithms can be found for "bias epilogues". This is to be removed
    // later when this limitation is gone.
    if (op_desc_.has_bias_epilogue()) {
      static int64_t dummyPointer = 0xACEBALL;
      TF_RETURN_IF_ERROR(SetAttr(
          op_desc_.get(), HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &dummyPointer));
    }

    int found_algorithm_count = 0;
    auto error = hipblasLtMatmulAlgoGetHeuristic(
        blas_lt_ref_.blas_lt_.get(), op_desc_.get(), a_desc_.get(),
        b_desc_.get(), c_desc_.get(), d_desc_.get(), preference.get(),
        max_algorithm_count, results.data(), &found_algorithm_count);
    if (error != 0) {
      VLOG(0) << "hipblasLtMatmulAlgoGetHeuristic returned " << (int)error;
      SE_HIPBLAS_RETURN_IF_ERROR(error);
    }
    results.resize(found_algorithm_count);
  }  // end mutex block

  std::vector<MatmulAlgorithm> algorithms;
  algorithms.reserve(results.size());
  for (const hipblasLtMatmulHeuristicResult_t& result : results) {
    if (result.state == HIPBLAS_STATUS_SUCCESS) {  // Skip failed algos.
      algorithms.push_back({result.algo, result.workspaceSize});
    }
  }
  return std::move(algorithms);
}

xla::Status BlasLt::MatmulPlan::SetAlgorithm(const MatmulAlgorithm& algorithm) {
  algorithm_ = algorithm;
  return xla::Status::OK();
}

auto BlasLt::GetMatmulPlan(const gpu::GemmConfig& cfg, Epilogue epilogue) const
    -> xla::StatusOr<MatmulPlanPtr> {
  auto lhs_layout = cfg.lhs_layout, rhs_layout = cfg.rhs_layout,
       output_layout = cfg.output_layout, c_layout = cfg.c_layout;

  // cublasLt matmul requires batch sizes to be equal. If only one operand has a
  // batch, the other will be broadcast (as its batch_stride == 0).
  size_t batch_size = std::max(lhs_layout.batch_size, rhs_layout.batch_size);
  lhs_layout.batch_size = batch_size;
  rhs_layout.batch_size = batch_size;

  bool must_swap_operands =
      MakeOutputColumnMajor(lhs_layout, rhs_layout, output_layout, &c_layout);

  // Do not transpose either input. Note the cuBLASLt documentation somewhat
  // incorrectly claims "A must be transposed and B non-transposed" when A and B
  // are FP8 (https://docs.nvidia.com/cuda/cublas/#cublasltmatmul). In reality,
  // this is only true if A and B are column-major. If A is row-major, A must
  // *not* be transposed, and if B is row-major, B must be transposed. We never
  // transpose A or B, and expect the caller to ensure A is row-major and B is
  // column when A and B are FP8.
  auto trans_a = lhs_layout.transpose, trans_b = rhs_layout.transpose;

  // if (xla::primitive_util::IsF8Type(lhs_layout.dtype) &&
  //     lhs_layout.order == gpu::MatrixLayout::Order::kColumnMajor) {
  //   return xla::Internal("The F8 LHS must be column-major");
  // }
  // if (xla::primitive_util::IsF8Type(rhs_layout.dtype) &&
  //     rhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
  //   return xla::Internal("The F8 RHS must be row-major");
  // }

  auto compute_type = cfg.compute_type;
  if (!compute_type) {  // obtain compute_type unless provided by the user
    TF_ASSIGN_OR_RETURN(compute_type, gpu::GetBlasComputationType(
        lhs_layout.dtype, output_layout.dtype, cfg.compute_precision));
  }

  if (lhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
    trans_a = blas::Transpose::kTranspose;
    lhs_layout.Transpose();
  }
  if (rhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
    trans_b = blas::Transpose::kTranspose;
    rhs_layout.Transpose();
  }

  TF_ASSIGN_OR_RETURN(
      auto op_desc,
      MatmulDesc::Create(*compute_type,
                         gpu::GetScaleType(output_layout.dtype, *compute_type),
                         trans_a, trans_b, epilogue));

  TF_ASSIGN_OR_RETURN(auto a_desc, MatrixLayout::Create(lhs_layout));
  TF_ASSIGN_OR_RETURN(auto b_desc, MatrixLayout::Create(rhs_layout));
  TF_ASSIGN_OR_RETURN(auto c_desc, MatrixLayout::Create(c_layout));
  TF_ASSIGN_OR_RETURN(auto d_desc, MatrixLayout::Create(output_layout));

  // std::make_unique won't work with brace initialization in C++17 ;(
  auto M = std::make_unique<MatmulPlan>(*this, std::move(op_desc),
                                      std::move(a_desc), std::move(b_desc),
                                      std::move(c_desc), std::move(d_desc),
                                      cfg.alpha, cfg.beta, must_swap_operands);

  return xla::StatusOr<MatmulPlanPtr>{std::move(M)};
}

xla::Status BlasLt::MatmulPlan::DoMatmul(
    Stream* stream, const void* alpha, DeviceMemoryBase a, DeviceMemoryBase b,
    const void* beta, DeviceMemoryBase c, DeviceMemoryBase d,
    DeviceMemoryBase bias,
    DeviceMemoryBase aux, DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
    DeviceMemoryBase c_scale, DeviceMemoryBase d_scale, DeviceMemoryBase d_amax,
    absl::optional<DeviceMemoryBase> workspace,
    absl::optional<ScratchAllocator *> allocator,
    blas::ProfileResult* profile_result) const {
  VLOG(1) << "BlasLt::MatmulPlan::DoMatmul";

  std::unique_ptr<gpu::GpuTimer, gpu::GpuTimerDeleter> timer;
  if (profile_result != nullptr) {
    timer.reset(new gpu::GpuTimer(blas_lt_ref_.parent_));
    if (!timer->Init() || !timer->Start(gpu::AsGpuStream(stream))) {
      return xla::InternalError("Unable to start gpu timer");
    }
  }

  if(!algorithm_.has_value()) return xla::InternalError("Algorithm is not set!");

  void* workspace_addr = nullptr;
  uint64_t workspace_size = 0;
  if (workspace.has_value()) {
    workspace_addr = workspace.value().opaque();
    workspace_size = workspace.value().size();
    TF_RET_CHECK(workspace_size >= algorithm_->workspace_size);
  } else if (algorithm_->workspace_size > 0) {
    
    if (!allocator || allocator.value() == nullptr) {
      return xla::InternalError("Allocator is not set: skipping solution!");
    }
    TF_ASSIGN_OR_RETURN(auto alloc,
        allocator.value()->AllocateBytes(algorithm_->workspace_size));
    workspace_addr = gpu::GpuMemoryMutable(&alloc);
    workspace_size = algorithm_->workspace_size;
  }

  auto palgo = absl::any_cast<hipblasLtMatmulAlgo_t>(&algorithm_->opaque_algo);
  {
    absl::MutexLock lock(&blas_lt_ref_.mu_);
    TF_RET_CHECK(blas_lt_ref_.blas_lt_ != nullptr);
    // We must set the bias and aux pointers while holding the mutex, to avoid a
    // potential race condition from multiple threads sharing the same plan.
    if (op_desc_.has_bias_epilogue() && bias != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(
          op_desc_.get(), HIPBLASLT_MATMUL_DESC_BIAS_POINTER, bias.opaque()));
    }

    if (a_scale != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                 a_scale.opaque()));
    }
    if (b_scale != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                 b_scale.opaque()));
    }
    if (c_scale != nullptr || d_scale != nullptr) {
      return xla::InternalError(
          "hipblaslt does not support c_scale or d_scale.");
    }

    if (d_amax != nullptr) {
      return xla::InternalError("hipblaslt does not support amax");
    }

    if (aux != nullptr) {
      return xla::InternalError(
          "hipblaslt does not support auxiliary inputs / outputs");
    }

    gpu::ScopedActivateExecutorContext sac{blas_lt_ref_.parent_};

    if (palgo != nullptr) {
      char str[129];
      sprintf(str, "hipblasLtMatmul %d %d %d %d %d %d",
        a_desc_.m_.num_rows, a_desc_.m_.num_cols,
        b_desc_.m_.num_rows, b_desc_.m_.num_cols,
        c_desc_.m_.num_rows, c_desc_.m_.num_cols);
      //hipStreamSynchronize(stream_executor::gpu::AsGpuStreamValue(stream));
      int t = stream_executor::gpu::launch_notify2(str, stream_executor::gpu::AsGpuStreamValue(stream));
      SE_HIPBLAS_RETURN_IF_ERROR(hipblasLtMatmul(
          blas_lt_ref_.blas_lt_.get(), op_desc_.get(), alpha, a.opaque(),
          a_desc_.get(), b.opaque(), b_desc_.get(), beta, c.opaque(),
          c_desc_.get(), d.opaque(), d_desc_.get(), palgo, workspace_addr,
          workspace_size, gpu::AsGpuStreamValue(stream)));

        uint32_t checksums[3];
        int inf_counts[3];

      if(false) {
        if (a_desc_.type() == HIP_R_16F) {
          const Eigen::half* pa = (const Eigen::half*)(a.opaque());
          const Eigen::half* pb = (const Eigen::half*)(b.opaque());
          const Eigen::half* pc = (const Eigen::half*)(c.opaque());
          checksums[0]=checksum(pa, a_desc_.m_.num_rows*a_desc_.m_.num_cols*a_desc_.m_.batch_size, inf_counts[0]); 
          checksums[1]=checksum(pb, b_desc_.m_.num_rows*b_desc_.m_.num_cols*b_desc_.m_.batch_size, inf_counts[1]); 
          checksums[2]=checksum(pc, c_desc_.m_.num_rows*c_desc_.m_.num_cols*c_desc_.m_.batch_size, inf_counts[2]);

          printf("Hipblaslt<half>: %p %p %p   %f %f %f %f -> %f %f  %08x %08x %08x   %d %d %d\n", 
          pa, pb, pc,  
            float(pa[0]), float(pa[1]), float(pb[0]), float(pb[1]), float(pc[0]), float(pc[1]),
            checksums[0], checksums[1], checksums[2], inf_counts[0], inf_counts[1], inf_counts[2]
            );
          fflush(stdout);
          if (!isfinite(float(pc[0])))
            exit(0);
        } else {
          const float* pa = (const float*)(a.opaque());
          const float* pb = (const float*)(b.opaque());
          const float* pc = (const float*)(c.opaque());
          checksums[0]=checksum(pa, a_desc_.m_.num_rows*a_desc_.m_.num_cols*a_desc_.m_.batch_size, inf_counts[0]); 
          checksums[1]=checksum(pb, b_desc_.m_.num_rows*b_desc_.m_.num_cols*b_desc_.m_.batch_size, inf_counts[1]); 
          checksums[2]=checksum(pc, c_desc_.m_.num_rows*c_desc_.m_.num_cols*c_desc_.m_.batch_size, inf_counts[2]);
          printf("Hipblaslt<float>: %p %p %p   %f %f %f %f -> %f %f  %08x %08x %08x   %d %d %d\n", 
          pa, pb, pc,  
            float(pa[0]), float(pa[1]), float(pb[0]), float(pb[1]), float(pc[0]), float(pc[1]),
            checksums[0], checksums[1], checksums[2], inf_counts[0], inf_counts[1], inf_counts[2]
            );
          fflush(stdout);
          if (!isfinite(float(pc[0])))
            exit(0);
        }
      }
      //hipStreamSynchronize(stream_executor::gpu::AsGpuStreamValue(stream));
      stream_executor::gpu::launch_notify_finish2(str, stream_executor::gpu::AsGpuStreamValue(stream), t);
    } else {
      return xla::InternalError("hipblaslt: Invalid algorithm type");
    }
  }

  if (profile_result != nullptr) {
    if (!timer->Stop(gpu::AsGpuStream(stream))) {
      return xla::InternalError("Unable to stop gpu timer");
    }
    // set algorithm ID to be unique (otherwise it gets kDefaultAlgorithm ID)
    auto roc_algo = (const rocblaslt_matmul_algo*)palgo;
    auto pindex = (int*)roc_algo->data;
    profile_result->set_algorithm(static_cast<blas::AlgorithmType>(*pindex));
    profile_result->set_is_valid(true);
    profile_result->set_elapsed_time_in_ms(timer->GetElapsedMilliseconds());
  }
  return xla::Status::OK();
}

xla::Status BlasLt::MatmulPlan::ExecuteOnStream(
    Stream* stream, DeviceMemoryBase a, DeviceMemoryBase b, DeviceMemoryBase c,
    DeviceMemoryBase d, DeviceMemoryBase bias, DeviceMemoryBase aux,
    DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
    DeviceMemoryBase c_scale, DeviceMemoryBase d_scale, DeviceMemoryBase d_amax,
    absl::optional<DeviceMemoryBase> workspace,
    absl::optional<ScratchAllocator*> scratch_allocator,
    blas::ProfileResult* profile_result) const {
  if (must_swap_operands_) {
    std::swap(a, b);
  }

  auto operand_types = std::make_tuple(
        a_desc_.type(), b_desc_.type(), c_desc_.type(), d_desc_.type());

#define TYPED_MATMUL(SCALENTYPE, ATYPE, BTYPE, CTYPE, DTYPE)               \
  if (operand_types == std::make_tuple(ATYPE, BTYPE, CTYPE, DTYPE)) {      \
    return gpu::BlasLt::MatmulPlan::DoMatmul< SCALENTYPE >(                \
        stream, alpha_, a, b, beta_, c, d, bias, aux, a_scale, b_scale,    \
        c_scale, d_scale, d_amax, workspace, scratch_allocator,            \
        profile_result);                                                   \
  }
  // Other data types:
  TYPED_MATMUL(float, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF)
  TYPED_MATMUL(float, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F)
  TYPED_MATMUL(float, HIP_R_16BF, HIP_R_16BF, HIP_R_32F, HIP_R_32F)
  TYPED_MATMUL(float, HIP_R_16F, HIP_R_16F, HIP_R_32F, HIP_R_32F)
  TYPED_MATMUL(float, HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F)
  TYPED_MATMUL(double, HIP_R_64F, HIP_R_64F, HIP_R_64F, HIP_R_64F)
  TYPED_MATMUL(complex64, HIP_C_32F, HIP_C_32F, HIP_C_32F, HIP_C_32F)
  TYPED_MATMUL(complex128, HIP_C_64F, HIP_C_64F, HIP_C_64F, HIP_C_64F)

#undef TYPED_MATMUL

  return xla::InternalError("Unexpected dtype");
}


BlasLt::GroupedMatmulPlan::GroupedMatmulPlan(const BlasLt& blas_lt) : 
        blas_lt_ref_(blas_lt) {}

BlasLt::GroupedMatmulPlan::~GroupedMatmulPlan() {
  if(host_args_ != nullptr) {
    blas_lt_ref_.parent_->HostMemoryDeallocate(host_args_);
  }
  if(!device_args_.is_null()) {
    blas_lt_ref_.parent_->Deallocate(&device_args_);
  }
}

auto BlasLt::GetGroupedMatmulPlan(Stream *stream, 
          const gpu::GroupedGemmConfig& cfg) const 
    -> xla::StatusOr<GroupedMatmulPlanPtr> {

  auto plan = std::make_unique< GroupedMatmulPlan >(*this);

  plan->grouped_gemm_ = std::make_unique< GroupedGemm >(blas_lt_.get(),
          AsHipblasOperation(cfg.trans_a),
          AsHipblasOperation(cfg.trans_b),
          AsHipblasDataType(cfg.type_a),
          AsHipblasDataType(cfg.type_b),
          AsHipblasDataType(cfg.type_c),
          AsHipblasDataType(cfg.type_d),
          AsHipblasComputeType(cfg.compute_type));
  auto& ggemm = plan->grouped_gemm_;
  
  std::vector< int64_t > m(cfg.batch_count, cfg.m), 
                         n(cfg.batch_count, cfg.n), 
                         k(cfg.batch_count, cfg.k), 
                         batch_count(cfg.batch_count, 1), 
                         lda(cfg.batch_count, cfg.lda),
                         ldb(cfg.batch_count, cfg.ldb),
                         ldc(cfg.batch_count, cfg.ldc),
                         ldd(cfg.batch_count, cfg.ldd),
                         strideA(cfg.batch_count, cfg.m * cfg.k),
                         strideB(cfg.batch_count, cfg.n * cfg.k),
                         strideC(cfg.batch_count, cfg.m * cfg.n),
                         strideD(cfg.batch_count, cfg.m * cfg.n);

  std::vector< GemmEpilogue > epilogue(cfg.batch_count,
            GemmEpilogue{});
  std::vector< GemmInputs > inputs(cfg.batch_count);
  for(int64 i = 0; i < cfg.batch_count; i++) {
    inputs[i].a = const_cast< void * >(cfg.a[i]);
    inputs[i].b = const_cast< void * >(cfg.b[i]);
    inputs[i].c = const_cast< void * >(cfg.c[i]);
    inputs[i].d = cfg.d[i];
    inputs[i].alpha = const_cast< void * >(cfg.alpha);
    inputs[i].beta = const_cast< void * >(cfg.beta);
  }

  GemmProblemType problem = {
    .op_a = AsHipblasOperation(cfg.trans_a),
    .op_b = AsHipblasOperation(cfg.trans_b),
    .type_a = AsHipblasDataType(cfg.type_a),
    .type_b = AsHipblasDataType(cfg.type_b),
    .type_c = AsHipblasDataType(cfg.type_c),
    .type_d = AsHipblasDataType(cfg.type_d),
    .type_compute = AsHipblasComputeType(cfg.compute_type)
  };

  uint64 mem_size = cfg.batch_count * sizeof(UserArguments);
  {
    absl::MutexLock lock(&mu_);
    SE_HIPBLAS_RETURN_IF_ERROR(ggemm->setProblem(m, n, k, batch_count,
          lda, ldb, ldc, ldd, strideA, strideB, strideC, strideD,
          epilogue, inputs, problem));

    plan->host_args_ = static_cast< UserArguments *>(
          parent_->HostMemoryAllocate(mem_size));
    if(plan->host_args_ == nullptr) {
      return xla::InternalError("Unable to allocate host memory for user args!");
    }
    SE_HIPBLAS_RETURN_IF_ERROR(ggemm->
          getDefaultValueForDeviceUserArguments(plan->host_args_));

    // NOTE: memory must be aligned by 16 bytes ??
    auto raw_mem = parent_->Allocate(mem_size);
    // TF_ASSIGN_OR_RETURN(auto dev_mem, allocator->Allocate(parent_->device_ordinal(), 
    //       mem_size)));
    if(raw_mem == nullptr) {
      return xla::InternalError("Unable to allocate memory for grouped gemm params!");
    }
    plan->device_args_ = GroupedMatmulPlan::DeviceMemoryArgs{raw_mem, mem_size};

    if(!stream->ThenMemcpy(&plan->device_args_, plan->host_args_, mem_size).ok()) {
       return xla::InternalError("Memcpy failed!");
    }

  } // end block

  //for(const auto& a : plan->host_args_) 
  {
    // const auto& a = plan->host_args_[0];
    // std::ostringstream os;
    // for(int i = 0; i < sizeof(a.alpha); i++) {
    //   os << std::hex << (uint32_t)a.alpha[i];
    // }
    // VLOG(0) << a.m << "," << a.n << "," << a.batch << "," << a.k <<
    //   " alpha " << os.str() <<
    //   " pointers: " << a.d << "," << a.c << "," << a.a << "," << a.b <<
    //   " strides: " << a.strideD1 << "," << a.strideD2 << "," << a.strideA1 << "," << a.strideA2 <<
    //   " activate: " << a.activationType;
  }
  return xla::StatusOr<GroupedMatmulPlanPtr>(std::move(plan));
}

auto BlasLt::GroupedMatmulPlan::GetAlgorithms(
        size_t max_algorithm_count, size_t max_workspace_size) ->
    xla::StatusOr<std::vector<MatmulAlgorithm>> {

  // GemmPreference gemmPref;
  // gemmPref.setMaxWorkspaceBytes(max_workspace_size);
  
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
  std::vector<MatmulAlgorithm> algorithms;

  gpu::ScopedActivateExecutorContext sac{blas_lt_ref_.parent_};
  absl::MutexLock lock(&blas_lt_ref_.mu_);
  
  auto problem = grouped_gemm_->getProblemTypes()[0];
  // VLOG(0) << problem.op_a <<","<<
  //                           problem.op_b<<","<<
  //                           problem.type_a<<","<<
  //                           problem.type_b<<","<<
  //                           problem.type_c<<","<<
  //                           problem.type_d<<","<<
  //                           problem.type_compute;
    // HIPBLAS_OP_N = 111, /**<  Operate with the matrix. */
    // HIPBLAS_OP_T = 112, /**<  Operate with the transpose of the matrix. */
    // HIPBLAS_OP_C = 113 /**< Operate with the conjugate transpose of the matrix. */

  SE_HIPBLAS_RETURN_IF_ERROR(getAllAlgos(blas_lt_ref_.blas_lt_.get(),
                                       GemmType::HIPBLASLT_GROUPED_GEMM,
                                       problem.op_a,
                                       problem.op_b,
                                       problem.type_a,
                                       problem.type_b,
                                       problem.type_c,
                                       problem.type_d,
                                       problem.type_compute,
                                       heuristicResult));
  // SE_HIPBLAS_RETURN_IF_ERROR(
  //       grouped_gemm_->algoGetHeuristic(max_algorithm_count, gemmPref, 
  //             heuristicResult));
  VLOG(2) << "Total heuristics found: " << heuristicResult.size();
  algorithms.reserve(heuristicResult.size());
  for(auto& res : heuristicResult) {
    size_t workspace_size = 0;
    if(grouped_gemm_->isAlgoSupported(res.algo, workspace_size)) {
      algorithms.push_back({res.algo, workspace_size});
    }
  }
  return algorithms;
}

xla::Status BlasLt::GroupedMatmulPlan::SetAlgorithm(
            const MatmulAlgorithm& algorithm, 
            ScratchAllocator *allocator) 
{
  auto palgo = absl::any_cast<hipblasLtMatmulAlgo_t>(&algorithm.opaque_algo);
  if(palgo == nullptr) {
    return xla::InternalError("Wrong algorithm instance !");
  }
  algorithm_ = algorithm;

  void* workspace_addr = nullptr;
  uint64_t workspace_size = algorithm_->workspace_size;
  if (workspace_size > 0) {
    if (allocator == nullptr) {
      return xla::InternalError("This algorithm requires a non-zero workspace!");
    }
    TF_ASSIGN_OR_RETURN(auto alloc, allocator->AllocateBytes(workspace_size));
    workspace_addr = gpu::GpuMemoryMutable(&alloc);
  }

  gpu::ScopedActivateExecutorContext sac{blas_lt_ref_.parent_}; 
  absl::MutexLock lock(&blas_lt_ref_.mu_);
  // NOTE NOTE: it could be that workspace is no longer valid after
  // this function returns !!!!
  SE_HIPBLAS_RETURN_IF_ERROR(grouped_gemm_->initialize(
          *palgo, workspace_addr));
  return xla::Status::OK();
}

xla::Status BlasLt::GroupedMatmulPlan::ExecuteOnStream(Stream *stream,
          const gpu::GroupedGemmConfig& cfg, 
          blas::ProfileResult* profile_result) {
  
  if((size_t)cfg.batch_count * sizeof(UserArguments) != device_args_.size() || 
      !algorithm_.has_value())
  {
    return xla::InternalError("GroupedGemm config mismatch or algorithm is unset!");
  }

  std::unique_ptr<gpu::GpuTimer, gpu::GpuTimerDeleter> timer;
  if (profile_result != nullptr) {
    timer.reset(new gpu::GpuTimer(blas_lt_ref_.parent_));
    if (!timer->Init() || !timer->Start(gpu::AsGpuStream(stream))) {
      return xla::InternalError("Unable to start gpu timer");
    }
  }

  // VLOG(0) << "Cmp ptrs: " << host_args_[0].a << "," <<
  //     host_args_[0].b << "," << host_args_[0].c << " vs " <<
  //     cfg.a[0] << "," << cfg.b[0] << "," << cfg.c[0];

  // NOTE: we can also use GPU kernel to update pointers directly 
  // in device mem => then memcpy won't be necessary
  // for(size_t i = 0; i < cfg.batch_count; i++) {
  //   host_args_[i].a = const_cast< void * >(cfg.a[i]);
  //   host_args_[i].b = const_cast< void * >(cfg.b[i]);
  //   host_args_[i].c = const_cast< void * >(cfg.c[i]);
  //   host_args_[i].d = const_cast< void * >(cfg.d[i]);
  // }
  //TF_RETURN_IF_ERROR(UpdateArgs(stream, cfg));
  GroupGemmUpdateArgs(gpu::AsGpuStreamValue(stream), 
        static_cast<UserArguments *>(device_args_.opaque()),
        cfg.a, cfg.b, cfg.c, cfg.d,
        cfg.batch_count);

  gpu::ScopedActivateExecutorContext sac{blas_lt_ref_.parent_}; 
  {
    
  absl::MutexLock lock(&blas_lt_ref_.mu_);
  
  SE_HIPBLAS_RETURN_IF_ERROR(grouped_gemm_->run(
        device_args_.opaque(), gpu::AsGpuStreamValue(stream)));
  } // end block

  if (profile_result != nullptr) {
    if (!timer->Stop(gpu::AsGpuStream(stream))) {
      return xla::InternalError("Unable to stop gpu timer");
    }
    // algorithm_ is alrady verified for correctness !
    auto palgo = absl::any_cast<hipblasLtMatmulAlgo_t>(&algorithm_->opaque_algo);
    // set algorithm ID to be unique (otherwise it gets kDefaultAlgorithm ID)
    auto roc_algo = (const rocblaslt_matmul_algo*)palgo;
    auto pindex = (int*)roc_algo->data;
    profile_result->set_algorithm(static_cast<blas::AlgorithmType>(*pindex));
    profile_result->set_is_valid(true);
    profile_result->set_elapsed_time_in_ms(timer->GetElapsedMilliseconds());
  }
  return xla::Status::OK();
}

}  // namespace rocm

}  // namespace stream_executor

