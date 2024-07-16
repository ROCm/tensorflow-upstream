/* Copyright 2022 The OpenXLA Authors.

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

#ifndef TENSORFLOW_COMPILER_SERVICE_GPU_GPUBLAS_LT_MATMUL_THUNK_H_
#define TENSORFLOW_COMPILER_SERVICE_GPU_GPUBLAS_LT_MATMUL_THUNK_H_

#include <cstdint>
#include <optional>

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/gpu/gpu_blas_lt.h"
#include "tensorflow/stream_executor/blas.h"

namespace xla {
namespace gpu {

struct GemmConfig;

class CublasLtMatmulThunk : public Thunk {
 public:
  CublasLtMatmulThunk(
      const HloInstruction *hlo_instruction,
      GemmBackendConfig&& backend_config,
      se::gpu::BlasLt::Epilogue epilogue, int64_t algorithm_idx,
      BufferAllocation::Slice a_buffer, BufferAllocation::Slice b_buffer,
      BufferAllocation::Slice c_buffer, BufferAllocation::Slice d_buffer,
      BufferAllocation::Slice bias_buffer /* may be null */,
      BufferAllocation::Slice aux_buffer /* may be null */,
      BufferAllocation::Slice a_scale_buffer /* may be null */,
      BufferAllocation::Slice b_scale_buffer /* may be null */,
      BufferAllocation::Slice c_scale_buffer /* may be null */,
      BufferAllocation::Slice d_scale_buffer /* may be null */,
      BufferAllocation::Slice d_amax_buffer /* may be null */,
      absl::optional<const BufferAllocation::Slice> workspace_buffer);

  Status ExecuteOnStream(const ExecuteParams& params) override;
  Status Initialize(const GpuExecutable& executable,
                            se::StreamExecutor* executor) override;

 private:
  StatusOr<se::gpu::BlasLt::MatmulPlan*> GetMatmulPlan(
      const stream_executor::Stream* stream);
  StatusOr<se::gpu::BlasLt::MatmulAlgorithm> GetMatmulAlgorithm(
      const se::gpu::BlasLt::MatmulPlan* plan, int64_t max_workspace);

  absl::Mutex matmul_plans_cache_mutex_;
  absl::flat_hash_map<const stream_executor::Stream*,
                      se::gpu::BlasLt::MatmulPlanPtr>
      matmul_plans_cache_ ABSL_GUARDED_BY(matmul_plans_cache_mutex_);

  absl::Mutex matmul_algorithm_cache_mutex_;
  absl::flat_hash_map<const se::gpu::BlasLt::MatmulPlan*,
                      se::gpu::BlasLt::MatmulAlgorithm>
      matmul_algorithm_cache_ ABSL_GUARDED_BY(matmul_algorithm_cache_mutex_);

  GemmBackendConfig backend_config_;
  //std::unique_ptr< GemmConfig > gemm_config_ptr_; // TODO!
  se::gpu::BlasLt::Epilogue epilogue_;
  int64_t algorithm_idx_;
  BufferAllocation::Slice a_buffer_;
  BufferAllocation::Slice b_buffer_;
  BufferAllocation::Slice c_buffer_;
  BufferAllocation::Slice d_buffer_;
  BufferAllocation::Slice bias_buffer_;
  BufferAllocation::Slice aux_buffer_;
  BufferAllocation::Slice a_scale_buffer_;
  BufferAllocation::Slice b_scale_buffer_;
  BufferAllocation::Slice c_scale_buffer_;
  BufferAllocation::Slice d_scale_buffer_;
  BufferAllocation::Slice d_amax_buffer_;
  absl::optional<const BufferAllocation::Slice> workspace_buffer_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_SERVICE_GPU_RUNTIME_GPUBLAS_LT_MATMUL_THUNK_H_
