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
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {
namespace gpu {

class CublasLtMatmulThunk : public Thunk {
 public:

  CublasLtMatmulThunk(
      ThunkInfo thunk_info, GemmConfig config,
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
  StatusOr<se::gpu::BlasLt::MatmulPlan *> GetCachedMatmulPlan(
                const ExecuteParams& params);

  GemmConfig gemm_config_;
  std::string canonical_hlo_;
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

