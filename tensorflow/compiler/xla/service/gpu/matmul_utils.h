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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MATMUL_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MATMUL_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/stream_executor/gpu/gpu_blas_lt.h"
#include "tensorflow/stream_executor/blas.h"
//#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Ordered non-contracting dimensions for a dot instruction operand.
StatusOr<std::vector<int64>> GetNonContractingDims(
    const Shape& shape, absl::Span<const int64> batch_dims,
    absl::Span<const int64> contracting_dims);

// Normalize shape to (batch, rows, columns) logical dimensions.
StatusOr<Shape> GetBatchRowColumnShape(
    const Shape& shape, absl::Span<const int64> batch_dims,
    absl::Span<const int64> row_dims, absl::Span<const int64> col_dims);

// Returns true if the backend can lower the dot. Currently the classical
// emitters cannot handle some dots, e.g., i8[] x i8[] -> i32[] dots,
// so we need to always use cuBLAS or Triton for those.
bool IsDotSupportedByClassicalEmitters(const HloInstruction& dot);

// extending plain MatrixLayout struct with creator functions
struct MatrixLayout : public se::gpu::MatrixLayout {

  MatrixLayout(xla::PrimitiveType dtype_, int64 num_rows_, int64 num_cols_,
               Order order_, int64 batch_size_ = 1,
               absl::optional<int64> leading_dim_stride_ = {},
               absl::optional<int64> batch_stride_ = {},
               absl::optional<se::blas::Transpose> transpose_ = {}) :
    se::gpu::MatrixLayout(dtype_, num_rows_, num_cols_,
               order_, batch_size_, leading_dim_stride_,
               batch_stride_, transpose_) {}

  // Returns the matrix layout for a logical shape (batch, rows, columns).
  static StatusOr<MatrixLayout> For(const Shape& shape);
  // Returns the matrix layout with the given batch, row, col dimensions.
  static StatusOr<MatrixLayout> For(const Shape& shape,
                                          absl::Span<const int64> batch_dims,
                                          absl::Span<const int64> row_dims,
                                          absl::Span<const int64> col_dims);
  // Returns the matrix layout for the output.
  static StatusOr<MatrixLayout> For(const Shape& shape,
                                          size_t lhs_num_batch_dims,
                                          size_t lhs_num_row_dims,
                                          size_t rhs_num_batch_dims,
                                          size_t rhs_num_col_dims);
};

struct GemmConfig : public se::gpu::GemmConfig {
  // For legacy Gemm operations XLA:GPU allocates its own workspace and passes
  // it to all BLAS API calls.
  //
  // Size of the workspace based on NVIDIA recommendation:
  // https://docs.nvidia.com/cuda/cublas/#cublassetworkspace
  static constexpr int64 kHopperWorkspace = 32 * 1024 * 1024;  // 32 MiB
  static constexpr int64 kDefaultWorkspace = 8 * 1024 * 1024;  // 16 MiB
  static constexpr int64 kMaxCublasLtAlgorithms = 512; 

  explicit GemmConfig(se::gpu::GemmConfig&& cfg) : 
    se::gpu::GemmConfig(std::move(cfg)) { }

  static StatusOr<GemmConfig> For(const HloInstruction* gemm,
        const GemmBackendConfig& config);

  static StatusOr<GemmConfig> For(
      const Shape& lhs_shape, absl::Span<const int64> lhs_batch_dims,
      absl::Span<const int64> lhs_contracting_dims, const Shape& rhs_shape,
      absl::Span<const int64> rhs_batch_dims,
      absl::Span<const int64> rhs_contracting_dims, const Shape& output_shape,
      double alpha_real, double alpha_imag, double beta,
      absl::optional<int64> algorithm, int64 compute_precision, bool grad_x,
      bool grad_y);

  struct DescriptorsTuple {
    se::gpu::MatrixDescriptor lhs;
    se::gpu::MatrixDescriptor rhs;
    se::gpu::OutputMatrixDescriptor output;
    bool operands_swapped;
  };
  StatusOr<DescriptorsTuple> GetMatrixDescriptors(
      se::DeviceMemoryBase lhs_buf, se::DeviceMemoryBase rhs_buf,
      se::DeviceMemoryBase out_buf) const;
};

// Run the given GEMM instruction `gemm` subject to the configuration
// in `gemm_config` and the passed buffers.
//
// If `algorithm` is provided, it overrides the one specified in `config`.
Status RunGemm(
    const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
    se::DeviceMemoryBase rhs_buffer, se::DeviceMemoryBase output_buffer,
    se::DeviceMemoryBase workspace_buffer, bool deterministic_ops,
    se::Stream* stream,
    absl::optional<se::blas::AlgorithmType> algorithm = absl::nullopt,
    se::blas::ProfileResult* profile_result = nullptr);

namespace gpublas_lt {

StatusOr<bool> EpilogueAddsVectorBias(
    GemmBackendConfig::Epilogue epilogue);
StatusOr<bool> EpilogueHasAuxiliaryOutput(
    GemmBackendConfig::Epilogue epilogue);

StatusOr<se::gpu::BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig::Epilogue epilogue);

}  // namespace gpublas_lt


}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MATMUL_UTILS_H_
