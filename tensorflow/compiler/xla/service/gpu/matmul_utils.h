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

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_blas_lt.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Ordered non-contracting dimensions for a dot instruction operand.
StatusOr<std::vector<int64_t>> GetNonContractingDims(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> contracting_dims);

// Batch dimensions of an operand of a dot instruction.
// Just an unified accessor to lhs_batch_dimensions and rhs_batch_dimensions.
const tsl::protobuf::RepeatedField<int64_t>& BatchDimensionsForOperand(
    const HloInstruction& dot, int operand_number);

// Index of the only contracting dimension of dot instruction operand.
StatusOr<int64_t> ContractingDimensionIndex(const HloInstruction& dot,
                                                  int operand_number);

// Index of the only non-contracting dimension of dot instruction operand.
StatusOr<int64_t> NonContractingDimensionIndex(const HloInstruction& dot,
                                                     int operand_number);

// Normalize shape to (batch, rows, columns) logical dimensions.
StatusOr<Shape> GetBatchRowColumnShape(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> row_dims, absl::Span<const int64_t> col_dims);

// GPU folding rule for the `TransposeFolding` pass.
StatusOr<bool> CanFoldTransposeOperandIntoDot(const HloInstruction& dot,
                                                    int64_t operand_idx);

// Returns true if the sum of the sizes of the unbatched operand matrices
// for the dot is smaller than the given threshold.
StatusOr<bool> IsMatrixMultiplicationTooSmallForRewriting(
    const HloInstruction& dot, int64_t threshold);

// Returns true if the backend can lower the dot. Currently the classical
// emitters cannot handle some dots, e.g., i8[] x i8[] -> i32[] dots,
// so we need to always use cuBLAS or Triton for those.
bool IsDotSupportedByClassicalEmitters(const HloInstruction& dot);

// extending plain MatrixLayout struct with creator functions
struct MatrixLayout : public se::gpu::MatrixLayout {

  MatrixLayout(se::blas::DataType dtype_, int64_t num_rows_, int64_t num_cols_,
               Order order_, int64_t batch_size_ = 1,
               absl::optional<int64_t> leading_dim_stride_ = {},
               absl::optional<int64_t> batch_stride_ = {},
               absl::optional<se::blas::Transpose> transpose_ = {}) :
    se::gpu::MatrixLayout(dtype_, num_rows_, num_cols_,
               order_, batch_size_, leading_dim_stride_,
               batch_stride_, transpose_) {}

  // Returns the matrix layout for a logical shape (batch, rows, columns).
  static StatusOr<MatrixLayout> For(const Shape& shape);
  // Returns the matrix layout with the given batch, row, col dimensions.
  static StatusOr<MatrixLayout> For(const Shape& shape,
                                          absl::Span<const int64_t> batch_dims,
                                          absl::Span<const int64_t> row_dims,
                                          absl::Span<const int64_t> col_dims);
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
  static constexpr int64_t kHopperWorkspace = 32 * 1024 * 1024;  // 32 MiB
  static constexpr int64_t kDefaultWorkspace = 4 * 1024 * 1024;  // 4 MiB

  explicit GemmConfig(const se::gpu::GemmConfig& cfg) : 
    se::gpu::GemmConfig(cfg) { }

  static StatusOr<GemmConfig> For(const HloInstruction* gemm);
  static StatusOr<GemmConfig> For(mlir::lmhlo_gpu::GEMMOp op);
  static StatusOr<GemmConfig> For(mlir::lmhlo_gpu::CublasLtMatmulOp op);

  static StatusOr<GemmConfig> For(
      const Shape& lhs_shape, absl::Span<const int64_t> lhs_batch_dims,
      absl::Span<const int64_t> lhs_contracting_dims, const Shape& rhs_shape,
      absl::Span<const int64_t> rhs_batch_dims,
      absl::Span<const int64_t> rhs_contracting_dims, const Shape& output_shape,
      double alpha_real, double alpha_imag, double beta,
      int64_t algorithm, int64_t compute_precision, 
      se::gpu::BlasLt::Epilogue epilogue);

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
    std::optional<se::blas::AlgorithmType> algorithm = std::nullopt,
    se::blas::ProfileResult* profile_result = nullptr);

namespace gpublas_lt {

StatusOr<bool> EpilogueAddsVectorBias(
    GemmBackendConfig_Epilogue epilogue);
StatusOr<bool> EpilogueHasAuxiliaryOutput(
    GemmBackendConfig_Epilogue epilogue);

StatusOr<se::gpu::BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue);
StatusOr<se::gpu::BlasLt::Epilogue> AsBlasLtEpilogue(
    mlir::lmhlo_gpu::CublasLtMatmulEpilogue epilogue);

}  // namespace gpublas_lt

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MATMUL_UTILS_H_
