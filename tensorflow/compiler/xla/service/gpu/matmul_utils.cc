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

#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace gpu {

StatusOr<std::vector<int64>> GetNonContractingDims(
    const Shape& shape, absl::Span<const int64> batch_dims,
    absl::Span<const int64> contracting_dims) {
  std::vector<int64> non_contracting_dims;
  // This is O(rank**2), but we expect rank to be small.
  for (int64 dim = 0; dim < shape.rank(); ++dim) {
    bool is_batch = absl::c_count(batch_dims, dim) != 0;
    bool is_contracting = absl::c_count(contracting_dims, dim) != 0;
    TF_RET_CHECK(!(is_batch && is_contracting));
    if (!(is_batch || is_contracting)) non_contracting_dims.push_back(dim);
  }

  TF_RET_CHECK(batch_dims.size() + contracting_dims.size() +
                   non_contracting_dims.size() ==
               shape.rank());
  return non_contracting_dims;
}

StatusOr<Shape> GetBatchRowColumnShape(
    const Shape& shape, absl::Span<const int64> batch_dims,
    absl::Span<const int64> row_dims, absl::Span<const int64> col_dims) {
  TF_RET_CHECK(shape.has_layout());

  std::vector<int64> minor_to_major;
  for (size_t i = 0; i < shape.rank();) {
    // The GeMM output always has its layout set such that the batch, row, and
    // col dim groups are each laid out physically sequentially. GeMM operands
    // must, therefore, be laid out similarly.
    auto check_physically_sequential =
        [&](absl::Span<const int64> dims) -> Status {
      for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
        // NOTE: `i` is incremented as we check the dimensions.
        if (*it != shape.layout().minor_to_major()[i++])
          return InvalidArgument("dims not physically_sequential");
      }
      return Status::OK();
    };

    int64 dim = shape.layout().minor_to_major()[i];
    if (!row_dims.empty() && dim == row_dims.back()) {
      minor_to_major.push_back(1);
      TF_RETURN_IF_ERROR(check_physically_sequential(row_dims));
    } else if (!col_dims.empty() && dim == col_dims.back()) {
      minor_to_major.push_back(2);
      TF_RETURN_IF_ERROR(check_physically_sequential(col_dims));
    } else if (!batch_dims.empty() && (dim == batch_dims.back())) {
      minor_to_major.push_back(0);
      TF_RETURN_IF_ERROR(check_physically_sequential(batch_dims));
    } else {
      return InvalidArgument("dims not physically sequential");
    }
  }

  if (col_dims.empty()) minor_to_major.push_back(2);
  if (row_dims.empty()) minor_to_major.push_back(1);
  if (batch_dims.empty()) minor_to_major.push_back(0);

  auto dim_size = [&](absl::Span<const int64> dims) {
    return absl::c_accumulate(dims, 1, [&](int64 size, int64 dim) {
      return size * shape.dimensions(dim);
    });
  };
  // absl::Span< const int64 > UU(minor_to_major);
  return ShapeUtil::MakeShapeWithLayout(
      shape.element_type(),
      {dim_size(batch_dims), dim_size(row_dims), dim_size(col_dims)},
      minor_to_major);
}

// Returns the matrix layout for a logical shape (batch, rows, columns).
/*static*/ StatusOr<MatrixLayout> MatrixLayout::For(const Shape& shape) {
  TF_RET_CHECK(shape.rank() == 3);
  TF_RET_CHECK(shape.has_layout());

  int64 batch_size = shape.dimensions(0);
  int64 num_rows = shape.dimensions(1);
  int64 num_cols = shape.dimensions(2);

  Order order{Order::kRowMajor};
  int64 leading_dim_stride = num_cols;
  int64 batch_stride = num_rows * num_cols;

  // `MatrixLayout`, like BLAS, uses only two strides, so either the row or
  // column must be contiguous in memory (i.e. most minor physical dimension).
  absl::Span<const int64> minor_to_major = shape.layout().minor_to_major();
  switch (64 * minor_to_major[2] + 8 * minor_to_major[1] + minor_to_major[0]) {
    case 012:  // (B,R,C) (major-to-minor)
      break;
    case 021:  // (B,C,R)
      order = Order::kColumnMajor;
      leading_dim_stride = num_rows;
      break;
    case 0102:  // (R,B,C)
      leading_dim_stride = batch_size * num_cols;
      batch_stride = num_cols;
      break;
    case 0201:  // (C,B,R)
      order = Order::kColumnMajor;
      leading_dim_stride = batch_size * num_rows;
      batch_stride = num_rows;
      break;
    default:
      return Unimplemented("batch in most minor dimension");
  }

  if (batch_size == 1) {
    batch_stride = 0;
  }

  TF_ASSIGN_OR_RETURN(auto dtype, se::gpu::AsBlasDataType(shape.element_type()));
  return MatrixLayout(dtype, num_rows, num_cols, order, 
        batch_size, leading_dim_stride, batch_stride);
}

/*static*/ StatusOr<MatrixLayout> MatrixLayout::For(
    const Shape& shape, absl::Span<const int64> batch_dims,
    absl::Span<const int64> row_dims, absl::Span<const int64> col_dims) {
  TF_ASSIGN_OR_RETURN(
      Shape batch_row_col_shape,
      GetBatchRowColumnShape(shape, batch_dims, row_dims, col_dims));
  return MatrixLayout::For(batch_row_col_shape);
}

/*static*/ StatusOr<MatrixLayout> MatrixLayout::For(
    const Shape& shape, size_t lhs_num_batch_dims, size_t lhs_num_row_dims,
    size_t rhs_num_batch_dims, size_t rhs_num_col_dims) {
  size_t num_batch_dims = std::max(lhs_num_batch_dims, rhs_num_batch_dims);

  TF_RET_CHECK(shape.rank() ==
               num_batch_dims + lhs_num_row_dims + rhs_num_col_dims);

  std::vector<int64> dims(shape.rank());
  absl::c_iota(dims, 0);

  auto batch_dims = absl::Span<const int64>(dims).first(num_batch_dims);
  auto row_dims =
      absl::Span<const int64>(dims).subspan(num_batch_dims, lhs_num_row_dims);
  auto col_dims = absl::Span<const int64>(dims).last(rhs_num_col_dims);

  return MatrixLayout::For(shape, batch_dims, row_dims, col_dims);
}

namespace {
// Returns the relative order of 'dims' as indices from 0 to dims.size() - 1.
// Let 'indices' be the returned vector, then it holds that
// dims[indices[i - 1]] < dims[indices[i]] for 0 < i < dims.size()
std::vector<int64> NormalizedRelativeOrder(absl::Span<const int64> dims) {
  // Remap the dimensions to values between 0 and dims.size() - 1, keeping their
  // relative order the same.
  std::vector<int64> indices(dims.size());
  absl::c_iota(indices, 0);
  absl::c_sort(indices,
               [&](int64 a, int64 b) { return dims[a] < dims[b]; });
  return indices;
}
}  // namespace

/*static*/ StatusOr<GemmConfig> GemmConfig::For(
    const Shape& lhs_shape, absl::Span<const int64> lhs_batch_dims,
    absl::Span<const int64> lhs_contracting_dims, const Shape& rhs_shape,
    absl::Span<const int64> rhs_batch_dims,
    absl::Span<const int64> rhs_contracting_dims, const Shape& output_shape,
    double alpha_real, double alpha_imag, double beta,
    absl::optional<int64> algorithm, int64 compute_precision, bool grad_x,
    bool grad_y) {

  absl::Span<const int64> lhs_col_dims = lhs_contracting_dims;
  TF_ASSIGN_OR_RETURN(
      std::vector<int64> lhs_row_dims,
      GetNonContractingDims(lhs_shape, lhs_batch_dims, lhs_col_dims));

  TF_ASSIGN_OR_RETURN(
      MatrixLayout lhs_layout,
      MatrixLayout::For(lhs_shape, lhs_batch_dims, lhs_row_dims, lhs_col_dims));

  absl::Span<const int64> rhs_row_dims = rhs_contracting_dims;
  TF_ASSIGN_OR_RETURN(
      std::vector<int64> rhs_col_dims,
      GetNonContractingDims(rhs_shape, rhs_batch_dims, rhs_row_dims));

  TF_ASSIGN_OR_RETURN(
      MatrixLayout rhs_layout,
      MatrixLayout::For(rhs_shape, rhs_batch_dims, rhs_row_dims, rhs_col_dims));

  int64 num_batch_dims =
      std::max(lhs_batch_dims.size(), rhs_batch_dims.size());

  TF_RET_CHECK(output_shape.rank() ==
               num_batch_dims + lhs_row_dims.size() + rhs_col_dims.size());

  std::vector<int64> output_dims(output_shape.rank());
  absl::c_iota(output_dims, 0);

  auto output_batch_dims =
      absl::Span<const int64>(output_dims).first(num_batch_dims);
  auto output_row_dims = absl::Span<const int64>(output_dims)
                             .subspan(num_batch_dims, lhs_row_dims.size());
  auto output_col_dims =
      absl::Span<const int64>(output_dims).last(rhs_col_dims.size());

  TF_ASSIGN_OR_RETURN(MatrixLayout output_layout,
                      MatrixLayout::For(output_shape, output_batch_dims,
                                        output_row_dims, output_col_dims));
  Shape c_matrix_shape = output_shape;

  TF_ASSIGN_OR_RETURN(MatrixLayout c_layout,
                      MatrixLayout::For(c_matrix_shape, output_batch_dims,
                                        output_row_dims, output_col_dims));

  // TODO(cjfj): We should also check that the batch, contracting and
  // non-contracting dimensions match in size and relative physical location.
  // TODO(philipphack): Check the remaining dimensions in the FP8 case once
  // cuBLASLt supports the NN configuration.
  TF_RET_CHECK(lhs_layout.num_cols == rhs_layout.num_rows);
  TF_RET_CHECK(output_layout.num_rows == lhs_layout.num_rows);
  TF_RET_CHECK(output_layout.num_cols == rhs_layout.num_cols);
  TF_RET_CHECK(c_layout.num_rows == output_layout.num_rows);
  TF_RET_CHECK(c_layout.num_cols == output_layout.num_cols);
  TF_RET_CHECK((lhs_layout.batch_size == output_layout.batch_size) ||
               (lhs_layout.batch_size == 1));
  TF_RET_CHECK((rhs_layout.batch_size == output_layout.batch_size) ||
               (rhs_layout.batch_size == 1));

  switch (output_shape.element_type()) {
    case F16:
    case BF16:
    case F32:
    case F64:
      TF_RET_CHECK(alpha_imag == 0);
      break;
    case C64:
    case C128:
      break;
    case S32:
      TF_RET_CHECK(alpha_imag == 0);
      if (lhs_layout.dtype != se::blas::DataType::kInt8 ||
          rhs_layout.dtype != se::blas::DataType::kInt8) {
        return Internal(
            "For int32 gemm output only int8 input is supported !");
      }
      break;
    default:
      return Internal("Unexpected GEMM datatype: %s",
                      primitive_util::LowercasePrimitiveTypeName(
                          output_shape.element_type()));
  }

  return GemmConfig(
      se::gpu::GemmConfig{
                    lhs_layout,
                    rhs_layout,
                    c_layout,
                    output_layout,
                    {alpha_real, alpha_imag},
                    beta,
                    compute_precision,
                    algorithm,
                    grad_x,
                    grad_y});
}

/*static*/ StatusOr<GemmConfig> GemmConfig::For(
    const HloInstruction* gemm, const GemmBackendConfig& config) {

  absl::optional<int64> algorithm;
  if (config.algorithm_case() != GemmBackendConfig::ALGORITHM_NOT_SET) {
    algorithm = config.selected_algorithm();
  } else {
    algorithm = se::blas::kDefaultAlgorithm;
  }

  const Shape& lhs_shape = gemm->operand(0)->shape();
  const Shape& rhs_shape = gemm->operand(1)->shape();
  const DotDimensionNumbers& dot_dims = config.dot_dimension_numbers();
  const Shape& output_shape =
      gemm->shape().IsTuple() ? gemm->shape().tuple_shapes(0) : gemm->shape();

  int64 precision = -1/*se::blas::kDefaultComputePrecision*/;
  // for (auto operand_precision : config.precision_config().operand_precision()) {
  //   precision = std::max(precision, static_cast<int64>(operand_precision));
  // }
  // const PrecisionConfig::Algorithm precision_algorithm =
  //     config.precision_config().algorithm();
  return GemmConfig::For(
      lhs_shape, AsInt64Slice(dot_dims.lhs_batch_dimensions()),
      AsInt64Slice(dot_dims.lhs_contracting_dimensions()), rhs_shape,
      AsInt64Slice(dot_dims.rhs_batch_dimensions()), 
      AsInt64Slice(dot_dims.rhs_contracting_dimensions()),
      output_shape, config.alpha_real(), config.alpha_imag(), config.beta(),
      algorithm, precision, /*grad_x*/false, /*grad_y*/false);
}

StatusOr<GemmConfig::DescriptorsTuple> GemmConfig::GetMatrixDescriptors(
    se::DeviceMemoryBase lhs_buf, se::DeviceMemoryBase rhs_buf,
    se::DeviceMemoryBase out_buf) const {
  auto create_matrix_desc = [](const se::gpu::MatrixLayout& layout,
                               se::DeviceMemoryBase data) {
    return se::gpu::MatrixDescriptor{
        data, layout.leading_dim_stride, layout.batch_stride, 
        layout.dtype,
        // BLAS is column-major by default.
        (layout.order == se::gpu::MatrixLayout::Order::kColumnMajor
             ? se::blas::Transpose::kNoTranspose
             : se::blas::Transpose::kTranspose)};
  };
  // TODO: make a local copy to prevent modification of layouts,
  // but maybe we can modify them once instead during creation ?
  se::gpu::MatrixLayout lhs = lhs_layout, rhs = rhs_layout, out = output_layout;

  bool must_swap_operands = MakeOutputColumnMajor(lhs, rhs, out);
  if (must_swap_operands) {
    std::swap(lhs_buf, rhs_buf);
  }

  se::gpu::OutputMatrixDescriptor out_desc = create_matrix_desc(out, out_buf);
  out_desc.batch_size = out.batch_size;
  out_desc.m = out.num_rows;
  out_desc.n = out.num_cols;
  out_desc.k = lhs.num_cols;
  // TODO(tdanyluk): Investigate why don't we use the actual precision (and
  // algorithm) here? Why do we use the default?
  TF_ASSIGN_OR_RETURN(out_desc.compute_type,
                      se::gpu::GetBlasComputationType(
                          lhs.dtype, out.dtype, -1));

  se::gpu::MatrixDescriptor lhs_desc = create_matrix_desc(lhs, lhs_buf),
                            rhs_desc = create_matrix_desc(rhs, rhs_buf);

  return DescriptorsTuple{lhs_desc, rhs_desc, out_desc, must_swap_operands};
}

namespace {

template <typename Scale, typename Input, typename Output>
Status DoGemmWithAlgorithm(const se::gpu::MatrixDescriptor& lhs,
                                 const se::gpu::MatrixDescriptor& rhs,
                                 const se::gpu::OutputMatrixDescriptor& output,
                                 se::DeviceMemoryBase workspace, Scale alpha,
                                 Scale beta, se::Stream* stream,
                                 se::blas::AlgorithmType algorithm,
                                 se::blas::ProfileResult* profile_result,
                                 se::blas::CallContext context) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  PrimitiveType lhs_type = primitive_util::NativeToPrimitiveType<Input>();
  PrimitiveType output_type = primitive_util::NativeToPrimitiveType<Output>();
  TF_ASSIGN_OR_RETURN(
      se::blas::ComputationType computation_type,
      se::gpu::GetBlasComputationType(lhs_type, output_type, -1));
  se::DeviceMemory<Output> output_data(output.data);

  // Set a workspace for all Blas operations launched below.
  auto* blas = stream->parent()->AsBlas();
  if (blas == nullptr) {
    return xla::InternalError("No Blas support for stream");
  }

  se::blas::BlasSupport::ScopedWorkspace scoped_workspace(blas, &workspace);

  if (output.batch_size != 1) {
    return blas->BlasGemmStridedBatchedWithAlgorithm(
        stream, lhs.transpose, rhs.transpose, output.m, output.n, output.k,
        alpha, lhs.cast<Input>(), lhs.leading_dim_stride, lhs.batch_stride,
        rhs.cast<Input>(), rhs.leading_dim_stride, rhs.batch_stride, beta,
        &output_data, output.leading_dim_stride, output.batch_stride,
        output.batch_size, computation_type, algorithm, 
        profile_result, context);
  } else {
    return blas->BlasGemmWithAlgorithm(
        stream, lhs.transpose, rhs.transpose, output.m, output.n, output.k,
        alpha, lhs.cast<Input>(), lhs.leading_dim_stride, rhs.cast<Input>(),
        rhs.leading_dim_stride, beta, &output_data, output.leading_dim_stride,
        computation_type, algorithm, profile_result, context);
  }
}

template <typename Scale, typename Input, typename Output>
Status DoGemm(const se::gpu::MatrixDescriptor& lhs,
                    const se::gpu::MatrixDescriptor& rhs,
                    const se::gpu::OutputMatrixDescriptor& output,
                    se::DeviceMemoryBase workspace, Scale alpha, Scale beta,
                    se::Stream* stream,
                    absl::optional<se::blas::AlgorithmType> algorithm,
                    se::blas::ComputePrecision compute_precision,
                    se::blas::ProfileResult* profile_result,
                    se::blas::CallContext context) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  se::DeviceMemory<Output> output_data(output.data);
  auto* blas = stream->parent()->AsBlas();
  if (blas == nullptr) {
    return xla::InternalError("No Blas support for stream");
  }

  if (algorithm) {
    return DoGemmWithAlgorithm<Scale, Input, Output>(
        lhs, rhs, output, workspace, alpha, beta, stream,
        *algorithm, compute_precision, profile_result,
        context);
  }

  // Set a workspace for all Blas operations launched below.
  se::blas::BlasSupport::ScopedWorkspace scoped_workspace(blas, &workspace);

  if (output.batch_size != 1) {
    return blas->BlasGemmStridedBatched(
        stream, lhs.transpose, rhs.transpose, output.m, output.n, output.k,
        alpha, lhs.cast<Input>(), lhs.leading_dim_stride, lhs.batch_stride,
        rhs.cast<Input>(), rhs.leading_dim_stride, rhs.batch_stride, beta,
        &output_data, output.leading_dim_stride, output.batch_stride,
        output.batch_size, context);
  }

  return blas->BlasGemm(stream, lhs.transpose, rhs.transpose, output.m,
                        output.n, output.k, alpha, lhs.cast<Input>(),
                        lhs.leading_dim_stride, rhs.cast<Input>(),
                        rhs.leading_dim_stride, beta, &output_data,
                        output.leading_dim_stride, context);
}

}  // namespace

// Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
//                      se::DeviceMemoryBase rhs_buffer,
//                      se::DeviceMemoryBase output_buffer,
//                      se::DeviceMemoryBase workspace_buffer,
//                      bool deterministic_ops, se::Stream* stream,
//                      absl::optional<se::blas::AlgorithmType> algorithm,
//                      se::blas::ProfileResult* profile_result) {
//   VLOG(2) << "Executing a GemmThunk";

//   TF_ASSIGN_OR_RETURN(
//       GemmConfig::DescriptorsTuple desc,
//       config.GetMatrixDescriptors(lhs_buffer, rhs_buffer, output_buffer));

//   se::NumericOptions numeric_options{
//       deterministic_ops,
//       /*allow_tf32=*/true};

//   if (!algorithm) algorithm = config.algorithm;

//   se::blas::CallContext context = se::blas::CallContext::kNone;
//   if (config.grad_x) {
//     context = desc.operands_swapped ? se::blas::CallContext::kBackpropInput2
//                                     : se::blas::CallContext::kBackpropInput1;
//   }
//   if (config.grad_y) {
//     context = desc.operands_swapped ? se::blas::CallContext::kBackpropInput1
//                                     : se::blas::CallContext::kBackpropInput2;
//   }

//   auto operand_types = std::make_tuple(config.lhs_layout.dtype, 
//       config.rhs_layout.dtype, config.output_layout.dtype);

//   // Skip degenerate gemm with memzero. In general this is not safe, because it
//   // will suppress NaN propagation, however cuBLAS internally has exactly the
//   // same optimization for compatibility with NETLIB implementation, so we are
//   // not making things worse (and cuBLAS optimization is incompatible with CUDA
//   // graphs, so we are making sure we do not trigger it).
//   if (config.alpha.real() == 0.0 && config.alpha.imag() == 0.0 &&
//       config.beta == 0.0) {
//     return stream->MemZero(&output_buffer, output_buffer.size());
//   }

// #define TYPED_GEMM(SCALENTYPE, ATYPE, BTYPE, CTYPE)                         \
//   if (operand_types == std::make_tuple(ATYPE, BTYPE, CTYPE)) {              \
//     using NativeScaleType =                                                 \
//         primitive_util::PrimitiveTypeToNative<SCALENTYPE>::type;            \
//     using NativeAType = primitive_util::PrimitiveTypeToNative<ATYPE>::type; \
//     using NativeCType = primitive_util::PrimitiveTypeToNative<CTYPE>::type; \
//     return DoGemm<NativeScaleType, NativeAType, NativeCType>(               \
//         desc.lhs, desc.rhs, desc.output, workspace_buffer,                  \
//         static_cast<NativeScaleType>(config.alpha.real()),                  \
//         static_cast<NativeScaleType>(config.beta), stream,                  \
//         algorithm, config.compute_precision,                                \
//         numeric_options, profile_result, context);                          \
//   }

// #define TYPED_GEMM_COMPLEX(SCALENTYPE, ATYPE, BTYPE, CTYPE)                 \
//   if (operand_types == std::make_tuple(ATYPE, BTYPE, CTYPE)) {              \
//     using NativeScaleType =                                                 \
//         primitive_util::PrimitiveTypeToNative<SCALENTYPE>::type;            \
//     using NativeAType = primitive_util::PrimitiveTypeToNative<ATYPE>::type; \
//     using NativeCType = primitive_util::PrimitiveTypeToNative<CTYPE>::type; \
//     return DoGemm<NativeScaleType, NativeAType, NativeCType>(               \
//         desc.lhs, desc.rhs, desc.output, workspace_buffer,                  \
//         static_cast<NativeScaleType>(config.alpha),                         \
//         static_cast<NativeScaleType>(config.beta), stream,                  \
//         algorithm, config.compute_precision,                                \
//         numeric_options, profile_result, context);                          \
//   }

//   if (config.output_layout.dtype == S32) {
//     if (!algorithm) algorithm = se::blas::kDefaultGemmAlgo;
//     // TODO(tdanyluk): Investigate why don't we use the actual precision (and
//     // algorithm) here? Why do we use the default?
//     return DoGemmWithAlgorithm<int32_t, int8_t, int32_t>(
//         desc.lhs, desc.rhs, desc.output, workspace_buffer,
//         static_cast<int32_t>(config.alpha.real()),
//         static_cast<int32_t>(config.beta), stream, 
//         *algorithm, se::blas::kDefaultComputePrecision, numeric_options,
//         profile_result, context);
//   }

//   TYPED_GEMM(F32, BF16, BF16, BF16)
//   TYPED_GEMM(F32, F16, F16, F16)
//   TYPED_GEMM(F32, S8, S8, F32)
//   TYPED_GEMM(F32, BF16, BF16, F32)
//   TYPED_GEMM(F32, F16, F16, F32)
//   TYPED_GEMM(F32, F32, F32, F32)
//   TYPED_GEMM(F64, F64, F64, F64)
//   TYPED_GEMM_COMPLEX(C64, C64, C64, C64)
//   TYPED_GEMM_COMPLEX(C128, C128, C128, C128)

// #undef TYPED_GEMM
// #undef TYPED_GEMM_COMPLEX
//   return Internal(
//       "Unexpected GEMM dtype: %s %s %s",
//       primitive_util::LowercasePrimitiveTypeName(config.lhs_layout.dtype),
//       primitive_util::LowercasePrimitiveTypeName(config.rhs_layout.dtype),
//       primitive_util::LowercasePrimitiveTypeName(config.output_layout.dtype));
// }  // namespace gpu

namespace gpublas_lt {

StatusOr<bool> EpilogueAddsVectorBias(
    GemmBackendConfig::Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
    case GemmBackendConfig::RELU:
    case GemmBackendConfig::GELU:
    case GemmBackendConfig::GELU_AUX:
      return false;
    case GemmBackendConfig::BIAS:
    case GemmBackendConfig::BIAS_RELU:
    case GemmBackendConfig::BIAS_GELU:
    case GemmBackendConfig::BIAS_GELU_AUX:
      return true;
    default:
      return Internal("Unknown Epilogue.");
  }
}

StatusOr<bool> EpilogueHasAuxiliaryOutput(
    GemmBackendConfig::Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
    case GemmBackendConfig::RELU:
    case GemmBackendConfig::GELU:
    case GemmBackendConfig::BIAS:
    case GemmBackendConfig::BIAS_RELU:
    case GemmBackendConfig::BIAS_GELU:
      return false;
    case GemmBackendConfig::GELU_AUX:
    case GemmBackendConfig::BIAS_GELU_AUX:
      return true;
    default:
      return Internal("Unknown Epilogue.");
  }
}

StatusOr<se::gpu::BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig::Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return se::gpu::BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return se::gpu::BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return se::gpu::BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::GELU_AUX:
      return se::gpu::BlasLt::Epilogue::kGELUWithAux;
    case GemmBackendConfig::BIAS:
      return se::gpu::BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return se::gpu::BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return se::gpu::BlasLt::Epilogue::kBiasThenGELU;
    case GemmBackendConfig::BIAS_GELU_AUX:
      return se::gpu::BlasLt::Epilogue::kBiasThenGELUWithAux;
    default:
      return Internal("unexpected epilogue value");
  }
}

}  // namespace gpublas_lt

bool IsDotSupportedByClassicalEmitters(const HloInstruction& dot) {
  // Let us be conservative and only throw float dots at the emitters.
  switch (dot.shape().element_type()) {
    case F16:
    case F32:
    case BF16:
      return true;
    default:
      return false;
  }
}

}  // namespace gpu
}  // namespace xla
