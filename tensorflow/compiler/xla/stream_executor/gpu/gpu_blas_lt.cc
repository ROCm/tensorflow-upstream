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

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_blas_lt.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/util/env_var.h"

namespace stream_executor {

namespace gpu {

using blas::ComputationType;
using blas::DataType;
using xla::PrimitiveType;

namespace {

bool TF32_Enabled() {
   static std::atomic_bool result{[] {
    bool value = false;	          	
    (void)tsl::ReadBoolFromEnvVar("ROCM_XF32",
        /*default_value=*/false, &value);
    return value;
  }()};
  return result;
}

bool Fast_16F_Enabled() {
  static std::atomic_bool result{[] {
  bool value = false;	          	
  (void)tsl::ReadBoolFromEnvVar("ROCM_FAST_16F",
        /*default_value=*/false, &value);
    return value;
  }()};
  return result;
}

} // namespace

xla::StatusOr<DataType> AsBlasDataType(PrimitiveType dtype) {
  switch (dtype) {
    case PrimitiveType::S8:
      return DataType::kInt8;
    case PrimitiveType::F16:
      return DataType::kHalf;
    case PrimitiveType::BF16:
      return DataType::kBF16;
    case PrimitiveType::F32:
      return DataType::kFloat;
    case PrimitiveType::S32:
      return DataType::kInt32;
    case PrimitiveType::F64:
      return DataType::kDouble;
    case PrimitiveType::C64:
      return DataType::kComplexFloat;
    case PrimitiveType::C128:
      return DataType::kComplexDouble;
    default:
      return xla::InternalError(
          "AsBlasDataType: unsupported type: %s",
          xla::primitive_util::LowercasePrimitiveTypeName(dtype));
  }
}

xla::StatusOr<ComputationType> GetBlasComputationType(
  DataType lhs_dtype, DataType output_dtype, int64_t /*compute_precision*/) {

  auto f16_comp = Fast_16F_Enabled() ? 
                   ComputationType::kF16AsF32 : ComputationType::kF32,
       bf16_comp = Fast_16F_Enabled() ? 
                   ComputationType::kBF16AsF32 : ComputationType::kF32;

  switch (output_dtype) {
    case DataType::kHalf:   // fall-through
      return f16_comp;
    case DataType::kBF16:
      return bf16_comp;
    case DataType::kFloat:  // fall-through
      if (lhs_dtype == DataType::kHalf) return f16_comp;
      if (lhs_dtype == DataType::kBF16) return bf16_comp;
      return ComputationType::kF32;
    case DataType::kComplexFloat:
      return ComputationType::kF32;
    case DataType::kDouble: // fall-through
    case DataType::kComplexDouble:
      return ComputationType::kF64;
    case DataType::kInt32:
      return ComputationType::kI32;
    default:
      return xla::InternalError("GetBlasComputationType: unsupported type");
  }
}

MatrixLayout::MatrixLayout(blas::DataType dtype_, int64_t num_rows_,
                           int64_t num_cols_, MatrixLayout::Order order_,
                           int64_t batch_size_,
                           absl::optional<int64_t> leading_dim_stride_,
                           absl::optional<int64_t> batch_stride_,
                           absl::optional<blas::Transpose> transpose_)
    : dtype(dtype_),
      num_rows(num_rows_),
      num_cols(num_cols_),
      order(order_),
      batch_size(batch_size_) {
  if (!leading_dim_stride_) {
    leading_dim_stride = order == Order::kRowMajor ? num_cols : num_rows;
  } else {
    leading_dim_stride = *leading_dim_stride_;
  }
  if (!batch_stride_) {
    batch_stride = (batch_size > 1) ? num_rows * num_cols : 0;
  } else {
    batch_stride = *batch_stride_;
  }
  transpose = transpose_ ? *transpose_ : blas::Transpose::kNoTranspose;
}

void MatrixLayout::Transpose() {
  std::swap(num_rows, num_cols);
  order = (order == Order::kRowMajor) ? Order::kColumnMajor : Order::kRowMajor;
}

// BLAS GeMM's output is column-major. If we require row-major, use identity:
// C^T = (A @ B)^T = B^T @ A^T.
bool MakeOutputColumnMajor(MatrixLayout& lhs, MatrixLayout& rhs,
                           MatrixLayout& output, MatrixLayout* c) {
  bool swap_operands = output.order != MatrixLayout::Order::kColumnMajor;
  if (swap_operands) {
    std::swap(lhs, rhs);
    rhs.Transpose();
    // prevent layouts from being swapped two times if they are equal
    if (&lhs != &rhs) {
      lhs.Transpose();
    }
    if (c != nullptr && c != &output) {
      c->Transpose();
    }
    output.Transpose();
  }
  return swap_operands;
}

/*static*/ auto BlasLt::GetMatmulPlan(const Stream* stream,
                                      const GemmConfig& cfg)
    -> xla::StatusOr<MatmulPlanPtr> {
  auto blas = Get(stream);
  if (blas == nullptr) {
    return xla::InternalError("BlasLt is unavailable");
  }
  return blas->GetMatmulPlan(cfg);
}

/* static */ auto BlasLt::CreateGroupedMatmulPlan(Stream* stream, 				
		const GroupedGemmConfig& cfg) -> xla::StatusOr<GroupedMatmulPlanPtr> {
  auto blas = Get(stream);
  if (blas == nullptr) {
    return xla::InternalError("BlasLt is unavailable");
  }
  return blas->GetGroupedMatmulPlan(stream, cfg);
}

/*static*/ BlasLt* BlasLt::Get(const Stream* stream) {
  auto blas = stream->parent()->AsBlas();
  return (blas != nullptr ? blas->GetBlasLt() : nullptr);
}

DataType GetScaleType(DataType c_type, ComputationType compute_type) {
  if (compute_type == ComputationType::kF32 && 
        c_type != DataType::kComplexFloat) {
    return DataType::kFloat;
  }
  if (compute_type == ComputationType::kF16) return DataType::kFloat;
  return c_type;
}


namespace {

const std::vector<absl::string_view> TransposeNames = {
    "N", // kNoTranspose
    "T", // kTranspose
    "C", // kConjugateTranspose
};

xla::StatusOr<absl::string_view> Transpose2String(blas::Transpose type) {
  size_t idx = static_cast< size_t >(type);
  if (idx < TransposeNames.size()) return TransposeNames[idx];
  return xla::InternalError("Unknown transpose type!");
}

xla::StatusOr<blas::Transpose> String2Transpose(absl::string_view s) {
  for(size_t i = 0; i < TransposeNames.size(); i++) {
    if (s == TransposeNames[i]) return static_cast< blas::Transpose >(i);
  }
  return xla::InternalError("Unknown tranpose type!");
}

const std::vector<absl::string_view> TypeNames = {
    "f32_r",  //kFloat = 0,
    "f64_r",  //kDouble = 1,
    "f16_r",  //kHalf = 2,
    "i8_r",   //kInt8 = 3,
    "i32_r",  //kInt32 = 4,
    "f32_c",  //kComplexFloat = 5,
    "f64_c",  //kComplexDouble = 6,
    "bf16_r", //kBF16 = 7,
};

xla::StatusOr<absl::string_view> Type2String(blas::DataType type) {
  size_t idx = static_cast< size_t >(type);
  if (idx < TypeNames.size()) return TypeNames[idx];
  return xla::InternalError("Unknown data type!");
}

}  // namespace

std::string ToCSVString(const GemmConfig& cfg, bool full_string) {

  ///constexpr char kCsvComment = '#';
  constexpr char kCsvSep = ',';

  const auto& L = cfg.lhs_layout, &R = cfg.rhs_layout, &O = cfg.output_layout;

  std::ostringstream oss;
  auto type_a = Type2String(L.dtype).value(),
       type_b = Type2String(R.dtype).value(),
       type_c = Type2String(O.dtype).value(),
       trans_a = Transpose2String(L.transpose).value(),
       trans_b = Transpose2String(R.transpose).value();

// LHS: k x n
// RHS: m x k
// OUT: m x n
  // VLOG(0) << "LHS: " << L.num_cols << "x" << L.num_rows;
  // VLOG(0) << "RHS: " << R.num_cols << "x" << R.num_rows;
  // VLOG(0) << "OUT: " << O.num_cols << "x" << O.num_rows;
  int n = L.num_rows, k = L.num_cols, m = O.num_cols;
  oss << m << kCsvSep << n << kCsvSep << k << kCsvSep
     << O.batch_size << kCsvSep << trans_a << kCsvSep 
     << trans_b << kCsvSep << type_a << kCsvSep 
     << type_b << kCsvSep << type_c << kCsvSep << L.leading_dim_stride 
     << kCsvSep << R.leading_dim_stride << kCsvSep
     << O.leading_dim_stride  << kCsvSep << L.batch_stride << kCsvSep
     << R.batch_stride << kCsvSep << O.batch_stride;

  if (full_string) {
    // NOTE: epilogue is required for MatmulPlan caching !
    oss //<< kCsvSep << cfg.alpha << kCsvSep << cfg.beta 
        << kCsvSep << (int64_t)cfg.epilogue;
  }

  return oss.str();
}

}  // namespace gpu

}  // namespace stream_executor
