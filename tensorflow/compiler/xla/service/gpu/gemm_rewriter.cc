/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {

namespace m = match;

// auto Gemm(HloInstruction **instr) {
//   return m::CustomCall(instr, {kGemmCallTarget});
// }

// auto CublasLtMatmul(HloInstruction **instr) {
//   return m::CustomCall(instr, {kCublasLtMatmulCallTarget});
// }

// auto GemmOrCublasLtMatmul(HloInstruction **instr) {
//   return m::CustomCall(instr, {kGemmCallTarget, kCublasLtMatmulCallTarget});
// }

// The rewriting proceeds in a bottom-up way:
//
// (kDot A B) is rewritten into a (kCustomCall:gemm A B)
//
// (kMultiply (kCustomCall:gemm A B) C) is folding C (provided it's a constant)
// into an alpha parameter of the custom call.
//
// (kAdd (kCustomCall:gemm A B) C) is rewritten into (kCustomCall:gemm A B C),
// where the "beta" parameter is set to 1 (provided it was zero before,
// and provided C has no other users).
// We then guide the buffer assignment to alias the buffer of the custom call
// and C.
class GemmRewriterVisitor : public DfsHloRewriteVisitor {
 public:

  GemmRewriterVisitor(GpuVersion ver) : gpu_version_(ver) {}
 
  Status HandleDot(HloInstruction *instr) override {
    if (IsMatrixMultiplication(*instr)) {
      VLOG(-1) << "Handling Dot";
      CHECK(!instr->IsRank2Transpose());
      HloInstruction *lhs = instr->mutable_operand(0);
      HloInstruction *rhs = instr->mutable_operand(1);
      CHECK(!lhs->IsRank2Transpose());
      CHECK(!rhs->IsRank2Transpose());
      const Shape &output_shape = instr->shape();
      int64 batch_size = std::accumulate(output_shape.dimensions().begin(),
                                         output_shape.dimensions().end() - 2, 1,
                                         std::multiplies<int64>());
      GemmBackendConfig gemm_config;
      gemm_config.set_alpha_real(1.0);
      gemm_config.set_alpha_imag(0.0);
      gemm_config.set_beta(0.0);
      *gemm_config.mutable_dot_dimension_numbers() =
          instr->dot_dimension_numbers();
      gemm_config.set_batch_size(batch_size);
      //auto attributes = instr->frontend_attributes().map();
      //gemm_config.set_grad_x(attributes["grad_x"] == "true");
      //gemm_config.set_grad_y(attributes["grad_y"] == "true");

      TF_ASSIGN_OR_RETURN(
          auto gemm_target,
          GetGemmCustomCallTarget(*instr, gemm_config));

      auto gemm_call =
          HloInstruction::CreateCustomCall(output_shape, {lhs, rhs},
                                           gemm_target);

      TF_RETURN_IF_ERROR(gemm_call->set_backend_config(gemm_config));
      TF_RETURN_IF_ERROR(
          ReplaceWithNewInstruction(instr, std::move(gemm_call)));
    }

    return Status::OK();
  }

  Status HandleMultiply(HloInstruction *instr) override {
    HloInstruction *alpha, *existing_gemm;
    if (Match(instr,
              m::MultiplyAnyOrder(
                  m::Op(&existing_gemm).WithCustomCallTarget(
                    {kGemmCallTarget, kCublasLtMatmulCallTarget}),
                  m::Broadcast(m::ConstantScalar(&alpha))))) {
      TF_ASSIGN_OR_RETURN(auto config,
                          existing_gemm->backend_config<GemmBackendConfig>());
      if (config.beta() == 0.0 && existing_gemm->user_count() == 1) {
        complex128 prev_alpha = {config.alpha_real(), config.alpha_imag()};
        complex128 new_alpha =
            *alpha->literal().GetAsComplex128({}) * prev_alpha;
        config.set_alpha_real(new_alpha.real());
        config.set_alpha_imag(new_alpha.imag());
        TF_RETURN_IF_ERROR(existing_gemm->set_backend_config(config));
        TF_RETURN_IF_ERROR(ReplaceInstruction(instr, existing_gemm));
      }
    }
    return Status::OK();
  }

  Status HandleAdd(HloInstruction *instr) override {
    HloInstruction *bias, *existing_gemm;
    if (Match(instr,
              m::AddAnyOrder(
                  m::Op(&existing_gemm).WithCustomCallTarget(
                        {kGemmCallTarget, kCublasLtMatmulCallTarget}),
                  m::Op(&bias)))) {
      auto config =
          existing_gemm->backend_config<GemmBackendConfig>().ValueOrDie();
      if (config.beta() == 0 && bias->user_count() == 1 &&
          existing_gemm->user_count() == 1 &&
          bias->shape() == existing_gemm->shape()) {
        config.set_beta(1.0);
        CHECK_EQ(existing_gemm->operand_count(), 2);
        std::unique_ptr<HloInstruction> gemm_call =
            HloInstruction::CreateCustomCall(
                instr->shape(),
                {existing_gemm->mutable_operand(0),
                 existing_gemm->mutable_operand(1), bias},
          static_cast<HloCustomCallInstruction *>(existing_gemm)
                                          ->custom_call_target());
        TF_RETURN_IF_ERROR(gemm_call->set_backend_config(config));
        TF_RETURN_IF_ERROR(
            ReplaceWithNewInstruction(instr, std::move(gemm_call)));
      }
    }
    return Status::OK();
  }
private:

  static auto GetCuda(const GpuVersion &gpu_version) {
    return absl::get_if<std::pair<int, int>>(&gpu_version);
  }

  static auto GetRocm(const GpuVersion &gpu_version) {
    return absl::get_if<int>(&gpu_version);
  }

  // Choose cublas or cublasLt for the target of the custom call that instr will
  // be rewritten into.
  StatusOr<absl::string_view> GetGemmCustomCallTarget(
      const HloInstruction &instr,
      const GemmBackendConfig &gemm_backend_config) const {
    if (!instr.GetModule()
             ->config()
             .debug_options()
             .xla_gpu_enable_cublaslt()) {
      // cublasLt is not enabled.
      return absl::string_view(kGemmCallTarget);
    }
    // cublasLt is enabled, check if other internal conditions are met.
    const HloInstruction *lhs = instr.operand(0);
    const HloInstruction *rhs = instr.operand(1);
    if (lhs->shape().element_type() == S8 ||
        rhs->shape().element_type() == S8) {
      // TODO(b/241446501) The XLA usage of cublasLt does not yet handle
      // int8 matmuls. Fallback to legacy cublas.
      return absl::string_view(kGemmCallTarget);
    }

    // All internal conditions are met, check if we meet the requirements of
    // cublasLt.
    TF_ASSIGN_OR_RETURN(bool gemm_is_supported_by_cublas_lt,
                        GemmIsSupportedByCublasLt(instr, gemm_backend_config));
    if (gemm_is_supported_by_cublas_lt) {
      return absl::string_view(kCublasLtMatmulCallTarget);
    }

    // This case is not supported by cublasLt, fallback to legacy cublas.
    return absl::string_view(kGemmCallTarget);
  }

  StatusOr<bool> MatrixIsColumnMajor(
      const HloInstruction &instr, const GemmBackendConfig &gemm_backend_config,
      const std::string matrix_name = "output") const {
    const HloInstruction *lhs = instr.operand(0);
    const HloInstruction *rhs = instr.operand(1);

    const DotDimensionNumbers &dot_dims =
        gemm_backend_config.dot_dimension_numbers();
    // We use ALG_UNSET and kDefaultComputePrecision because we don't care about
    // the precision, just the layout, since we're just checking if the matrix
    // is column-major.
    TF_ASSIGN_OR_RETURN(
        auto gemm_config,
        GemmConfig::For(
            lhs->shape(), 
            AsInt64Slice(dot_dims.lhs_batch_dimensions()),
            AsInt64Slice(dot_dims.lhs_contracting_dimensions()), rhs->shape(),
            AsInt64Slice(dot_dims.rhs_batch_dimensions()),
            AsInt64Slice(dot_dims.rhs_contracting_dimensions()),
            /*output_shape=*/instr.shape(), gemm_backend_config.alpha_real(),
            gemm_backend_config.alpha_imag(), gemm_backend_config.beta(),
            /*algorithm*/ absl::nullopt, se::blas::kDefaultComputePrecision,
            false, false));

    if (matrix_name == "lhs" || matrix_name == "a") {
      return gemm_config.lhs_layout.order == MatrixLayout::Order::kColumnMajor;
    } else if (matrix_name == "rhs" || matrix_name == "b") {
      return gemm_config.rhs_layout.order == MatrixLayout::Order::kColumnMajor;
    } else if (matrix_name == "output" || matrix_name == "d") {
      return gemm_config.output_layout.order ==
             MatrixLayout::Order::kColumnMajor;
    } else {
      return Internal("Invalid matrix name.");
    }
  }
  
  StatusOr<bool> GemmIsSupportedByCublasLt(
      const HloInstruction &instr,
      const GemmBackendConfig &gemm_backend_config) const {
    const HloInstruction *lhs = instr.operand(0);
    const HloInstruction *rhs = instr.operand(1);
    const Shape &output_shape = instr.shape();

    TF_ASSIGN_OR_RETURN(
        bool types_are_supported_by_cublas_lt,
        TypesAreSupportedByCublasLt(instr, gemm_backend_config));
    if (!types_are_supported_by_cublas_lt) {
      VLOG(-1) << "Not converting to cublaslt: unsupported types";
      return false;
    }

    // The cublasLt API has two currently known limitations:
    // 1. Batch count must be <2^16.
    constexpr int64 kMaxBatchCount = 65535;
    // We get the batch dimension size from lhs here, but we could just as well
    // use rhs; they are guaranteed to be the same (TODO:Verify).
    const auto &batch_dimensions =
        gemm_backend_config.dot_dimension_numbers().lhs_batch_dimensions();
    int batch_count = (batch_dimensions.empty() ? 0 : 1);
    // All batch dimensions get flattened into a single batch dimension.
    for (auto batch_dimension : batch_dimensions) {
      batch_count *= lhs->shape().dimensions(batch_dimension);
    }
    if (batch_count > kMaxBatchCount) {
      // This is not supported by cublasLt.
      return false;
    }

    TF_ASSIGN_OR_RETURN(bool output_is_column_major,
                        MatrixIsColumnMajor(instr, gemm_backend_config));

    // 2. cublasLt does not support rhs col dimension size > 4194240 for
    // C64.
    constexpr int kMaxDimensionSize{4194240};
    if (output_shape.element_type() != C64) {
      // Does not match type in unsupported case.
      VLOG(-1) << "Converting to cublaslt";
      return true;
    }

    auto pcuda = GetCuda(gpu_version_);
    if (pcuda != nullptr) {
      if (!(pcuda->first < 9)) { // se::CudaComputeCapability::AMPERE
        // cuBlasLt has an implementation for complex data with compute type
        // 32F_FAST_32TF that uses tensor cores and that is free from the
        // restriction. This implementation only works on Ampere
        // architecture though (where TF32 was introduced).
        return true;
      }
    }
    // Get the rhs non-contracting dimensions as they will eventually be at the
    // cublasLt level.
    std::vector<int64> rhs_non_contracting_dims;
    const DotDimensionNumbers &dot_dims =
        gemm_backend_config.dot_dimension_numbers();

    if (!output_is_column_major) {
      // cublasLt's matmul output is column major by default. This gemm requires
      // the output to be in row major. Later we will swap lhs & rhs (and
      // transpose each operand) of this gemm. Since we care about the rhs at
      // the cublasLt level, this swap means that we care about the lhs right
      // here.
      TF_ASSIGN_OR_RETURN(
          rhs_non_contracting_dims,
          GetNonContractingDims(lhs->shape(), 
              AsInt64Slice(dot_dims.lhs_batch_dimensions()),
              AsInt64Slice(dot_dims.lhs_contracting_dimensions())));
    } else {
      TF_ASSIGN_OR_RETURN(
          rhs_non_contracting_dims,
          GetNonContractingDims(rhs->shape(), 
            AsInt64Slice(dot_dims.rhs_batch_dimensions()),
            AsInt64Slice(dot_dims.rhs_contracting_dimensions())));
    }

    const auto lhs_non_contracting_dimension_size = absl::c_accumulate(
        rhs_non_contracting_dims, 1, [&](int64 size, int64 dim) {
          return size * lhs->shape().dimensions(dim);
        });

    // Check that the size of the non-contracting dimension is not too large.
    return lhs_non_contracting_dimension_size <= kMaxDimensionSize;
  }

  
  StatusOr<bool> TypesAreSupportedByCublasLt(
      const HloInstruction &instr, const GemmBackendConfig &backend_config,
      const HloInstruction *bias = nullptr) const {
    // Figure out the Atype/Btype.
    const PrimitiveType a_dtype = instr.operand(0)->shape().element_type();
    const PrimitiveType b_dtype = instr.operand(1)->shape().element_type();
    const PrimitiveType output_type =
        bias ? bias->shape().element_type() : instr.shape().element_type();
    const std::array<PrimitiveType, 8> supported_type = {
        PrimitiveType::S8,         PrimitiveType::F16,
        PrimitiveType::BF16,       PrimitiveType::F32,
        PrimitiveType::S32,        PrimitiveType::F64,
        PrimitiveType::C64,        PrimitiveType::C128};
    if (!absl::c_linear_search(supported_type, output_type)) return false;
    // cublasLt has a defined set of combinations of types that it supports.
    // Figure out the computeType and scaleType.
    TF_ASSIGN_OR_RETURN(const se::blas::DataType output_dtype,
                        se::gpu::AsBlasDataType(output_type));
    // const int max_precision = *absl::c_max_element(
    //     backend_config.precision_config().operand_precision());
    // const PrecisionConfig::Algorithm algorithm =
    //     backend_config.precision_config().algorithm();
    // if (!algorithm_util::IsSupportedByCublasOrCublasLt(algorithm)) return false;

    TF_ASSIGN_OR_RETURN(
        const se::blas::ComputationType compute_type,
        se::gpu::GetBlasComputationType(
            a_dtype, instr.shape().element_type(), -1));
    se::blas::DataType scale_type =
        se::gpu::GetScaleType(output_dtype, compute_type);

    using se::blas::ComputationType;
    using se::blas::DataType;
    using TypeCombinations = std::initializer_list<std::tuple<
        ComputationType, DataType /*scale_type*/, PrimitiveType /*a_dtype*/,
        PrimitiveType /*b_dtype*/, DataType /*output_dtype*/>>;
    // This matrix of supported types is taken directly from cublasLt
    // documentation.
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul
    const TypeCombinations supported_cublas_type_combinations = {
        // There would be an entry here for A/BType complex int8, but we do
        // not support that type.
        {ComputationType::kF32, DataType::kComplexFloat, PrimitiveType::C64,
         PrimitiveType::C64, DataType::kComplexFloat},

        // {ComputationType::kF16AsF32, DataType::kFloat, PrimitiveType::F32,
        //  PrimitiveType::F32, DataType::kFloat},
        // {ComputationType::kF16AsF32, DataType::kComplexFloat,
        //  PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},
        // The next 4 may be supported by hipblaslt, but they are not
        // covered by any unit tests
        // {ComputationType::kBF16AsF32, DataType::kFloat, PrimitiveType::F32,
        //  PrimitiveType::F32, DataType::kFloat},
        // {ComputationType::kBF16AsF32, DataType::kComplexFloat,
        //  PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},

        // {ComputationType::kTF32AsF32, DataType::kFloat, PrimitiveType::F32,
        //  PrimitiveType::F32, DataType::kFloat},
        // {ComputationType::kTF32AsF32, DataType::kComplexFloat,
        //  PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},

        {ComputationType::kF64, DataType::kDouble, PrimitiveType::F64,
         PrimitiveType::F64, DataType::kDouble},
        {ComputationType::kF64, DataType::kComplexDouble, PrimitiveType::C128,
         PrimitiveType::C128, DataType::kComplexDouble},
    };
    if (GetCuda(gpu_version_) != nullptr &&
        absl::c_linear_search(supported_cublas_type_combinations,
                         std::make_tuple(compute_type, scale_type, a_dtype,
                                         b_dtype, output_dtype))) {
      return true;
    }
    const TypeCombinations supported_type_combinations = {
        // Other data types:
        {ComputationType::kF16, DataType::kHalf, PrimitiveType::F16,
         PrimitiveType::F16, DataType::kHalf},

        {ComputationType::kI32, DataType::kInt32, PrimitiveType::S8,
         PrimitiveType::S8, DataType::kInt32},
        {ComputationType::kI32, DataType::kFloat, PrimitiveType::S8,
         PrimitiveType::S8, DataType::kInt8},

        {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
         PrimitiveType::BF16, DataType::kBFloat16},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F16,
         PrimitiveType::F16, DataType::kHalf},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::S8,
         PrimitiveType::S8, DataType::kFloat},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
         PrimitiveType::BF16, DataType::kFloat},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F16,
         PrimitiveType::F16, DataType::kFloat},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F32,
         PrimitiveType::F32, DataType::kFloat},
    };

    return absl::c_linear_search(
        supported_type_combinations,
        std::make_tuple(compute_type, scale_type, a_dtype, b_dtype,
                        output_dtype));
  }
private:
  GpuVersion gpu_version_;
};

static StatusOr<bool> RunOnComputation(HloComputation *computation,
        GpuVersion version) {
  GemmRewriterVisitor visitor(version);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

StatusOr<bool> GemmRewriter::Run(HloModule *module) {
  bool changed = false;
  for (HloComputation *computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation, gpu_version_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
