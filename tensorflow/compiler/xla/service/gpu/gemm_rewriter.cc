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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/evaluator/hlo_evaluator.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/literal_comparison.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/protobuf/dnn.pb.h"


namespace xla {
namespace gpu {
namespace {

namespace m = match;

// Give this instruction a more useful name than "custom-call.42".
Status SetName(HloModule *module, HloInstruction *gemm) {
  if (IsCublasLtMatmul(*gemm)) {
    module->SetAndUniquifyInstrName(gemm, "cublas-lt-matmul");
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(auto config,
                      gemm->backend_config<GemmBackendConfig>());
  const DotDimensionNumbers &dot_dims = config.dot_dimension_numbers();
  bool is_batch_dot = !dot_dims.lhs_batch_dimensions().empty() ||
                      !dot_dims.rhs_batch_dimensions().empty();

  module->SetAndUniquifyInstrName(
      gemm, is_batch_dot ? "cublas-batch-gemm" : "cublas-gemm");
  return absl::OkStatus();
}

// Returns whether a given PrimitiveType is supported by cuBLASLt Epilogue
// Fusion. A table of supported data types can be found in the cuBLASLt
// documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul.
// Note that `Ctype` also describes the output type of the GEMM. Rows with
// `Non-default epilogue not supported` entries in the last column indicate data
// types not compatible with Epilogue Fusion.
bool SupportsEpilogueFusion(PrimitiveType type) {
  switch (type) {
    case F8E4M3FN:
    case F8E5M2:
    case F16:
    case BF16:
    case F32:
    case F64:
      return true;
    default:
      return false;
  }
}

bool IsF8Type(const HloInstruction *instr) {
  return primitive_util::IsF8Type(instr->shape().element_type());
}

// Returns a new shape with non-batch dimensions padded to multiples of 16, as
// required by cuBLASLt FP8 gemms.
Shape PadShapeToMultipleOf16(const Shape old_shape,
                             const absl::Span<const int64_t> batch_dims) {
  Shape padded_shape = old_shape;
  for (int i = 0; i < old_shape.rank(); ++i) {
    if (!absl::c_linear_search(batch_dims, i)) {
      int64_t padded_dimension =
          RoundUpTo<int64_t>(old_shape.dimensions(i), 16);
      padded_shape.set_dimensions(i, padded_dimension);
    }
  }
  return padded_shape;
}

// Pad the dimensions of the operands to the target shape.
HloInstruction *PadOperandToTargetShape(const Shape &target,
                                        HloInstruction *x) {
  if (ShapeUtil::Equal(target, x->shape()) ||
      !ShapeUtil::SameElementType(x->shape(), target)) {
    return x;
  }

  PaddingConfig padding_config;
  for (int i = 0; i < x->shape().rank(); ++i) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(0);
    dimension->set_edge_padding_high(target.dimensions(i) -
                                     x->shape().dimensions(i));
    dimension->set_interior_padding(0);
  }

  HloInstruction *zero = x->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(x->shape().element_type())));
  return x->AddInstruction(
      HloInstruction::CreatePad(target, x, zero, padding_config));
}

// Pad the non-batch dimensions of the operands to multiples of 16 as required
// by cuBLASLt FP8 gemms.
HloInstruction *PadOperandToMultipleOf16(absl::Span<const int64_t> batch_dims,
                                         HloInstruction *x) {
  Shape padded_shape = PadShapeToMultipleOf16(x->shape(), batch_dims);
  return PadOperandToTargetShape(padded_shape, x);
}

// Calculates the reciprocal of scalar when invert is true and converts to FP32.
StatusOr<HloInstruction *> InvertAndConvertScalar(HloInstruction *scalar,
                                                        bool invert) {
  DCHECK(ShapeUtil::IsScalar(scalar->shape()));

  if (invert) {
    Literal one_literal = LiteralUtil::One(scalar->shape().element_type());
    HloInstruction *one = scalar->parent()->AddInstruction(
        HloInstruction::CreateConstant(one_literal.Clone()));
    TF_ASSIGN_OR_RETURN(scalar, MakeBinaryHlo(HloOpcode::kDivide, one, scalar,
                                              &scalar->metadata()));
  }
  if (scalar->shape().element_type() != F32) {
    scalar = MakeConvertToHlo(scalar, F32, &scalar->metadata());
  }

  return scalar;
}

// A path of instructions by traversing downwards through users, as (op,
// operand_index) pairs. operand_index is the index to get to the previous
// element in the path. I.e.,
// path[i].first->operand(path[i].second) == path[i-1].first
using InstrPath = std::vector<std::pair<HloInstruction *, int>>;

// Transposes a matrix by swapping the contracting and non-contracting
// dimension. There must be only one contracting and only one non-contracting
// dimension. Keeps the layout the same.
HloInstruction *TransposeMatrix(HloInstruction *instr, int64_t contracting_dim,
                                absl::Span<const int64_t> batch_dims) {
  // Identify the dimensional order which describes a transpose of the
  // contracting and non-contracting dimensions of the GEMM.
  std::vector<int64_t> permutation(instr->shape().dimensions_size(), -1);
  // Discard the batch dimensions.
  for (int64_t batch_dim : batch_dims) {
    permutation[batch_dim] = batch_dim;
  }
  // Identify the non-contracting dimension.
  int non_contracting_dim;
  for (int i = 0; i < instr->shape().dimensions_size(); ++i) {
    if (permutation[i] == -1 && contracting_dim != i) {
      non_contracting_dim = i;
    }
  }
  permutation[non_contracting_dim] = contracting_dim;
  permutation[contracting_dim] = non_contracting_dim;

  Shape new_shape = ShapeUtil::PermuteDimensions(permutation, instr->shape());
  *new_shape.mutable_layout() = instr->shape().layout();
  return instr->AddInstruction(
      HloInstruction::CreateTranspose(new_shape, instr, permutation));
}

// If the bias is a sequence of ops that depend only on broadcasts of
// constants, materialize the bias if it's small.
//
// Normally the constant-folding pass would materialize the bias if it is
// calculated entirely from constants. But if the bias is a broadcast of a
// constant, constant-folding won't expand the broadcast, on the theory that
// folding broadcasts of constants causes us to consume more memory and can
// actually make things slower (because any op which reads the constant has
// to read more memory).
//
// OTOH in our case, we don't want to run an op that just broadcasts a
// constant so we can fuse it into this gemm. That would defeat the whole
// purpose of this fusion, which is to launch fewer kernels.  So if we can,
// we expand out this constant ourselves.
//
// TODO(b/192499646): Even better would be to use cublasLT to fuse the
// broadcasted bias, if it supports that fusion efficiently.
HloInstruction *MaybeConstantFoldBias(HloInstruction *bias) {
  // This limit was not chosen carefully.
  constexpr int kMaxMaterializeBiasBytes = 8 * 1024 * 1024;

  // Don't fold broadcasts of scalars -- algsimp will just collapse it again.
  auto is_nonscalar = [](const HloInstruction *instr) {
    return !ShapeUtil::IsEffectiveScalar(instr->shape());
  };

  // For now, only fold broadcast(constant) or
  // reshape/transpose/bitcast(broadcast(constant)). This lets us avoid the
  // complexity in the constant-folding pass about what is and isn't legal to
  // fold.
  auto broadcast_of_nonscalar =
      m::Broadcast(m::Constant().WithPredicate(is_nonscalar));

  if (ShapeUtil::ByteSizeOf(bias->shape()) <= kMaxMaterializeBiasBytes &&
      (Match(bias, broadcast_of_nonscalar) ||
       Match(bias, m::Reshape(broadcast_of_nonscalar)) ||
       Match(bias, m::Transpose(broadcast_of_nonscalar)) ||
       Match(bias, m::Bitcast(broadcast_of_nonscalar)))) {
    HloEvaluator evaluator(/*max_loop_iterations=*/0);
    Literal result;
    if (evaluator.TryEvaluate(
            bias, &result,
            /*recursively_evaluate_nonconstant_operands=*/true)) {
      return bias->parent()->AddInstruction(
          HloInstruction::CreateConstant(std::move(result)));
    }
  }

  return bias;
}

auto Gemm(HloInstruction **instr) {
  return m::CustomCall(instr, {kGemmCallTarget});
}

auto CublasLtMatmul(HloInstruction **instr) {
  return m::CustomCall(instr, {kCublasLtMatmulCallTarget});
}

auto GemmOrCublasLtMatmul(HloInstruction **instr) {
  return m::CustomCall(instr, {kGemmCallTarget, kCublasLtMatmulCallTarget});
}

auto BcastConstScalar(HloInstruction **instr, double value) {
  return m::Broadcast(instr, m::ConstantScalar(value));
}

auto BcastConstScalar(double value) { return BcastConstScalar(nullptr, value); }

auto BcastConstScalarNear(double value) {
  return m::Broadcast(m::ConstantScalar().WithPredicate(
      [expected = value](const HloInstruction *instr) {
        // Not a very robust floating-point comparison, but good enough for our
        // purposes.
        std::optional<double> actual =
            xla::Cast<const HloConstantInstruction>(instr)
                ->literal()
                .GetAsDouble({});
        if (!actual.has_value()) return false;
        double epsilon;
        switch (instr->shape().element_type()) {
          case F16:
            epsilon = 128 * std::numeric_limits<Eigen::half>::epsilon();
            break;
          case BF16:
            epsilon = 128 * std::numeric_limits<bfloat16>::epsilon();
            break;
          case F32:
            epsilon = 128 * std::numeric_limits<float>::epsilon();
            break;
          case F64:
            epsilon = 128 * std::numeric_limits<double>::epsilon();
            break;
          default:
            return false;
        }
        return abs(*actual - expected) < (abs(*actual + expected) * epsilon);
      }));
}

template <typename Pattern>
auto OptionalSlice(HloInstruction **optional_slice, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Slice(optional_slice, pattern),
                                  std::move(pattern));
}

template <typename Pattern>
auto OptionalConvert(HloInstruction **optional_convert, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Convert(optional_convert, pattern),
                                  std::move(pattern));
}

template <typename Pattern>
auto OptionalBitcast(HloInstruction **optional_bitcast, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Bitcast(optional_bitcast, pattern),
                                  std::move(pattern));
}

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
//
// For scaled FP8 GEMMs on Hopper systems, the following steps mentioned in
// RFC #22 (https://github.com/openxla/xla/discussions/22) are elided and
// rewritten into a Custom Call:
//
// 1. Cast each input from FP8 to a wider type such as FP16 or FP32.
// 2. Unscale each input by multiplying each input by the corresponding input
// scale.
// 3. Evaluate the matrix multiplication on the scaled inputs.
// 4. Compute the maximum of the absolute values in the result of the GEMM
// (DAmax).
// 5. Scale the output by dividing the output by the output scale.
// 6. Cast the output back to FP8. Since saturation should be done on
// overflow, this is represented by a Clamp instruction followed by a Convert
// instruction.

// Steps 1 through 3 can be elided independently of the remainder. Steps 5 and
// 6 are elided only if steps 1 through 3 were successfully transformed. Step
// 4 requires steps 5 and 6, i.e. the computation of DAmax can be elided only
// when the output of the GEMM is requested in FP8 format.
class GemmRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmRewriterVisitor(const GpuVersion &gpu_version)
      : gpu_version_(gpu_version) {}

  Status HandleDot(HloInstruction *instr) override {
    if (!IsMatrixMultiplication(*instr) &&
        !IsMatrixVectorMultiplication(*instr)) {
      return absl::OkStatus();
    }
    // Sparse dot is not supported.
    // if (Cast<HloDotInstruction>(instr)->sparse_operands()) {
    //   return absl::OkStatus();
    // }

    int64_t gemm_rewrite_size_threshold =
        instr->GetModule()
            ->config()
            .debug_options()
            .xla_gpu_gemm_rewrite_size_threshold();
    TF_ASSIGN_OR_RETURN(bool is_matmul_tiny,
                        IsMatrixMultiplicationTooSmallForRewriting(
                            *instr, gemm_rewrite_size_threshold));
    if (is_matmul_tiny && IsDotSupportedByClassicalEmitters(*instr)) {
      return absl::OkStatus();
    }

    CHECK(!instr->IsRank2Transpose());
    if (instr->operand(0)->IsRank2Transpose() ||
        instr->operand(1)->IsRank2Transpose()) {
      return absl::OkStatus();
    }
    // Create a GemmBackendConfig based on the instruction.
    GemmBackendConfig gemm_backend_config;
    gemm_backend_config.set_alpha_real(1.0);
    gemm_backend_config.set_alpha_imag(0.0);
    gemm_backend_config.set_beta(0.0);
    *gemm_backend_config.mutable_dot_dimension_numbers() =
        instr->dot_dimension_numbers();
    *gemm_backend_config.mutable_precision_config() = instr->precision_config();

    HloInstruction *lhs = instr->mutable_operand(0);
    HloInstruction *rhs = instr->mutable_operand(1);
    auto attributes = instr->frontend_attributes().map();
    gemm_backend_config.set_grad_x(attributes["grad_x"] == "true");
    gemm_backend_config.set_grad_y(attributes["grad_y"] == "true");

    int64_t lhs_batch_dims_size =
        instr->dot_dimension_numbers().lhs_batch_dimensions_size();
    bool is_lhs_vector =
        lhs->shape().dimensions_size() == lhs_batch_dims_size + 1;
    bool is_rhs_vector =
        rhs->shape().dimensions_size() == lhs_batch_dims_size + 1;
    int64_t lhs_stride =
        is_lhs_vector ? lhs->shape().dimensions(lhs_batch_dims_size)
                      : lhs->shape().dimensions(lhs_batch_dims_size) *
                            lhs->shape().dimensions(lhs_batch_dims_size + 1);
    int64_t rhs_stride =
        is_rhs_vector ? rhs->shape().dimensions(lhs_batch_dims_size)
                      : rhs->shape().dimensions(lhs_batch_dims_size) *
                            rhs->shape().dimensions(lhs_batch_dims_size + 1);

    gemm_backend_config.set_lhs_stride(lhs_stride);
    gemm_backend_config.set_rhs_stride(rhs_stride);

    {
      // Rewrite non-FP8 GEMMs into a cublas or cublasLT Custom Call.
      TF_ASSIGN_OR_RETURN(
          absl::string_view gemm_custom_call_target,
          GetNonFp8GemmCustomCallTarget(*instr, gemm_backend_config));
      const Shape &output_shape = instr->shape();
      HloInstruction *gemm_call =
          instr->AddInstruction(HloInstruction::CreateCustomCall(
              output_shape,
              {instr->mutable_operand(0), instr->mutable_operand(1)},
              gemm_custom_call_target));
      TF_RETURN_IF_ERROR(gemm_call->set_backend_config(gemm_backend_config));
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, gemm_call));
    }
    return absl::OkStatus();
  }

  Status HandleMultiply(HloInstruction *instr) override {
    HloInstruction *alpha, *existing_gemm;
    if (Match(instr,
              m::MultiplyAnyOrder(
                  GemmOrCublasLtMatmul(&existing_gemm).WithOneUser(),
                  m::Broadcast(m::ConstantScalar(&alpha)).WithOneUser()))) {
      TF_ASSIGN_OR_RETURN(auto config,
                          existing_gemm->backend_config<GemmBackendConfig>());
      // Do not fuse alpha into S32 GEMM, as they only support fixed values for
      // alpha/beta.
      if (existing_gemm->shape().element_type() == S32) {
        return absl::OkStatus();
      }

      if (config.beta() == 0.0 && existing_gemm->user_count() == 1) {
        complex128 prev_alpha = {config.alpha_real(), config.alpha_imag()};
        complex128 new_alpha =
            *alpha->literal().GetAsComplex128({}) * prev_alpha;
        config.set_alpha_real(new_alpha.real());
        config.set_alpha_imag(new_alpha.imag());
        TF_RETURN_IF_ERROR(existing_gemm->set_backend_config(config));
        return ReplaceInstruction(instr, existing_gemm);
      }
    }

    // Attempt to match approximate GELU activation
    // (https://arxiv.org/abs/1606.08415), where:
    // approx_gelu(x) = x * cdf(x)
    // cdf(x) = 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3))
    HloInstruction *cdf, *slice_or_bitcast = nullptr;
    if (Match(instr, m::MultiplyAnyOrder(
                         m::AnyOf<HloInstruction>(
                             m::Slice(&slice_or_bitcast,
                                      CublasLtMatmul(&existing_gemm)),
                             m::Bitcast(&slice_or_bitcast,
                                        CublasLtMatmul(&existing_gemm)),
                             CublasLtMatmul(&existing_gemm)),
                         m::Op(&cdf).WithOneUser())) &&
        Match(cdf,
              m::MultiplyAnyOrder(
                  BcastConstScalar(0.5),
                  m::AddAnyOrder(
                      BcastConstScalar(1.0),
                      m::Tanh(
                          m::MultiplyAnyOrder(
                              BcastConstScalarNear(sqrt(M_2_PI)),
                              m::AddAnyOrder(
                                  m::Op().Is(slice_or_bitcast ? slice_or_bitcast
                                                              : existing_gemm),
                                  m::MultiplyAnyOrder(
                                      BcastConstScalarNear(0.044715),
                                      m::MultiplyAnyOrder(
                                          m::Op().Is(slice_or_bitcast
                                                         ? slice_or_bitcast
                                                         : existing_gemm),
                                          m::MultiplyAnyOrder(
                                              m::Op().Is(slice_or_bitcast
                                                             ? slice_or_bitcast
                                                             : existing_gemm),
                                              m::Op().Is(slice_or_bitcast
                                                             ? slice_or_bitcast
                                                             : existing_gemm))
                                              .WithOneUser())
                                          .WithOneUser())
                                      .WithOneUser())
                                  .WithOneUser())
                              .WithOneUser())
                          .WithOneUser())))) {
      return FuseGeluActivation(instr, existing_gemm, slice_or_bitcast);
    }
    return absl::OkStatus();
  }

  // Fuse the scaling of an FP8 GEMM into the Custom Call.
  Status HandleDivide(HloInstruction *instr) override {
    return absl::OkStatus();
  }

  Status HandleAdd(HloInstruction *instr) override {
    HloInstruction *bias, *existing_gemm = nullptr;
    HloInstruction *optional_slice = nullptr;
    HloInstruction *optional_convert = nullptr;
    HloInstruction *optional_bitcast = nullptr;
    // Attempt to elide broadcast and fuse addition of a vector bias into
    // GEMM, including when slicing is applied to the result.
    if (Match(instr,
              m::AddAnyOrder(
                  OptionalBitcast(
                      &optional_bitcast,
                      OptionalSlice(
                          &optional_slice,
                          CublasLtMatmul(&existing_gemm).WithOneUser())
                          .WithOneUser())
                      .WithOneUser(),
                  m::Broadcast(&bias,
                               OptionalConvert(&optional_convert, m::Op()))))) {
      TF_ASSIGN_OR_RETURN(
          bool was_fused,
          FuseVectorBiasAdd(instr, bias, existing_gemm, optional_slice,
                            optional_convert, optional_bitcast));

      if (was_fused) {
        return absl::OkStatus();
      }
    }
    // Attempt to elide broadcast and fuse addition of a vector bias into
    // *batched* GEMM as a matrix bias addition using FuseMatrixBiasAdd.
    // add(bitcast(gemm(a, b)), broadcast(bias)) ->
    //   bitcast(add(gemm(a, b), bitcast(broadcast(bias)))) ->
    //   bitcast(gemm(a, b, bitcast(broadcast(bias)))) (FuseMatrixBiasAdd)
    //
    if (Match(
            instr,
            m::AddAnyOrder(
                m::Bitcast(CublasLtMatmul(&existing_gemm).WithOneUser())
                    .WithOneUser(),
                m::Broadcast(&bias, m::Op()).WithOneUser()))) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_add,
          MakeBinaryHlo(HloOpcode::kAdd, existing_gemm,
                        MakeBitcastHlo(bias, existing_gemm->shape())));
      TF_RETURN_IF_ERROR(
          ReplaceInstruction(instr, MakeBitcastHlo(new_add, instr->shape())));

      // Continue below.
      instr = new_add;
    }

    // Do not fuse broadcast unless we can fuse its input, as it will cause
    // broadcast materialization.
    auto is_not_broadcast = [](const HloInstruction *instr) {
      return instr->opcode() != HloOpcode::kBroadcast;
    };

    // add(bitcast(gemm(a, b)), bias) ->
    //   bitcast(add(gemm(a, b), bitcast(bias))) ->
    //   bitcast(gemm(a, b, bitcast(bias))) (later down in this function).
    //
    // We see this idiom in models that contain batch-dots, where we cast
    // between a rank-2 shape for non-batch dots and a higher-rank shape for
    // batch-dots.
    //
    // The last stage of the transform may fail (because of any of the checks in
    // FuseMatrixBiasAdd), but if so that's okay -- we'll have done a useless
    // transformation, but it doesn't hurt anything.
    if (Match(instr,
              m::AddAnyOrder(
                  m::Bitcast(
                      GemmOrCublasLtMatmul(&existing_gemm).WithOneUser())
                      .WithOneUser(),
                  m::Op(&bias).WithPredicate(is_not_broadcast)))) {
      HloInstruction *new_bitcast =
          MakeBitcastHlo(bias, existing_gemm->shape(), &bias->metadata());
      TF_ASSIGN_OR_RETURN(HloInstruction * new_add,
                          MakeBinaryHlo(HloOpcode::kAdd, existing_gemm,
                                        new_bitcast, &bias->metadata()));
      TF_RETURN_IF_ERROR(
          ReplaceInstruction(instr, MakeBitcastHlo(new_add, instr->shape())));

      // Continue below transforming new_add.
      instr = new_add;
    }

    // Attempt to fuse matrix bias into gemm with optional convert
    // add(convert(gemm(a, b)), c) -> gemm(a, b, c)
    // add(gemm(a, b), c) -> gemm(a, b, c)
    if (Match(instr,
              m::AddAnyOrder(
                  m::AnyOf<HloInstruction>(
                      GemmOrCublasLtMatmul(&existing_gemm).WithOneUser(),
                      m::Convert(
                          GemmOrCublasLtMatmul(&existing_gemm).WithOneUser())
                          .WithOneUser()),
                  m::Op(&bias).WithPredicate(is_not_broadcast)))) {
      TF_ASSIGN_OR_RETURN(auto gemm_backend_config,
                          existing_gemm->backend_config<GemmBackendConfig>());
      // check if type combination is supported here
      TF_ASSIGN_OR_RETURN(
          bool types_are_supported,
          IsLegacyCublasMatmul(*existing_gemm)
              ? TypesAreSupportedByLegacyCublas(*existing_gemm,
                                                gemm_backend_config, instr)
              : TypesAreSupportedByCublasLt(*existing_gemm, gemm_backend_config,
                                            instr));

      // for mix type gemm, only fuse add if there is no consumers
      // ROOT add
      // ROOT tuple(add)
      bool has_no_consumer =
          instr->shape().element_type() ==
              existing_gemm->shape().element_type() ||
          instr->user_count() == 0 ||
          (instr->user_count() == 1 &&
           instr->users()[0]->opcode() == HloOpcode::kTuple &&
           instr->users()[0]->user_count() == 0);

      if (types_are_supported && has_no_consumer) {
        return FuseMatrixBiasAdd(instr, bias, existing_gemm);
      }
    }

    HloInstruction *optional_bitcast_matrix = nullptr;
    HloInstruction *optional_slice_matrix = nullptr;
    if (Match(instr,
              m::AddAnyOrder(
                  OptionalBitcast(
                      &optional_bitcast_matrix,
                      OptionalSlice(&optional_slice_matrix,
                                    GemmOrCublasLtMatmul(&existing_gemm)
                                        .WithOneUser()))
                      .WithOneUser(),
                  m::Op(&bias).WithPredicate(is_not_broadcast)))) {
      // The matrix bias must not be FP8, see
      // https://docs.nvidia.com/cuda/cublas/index.html.
      if (!IsF8Type(bias)) {
        return FuseMatrixBiasAdd(instr, bias, existing_gemm,
                                 optional_bitcast_matrix,
                                 optional_slice_matrix);
      }
    }

    return absl::OkStatus();
  }

  Status HandleMaximum(HloInstruction *instr) override {
    HloInstruction *existing_gemm, *zeros;
    HloInstruction *optional_slice_or_bitcast = nullptr;
    // Attempt to elide maximum and fuse ReLU activation into GEMM, including
    // when slicing or bitcasting is applied to the result.
    if (Match(instr,
              m::MaximumAnyOrder(
                  m::AnyOf<HloInstruction>(
                      m::Slice(
                          &optional_slice_or_bitcast,
                          CublasLtMatmul(&existing_gemm).WithOneUser()),
                      m::Bitcast(
                          &optional_slice_or_bitcast,
                          CublasLtMatmul(&existing_gemm).WithOneUser()),
                      CublasLtMatmul(&existing_gemm))
                      .WithOneUser(),
                  m::Broadcast(&zeros, m::ConstantScalar(0))))) {
      TF_RETURN_IF_ERROR(FuseReluActivation(instr, zeros, existing_gemm,
                                            optional_slice_or_bitcast));
    }
    return absl::OkStatus();
  }

  Status HandleConvert(HloInstruction *instr) override {
    return absl::OkStatus();
  }

  static bool IsCuda(const GpuVersion &gpu_version) {
    return std::holds_alternative<se::CudaComputeCapability>(gpu_version);
  }

  static StatusOr<se::CudaComputeCapability> GetCudaComputeCapability(
      const GpuVersion &gpu_version) {
    auto *cuda_cc = std::get_if<se::CudaComputeCapability>(&gpu_version);
    if (cuda_cc == nullptr) {
      return absl::InvalidArgumentError("Compute Capability is not CUDA.");
    }
    return *cuda_cc;
  }

  static bool IsRocm(const GpuVersion &gpu_version) {
    return std::holds_alternative<se::RocmComputeCapability>(gpu_version);
  }

  static StatusOr<se::RocmComputeCapability> GetRocmComputeCapability(
      const GpuVersion &gpu_version) {
    auto rocm_cc = std::get_if<se::RocmComputeCapability>(&gpu_version);
    if (rocm_cc == nullptr) {
      return absl::InvalidArgumentError("Compute Capability is not ROCm.");
    }
    return *rocm_cc;
  }

  // Fuses a matrix bias into a cuBLAS call. 'instr' should be an Add
  // instruction in the following form:
  //   Add(OptionalBitcast(OptionalSlice(gemm)), bias)
  // where 'gemm' is expected to be a cuBLAS custom_call. Slice is introduced
  // when the inputs of the gemm are possibly padded. Bitcast is introduced to
  // handle high rank input.
  Status FuseMatrixBiasAdd(HloInstruction *instr, HloInstruction *bias,
                                 const HloInstruction *gemm,
                                 HloInstruction *bitcast = nullptr,
                                 HloInstruction *slice = nullptr) {
    TF_RET_CHECK(Shape::Equal().IgnoreElementType()(bias->shape(),
                                                    bitcast ? bitcast->shape()
                                                    : slice ? slice->shape()
                                                            : gemm->shape()));

    // Do not fuse bias into S32 GEMM, as for this datatype cuBLAS only
    // supports fixed values for alpha/beta.
    if (gemm->shape().element_type() == S32) {
      return absl::OkStatus();
    }

    // To ensure correctness, only slices that chop off the ends of dimensions
    // are supported.
    if (slice) {
      int slice_op_dim = slice->operand(0)->shape().rank();
      if (slice->slice_starts() != std::vector<int64_t>(slice_op_dim, 0) ||
          slice->slice_strides() != std::vector<int64_t>(slice_op_dim, 1)) {
        return absl::OkStatus();
      }
    }
    // Cublas gemm overwrites the bias matrix, so fusion is only possible if the
    // gemm is the only user. CublasLt gemm can operate out-of-place.
    bool can_overwrite_bias = [bias]() {
      if (bias->user_count() > 1) {
        // There is another user of the data, do not overwrite it.
        return false;
      }

      if (bias->opcode() != HloOpcode::kParameter) {
        // Not a parameter; can overwrite.
        return true;
      }

      // The bias is a parameter of the computation; check if it is aliased.
      if (!bias->parent()->IsEntryComputation()) {
        // Only the HloModule has input/output aliasing, since this is not the
        // entry computation, there are no guarantees about aliasing; do not
        // overwrite.
        return false;
      }
      const auto &in_out_alias_config =
          bias->GetModule()->input_output_alias_config();
      // If the parameter is aliased, we can overwrite it.
      // TODO(victorstone): The assumption when calling ParameterHasAlias is
      // that bias is not a tuple. This is why we pass {} as the argument for
      // param_index.
      return in_out_alias_config.ParameterHasAlias(bias->parameter_number(),
                                                   /*param_index=*/{});
    }();
    bool want_to_fuse_bias = IsCublasLtMatmul(*gemm) || can_overwrite_bias;

    TF_ASSIGN_OR_RETURN(auto config, gemm->backend_config<GemmBackendConfig>());
    // It is possible to fuse into a cublasLt matmul that already has a vector
    // bias, but no other epilogue will commute with the matrix bias add.
    bool supported_epilogue =
        ((config.epilogue() == GemmBackendConfig::DEFAULT) ||
         (config.epilogue() == GemmBackendConfig::BIAS));

    if ((config.beta() != 0) || !want_to_fuse_bias ||
        (gemm->user_count() != 1) || !supported_epilogue) {
      return absl::OkStatus();
    }

    config.set_beta(1.0);

    std::vector<HloInstruction *> operands(gemm->operands().begin(),
                                           gemm->operands().end());
    HloInstruction *maybe_constant_folded_bias = MaybeConstantFoldBias(bias);
    if (bitcast) {
      maybe_constant_folded_bias =
          instr->AddInstruction(HloInstruction::CreateBitcast(
              slice->shape(), maybe_constant_folded_bias));
    }

    maybe_constant_folded_bias =
        PadOperandToTargetShape(gemm->shape(), maybe_constant_folded_bias);

    operands.insert(operands.begin() + 2, maybe_constant_folded_bias);

    std::unique_ptr<HloInstruction> fused_op =
        gemm->CloneWithNewOperands(gemm->shape(), operands);
    // set output shape to bias shape if mix type
    fused_op->mutable_shape()->set_element_type(bias->shape().element_type());
    TF_RETURN_IF_ERROR(fused_op->set_backend_config(config));

    // Choose whether the bias must alias the output. Legacy cublas GEMMs must
    // operate in place and alias the bias with the output, whereas with
    // cublasLt we can choose.
    //
    // Operating in place is always safe; copy-insertion will insert copies if
    // necessary.  But (we assume) copying is slower than operating
    // out-of-place, so for cublasLt (where we have the choice), we try to
    // operate in place if we think it a copy won't be necessary.
    //
    // We assume that parameters are always read-only and therefore we'd need to
    // copy if we were going to operate in place. (This is not quite true; the
    // param could have input/output aliasing.)  We also assume that if there
    // are other uses of the bias, we might need to copy.  (Again, not quite
    // true if those uses all come before this operation.  But copy-insertion
    // runs before scheduling, so it can't know and has to conservatively insert
    // copies.)
    if (IsLegacyCublasMatmul(*fused_op) || can_overwrite_bias) {
      xla::Cast<HloCustomCallInstruction>(fused_op.get())
          ->set_output_to_operand_aliasing({{{}, {2, {}}}});
    }
    TF_RETURN_IF_ERROR(SetName(instr->GetModule(), fused_op.get()));
    if (slice) {
      fused_op = slice->CloneWithNewOperands(
          slice->shape(),
          {slice->parent()->AddInstruction(std::move(fused_op))});
    }

    if (bitcast) {
      fused_op = bitcast->CloneWithNewOperands(
          bitcast->shape(),
          {bitcast->parent()->AddInstruction(std::move(fused_op))});
    }

    return ReplaceWithNewInstruction(instr, std::move(fused_op));
  }

  // Fuses a vector bias into a cuBLAS call. 'instr' should be an Add
  // instruction in the following form:
  //   Add(OptionalBitcast(OptionalSlice(gemm)), Broadcast(OptionalConvert()))
  // where 'gemm' is expected to be a cuBLAS custom_call. The optional
  // convert is only used for F8 matmuls as cublasLt has specific constraints
  // on the vector bias type for such matmuls. The optional bitcast is
  // necessary to handle high rank input cases.
  StatusOr<bool> FuseVectorBiasAdd(HloInstruction *instr,
                                         HloInstruction *broadcast,
                                         HloInstruction *gemm,
                                         HloInstruction *slice = nullptr,
                                         HloInstruction *convert = nullptr,
                                         HloInstruction *bitcast = nullptr) {
    if (!bitcast) {
      TF_RET_CHECK(ShapeUtil::Compatible(
          broadcast->shape(), (slice ? slice->shape() : gemm->shape())));
    }
    // Verify that the data type is supported by Epilogue Fusion.
    if (!SupportsEpilogueFusion(gemm->shape().element_type())) {
      return false;
    }

    HloInstruction *bias = broadcast->mutable_operand(0);

    TF_ASSIGN_OR_RETURN(auto config,
                        gemm->backend_config<GemmBackendConfig>());
    // # output column dims == # non-contracting rhs operand dims.
    const DotDimensionNumbers &dot_dims = config.dot_dimension_numbers();
    size_t num_col_dims = gemm->operand(1)->shape().rank() -
                          dot_dims.rhs_batch_dimensions_size() -
                          dot_dims.rhs_contracting_dimensions_size();

    if ((gemm->user_count() != 1) ||
        (config.epilogue() != GemmBackendConfig::DEFAULT) ||
        (bias->shape().rank() != num_col_dims)) {
      return false;
    }
    // We require the bias vector to have been broadcast in the most major
    // dimensions; i.e. its most minor physical dimensions align with most minor
    // physical dimensions of the gemm output.
    absl::Span<const int64_t> broadcast_dims = broadcast->dimensions();
    for (size_t i = 0; i < num_col_dims; ++i) {
      int64_t dim =
          (bitcast ? bitcast : gemm)->shape().layout().minor_to_major(i);

      // Find the corresponding dimension from the bias vector.
      auto it = absl::c_find(broadcast_dims, dim);

      if (it == broadcast_dims.end()) {
        return false;
      }

      int64_t vector_dim = it - broadcast_dims.begin();
      if (bias->shape().layout().minor_to_major(i) != vector_dim) {
        return false;
      }
    }

    std::vector<HloInstruction *> operands(gemm->operands().begin(),
                                           gemm->operands().end());

    // Replace add(gemm, broadcast) with fused new_gemm.
    operands.push_back(bias);
    config.set_epilogue(GemmBackendConfig::BIAS);
    std::unique_ptr<HloInstruction> result =
        gemm->CloneWithNewOperands(gemm->shape(), operands);
    TF_RETURN_IF_ERROR(result->set_backend_config(config));
    TF_RETURN_IF_ERROR(SetName(result->GetModule(), result.get()));
    if (slice) {
      result = slice->CloneWithNewOperands(
          slice->shape(), {slice->parent()->AddInstruction(std::move(result))});
    }

    if (bitcast) {
      result = bitcast->CloneWithNewOperands(
          bitcast->shape(),
          {bitcast->parent()->AddInstruction(std::move(result))});
    }
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(instr, std::move(result)));
    return true;
  }

  Status FuseReluActivation(HloInstruction *instr,
                                  HloInstruction *broadcast,
                                  HloInstruction *gemm,
                                  HloInstruction *slice_or_bitcast = nullptr) {
    TF_RET_CHECK(ShapeUtil::Compatible(
        broadcast->shape(),
        (slice_or_bitcast ? slice_or_bitcast->shape() : gemm->shape())));

    if (!SupportsEpilogueFusion(gemm->shape().element_type())) {
      return absl::OkStatus();
    }

    if (gemm->user_count() != 1) {
      return absl::OkStatus();
    }

    TF_ASSIGN_OR_RETURN(auto config,
                        gemm->backend_config<GemmBackendConfig>());
    if (config.epilogue() == GemmBackendConfig::DEFAULT) {
      config.set_epilogue(GemmBackendConfig::RELU);
    } else if (config.epilogue() == GemmBackendConfig::BIAS) {
      config.set_epilogue(GemmBackendConfig::BIAS_RELU);
    } else {
      return absl::OkStatus();
    }

    std::unique_ptr<HloInstruction> result = gemm->Clone();
    TF_RETURN_IF_ERROR(result->set_backend_config(config));
    TF_RETURN_IF_ERROR(SetName(result->GetModule(), result.get()));

    if (slice_or_bitcast) {
      result = slice_or_bitcast->CloneWithNewOperands(
          slice_or_bitcast->shape(),
          {slice_or_bitcast->parent()->AddInstruction(std::move(result))});
    }

    return ReplaceWithNewInstruction(instr, std::move(result));
  }

  Status FuseGeluActivation(HloInstruction *multiply,
                                  HloInstruction *gemm,
                                  HloInstruction *slice_or_bitcast = nullptr) {
    if (!SupportsEpilogueFusion(gemm->shape().element_type())) {
      return absl::OkStatus();
    }
    // There are four users of the gemm output within the GELU calculation.
    bool has_aux = gemm->user_count() > 4;

    TF_ASSIGN_OR_RETURN(auto config,
                        gemm->backend_config<GemmBackendConfig>());

    if (config.epilogue() == GemmBackendConfig::DEFAULT) {
      config.set_epilogue(has_aux ? GemmBackendConfig::GELU_AUX
                                  : GemmBackendConfig::GELU);
    } else if (config.epilogue() == GemmBackendConfig::BIAS) {
      config.set_epilogue(has_aux ? GemmBackendConfig::BIAS_GELU_AUX
                                  : GemmBackendConfig::BIAS_GELU);
    } else {
      return absl::OkStatus();
    }

    std::unique_ptr<HloInstruction> output = gemm->CloneWithNewShape(
        has_aux ? ShapeUtil::MakeTupleShape({gemm->shape(), gemm->shape()})
                : gemm->shape());
    TF_RETURN_IF_ERROR(output->set_backend_config(config));
    TF_RETURN_IF_ERROR(SetName(multiply->GetModule(), output.get()));

    if (slice_or_bitcast) {
      output = slice_or_bitcast->CloneWithNewOperands(
          slice_or_bitcast->shape(),
          {gemm->parent()->AddInstruction(std::move(output))});
    }

    if (has_aux) {
      HloInstruction *tuple_output =
          gemm->parent()->AddInstruction(std::move(output));
      TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
          gemm, HloInstruction::CreateGetTupleElement(tuple_output, 1)));
      output = HloInstruction::CreateGetTupleElement(tuple_output, 0);
    }

    return ReplaceWithNewInstruction(multiply, std::move(output));
  }

 private:
  GpuVersion gpu_version_;

  // Choose cublas or cublasLt for the target of the custom call that instr will
  // be rewritten into.
  StatusOr<absl::string_view> GetNonFp8GemmCustomCallTarget(
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
    return InternalError("Unsupported hipblaslt gemm config: %s", instr.ToString());
    // This case is not supported by cublasLt, fallback to legacy cublas.
    ///return absl::string_view(kGemmCallTarget);
  }

  StatusOr<bool> TypesAreSupportedByLegacyCublas(
      const HloInstruction &instr, const GemmBackendConfig &gemm_backend_config,
      const HloInstruction *bias = nullptr) const {
    // Figure out the Atype/Btype.
    const PrimitiveType a_dtype = instr.operand(0)->shape().element_type();
    const PrimitiveType b_dtype = instr.operand(1)->shape().element_type();
    const PrimitiveType output_type =
        bias ? bias->shape().element_type() : instr.shape().element_type();
    const std::array<PrimitiveType, 12> supported_type = {
        PrimitiveType::S8,  PrimitiveType::F16, PrimitiveType::BF16,
        PrimitiveType::F32, PrimitiveType::S32, PrimitiveType::F64,
        PrimitiveType::C64, PrimitiveType::C128};
    // legacy cublas has a defined set of combinations of types that it
    // supports. Figure out the computeType and scaleType.
    if (!absl::c_linear_search(supported_type, output_type)) return false;
    TF_ASSIGN_OR_RETURN(auto output_dtype, se::gpu::AsBlasDataType(output_type));
    TF_ASSIGN_OR_RETURN(auto blas_a_dtype, se::gpu::AsBlasDataType(a_dtype));
    // TODO(tdanyluk): Investigate why don't we use the actual precision (and
    // algorithm) here? Why do we use the default?
    TF_ASSIGN_OR_RETURN(auto compute_type, se::gpu::GetBlasComputationType(
                            blas_a_dtype, output_dtype,
                            se::blas::kDefaultComputePrecision));
    se::blas::DataType scale_type =
        se::gpu::GetScaleType(output_dtype, compute_type);

    using se::blas::ComputationType;
    using se::blas::DataType;
    // This matrix of supported types is taken directly from cublas
    // documentation.
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex

    using TypeCombinations = std::initializer_list<std::tuple<
        ComputationType, DataType /*scale_type*/, PrimitiveType /*a_dtype*/,
        PrimitiveType /*b_dtype*/, DataType /*output_dtype*/>>;

    const TypeCombinations supported_type_combinations = {
            {ComputationType::kF16, DataType::kHalf, PrimitiveType::F16,
             PrimitiveType::F16, DataType::kHalf},

            {ComputationType::kI32, DataType::kInt32, PrimitiveType::S8,
             PrimitiveType::S8, DataType::kInt32},

            {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
             PrimitiveType::BF16, DataType::kBF16},
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

            // There would be an entry here for A/BType complex int8, but we do
            // not support that type.
            {ComputationType::kF32, DataType::kComplexFloat, PrimitiveType::C64,
             PrimitiveType::C64, DataType::kComplexFloat},

            {ComputationType::kF16AsF32, DataType::kFloat, PrimitiveType::F32,
             PrimitiveType::F32, DataType::kFloat},
            {ComputationType::kF16AsF32, DataType::kComplexFloat,
             PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},

            {ComputationType::kBF16AsF32, DataType::kFloat, PrimitiveType::F32,
             PrimitiveType::F32, DataType::kFloat},
            {ComputationType::kBF16AsF32, DataType::kComplexFloat,
             PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},

            {ComputationType::kTF32AsF32, DataType::kFloat, PrimitiveType::F32,
             PrimitiveType::F32, DataType::kFloat},
            {ComputationType::kTF32AsF32, DataType::kComplexFloat,
             PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},

            {ComputationType::kF64, DataType::kDouble, PrimitiveType::F64,
             PrimitiveType::F64, DataType::kDouble},
            {ComputationType::kF64, DataType::kComplexDouble,
             PrimitiveType::C128, PrimitiveType::C128,
             DataType::kComplexDouble},
        };

    return absl::c_linear_search(
        supported_type_combinations,
        std::make_tuple(compute_type, scale_type, a_dtype, b_dtype,
                        output_dtype));
  }

  StatusOr<bool> TypesAreSupportedByCublasLt(
      const HloInstruction &instr, const GemmBackendConfig &backend_config,
      const HloInstruction *bias = nullptr) const {
    // Figure out the Atype/Btype.
    const PrimitiveType a_dtype = instr.operand(0)->shape().element_type();
    const PrimitiveType b_dtype = instr.operand(1)->shape().element_type();
    const PrimitiveType output_type =
        bias ? bias->shape().element_type() : instr.shape().element_type();
    const std::array<PrimitiveType, 12> supported_type = {
        PrimitiveType::F8E5M2,     PrimitiveType::F8E4M3FN,
        PrimitiveType::S8,         PrimitiveType::F16,
        PrimitiveType::BF16,       PrimitiveType::F32,
        PrimitiveType::S32,        PrimitiveType::F64,
        PrimitiveType::C64,        PrimitiveType::C128};
    if (!absl::c_linear_search(supported_type, output_type)) return false;
    // cublasLt has a defined set of combinations of types that it supports.
    // Figure out the computeType and scaleType.
    TF_ASSIGN_OR_RETURN(auto output_dtype, se::gpu::AsBlasDataType(output_type));
    TF_ASSIGN_OR_RETURN(auto blas_a_dtype, se::gpu::AsBlasDataType(a_dtype));

    const int max_precision = *absl::c_max_element(
        backend_config.precision_config().operand_precision());

    TF_ASSIGN_OR_RETURN(
        auto compute_type,
        se::gpu::GetBlasComputationType(
            blas_a_dtype, output_dtype, max_precision));
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
        // FP8 types:
        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
        //  PrimitiveType::F8E4M3FN, DataType::kBF16},
        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
        //  PrimitiveType::F8E4M3FN, DataType::kF8E4M3FN},
        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
        //  PrimitiveType::F8E4M3FN, DataType::kHalf},
        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
        //  PrimitiveType::F8E4M3FN, DataType::kFloat},

        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
        //  PrimitiveType::F8E5M2, DataType::kBF16},
        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
        //  PrimitiveType::F8E5M2, DataType::kF8E4M3FN},
        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
        //  PrimitiveType::F8E5M2, DataType::kF8E5M2},
        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
        //  PrimitiveType::F8E5M2, DataType::kHalf},
        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
        //  PrimitiveType::F8E5M2, DataType::kFloat},

        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
        //  PrimitiveType::F8E4M3FN, DataType::kBF16},
        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
        //  PrimitiveType::F8E4M3FN, DataType::kF8E4M3FN},
        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
        //  PrimitiveType::F8E4M3FN, DataType::kF8E5M2},
        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
        //  PrimitiveType::F8E4M3FN, DataType::kHalf},
        // {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
        //  PrimitiveType::F8E4M3FN, DataType::kFloat},
        // There would be an entry here for A/BType complex int8, but we do
        // not support that type.
        {ComputationType::kF32, DataType::kComplexFloat, PrimitiveType::C64,
         PrimitiveType::C64, DataType::kComplexFloat},

        {ComputationType::kF16AsF32, DataType::kFloat, PrimitiveType::F32,
         PrimitiveType::F32, DataType::kFloat},
        {ComputationType::kF16AsF32, DataType::kComplexFloat,
         PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},
        // The next 4 may be supported by hipblaslt, but they are not
        // covered by any unit tests
        {ComputationType::kBF16AsF32, DataType::kFloat, PrimitiveType::F32,
         PrimitiveType::F32, DataType::kFloat},
        {ComputationType::kBF16AsF32, DataType::kComplexFloat,
         PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},

        {ComputationType::kTF32AsF32, DataType::kFloat, PrimitiveType::F32,
         PrimitiveType::F32, DataType::kFloat},
        {ComputationType::kTF32AsF32, DataType::kComplexFloat,
         PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},

        {ComputationType::kF64, DataType::kDouble, PrimitiveType::F64,
         PrimitiveType::F64, DataType::kDouble},
        {ComputationType::kF64, DataType::kComplexDouble, PrimitiveType::C128,
         PrimitiveType::C128, DataType::kComplexDouble},
    };
    if (IsCuda(gpu_version_) &&
        absl::c_linear_search(supported_cublas_type_combinations,
                              std::tuple{compute_type, scale_type, a_dtype,
                                         b_dtype, output_dtype})) {
      return true;
    }
    const TypeCombinations supported_type_combinations = {
        // Other data types:

        {ComputationType::kI32, DataType::kInt32, PrimitiveType::S8,
         PrimitiveType::S8, DataType::kInt32},
        {ComputationType::kI32, DataType::kFloat, PrimitiveType::S8,
         PrimitiveType::S8, DataType::kInt8},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::S8,
         PrimitiveType::S8, DataType::kFloat},

        {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
         PrimitiveType::BF16, DataType::kBF16},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
         PrimitiveType::BF16, DataType::kFloat},
        {ComputationType::kBF16AsF32, DataType::kFloat, PrimitiveType::BF16,
         PrimitiveType::BF16, DataType::kBF16},
        {ComputationType::kBF16AsF32, DataType::kFloat, PrimitiveType::BF16,
         PrimitiveType::BF16, DataType::kFloat},

        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F16,
         PrimitiveType::F16, DataType::kHalf},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F16,
         PrimitiveType::F16, DataType::kFloat},
        {ComputationType::kF16AsF32, DataType::kFloat, PrimitiveType::F16,
         PrimitiveType::F16, DataType::kHalf},
        {ComputationType::kF16AsF32, DataType::kFloat, PrimitiveType::F16,
         PrimitiveType::F16, DataType::kFloat},

        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F32,
         PrimitiveType::F32, DataType::kFloat},
        {ComputationType::kTF32AsF32, DataType::kFloat, PrimitiveType::F32,
         PrimitiveType::F32, DataType::kFloat},
    };

    return absl::c_linear_search(
        supported_type_combinations,
        std::make_tuple(compute_type, scale_type, a_dtype, b_dtype,
                        output_dtype));
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
        GemmConfig gemm_config,
        GemmConfig::For(
            lhs->shape(), dot_dims.lhs_batch_dimensions(),
            dot_dims.lhs_contracting_dimensions(), rhs->shape(),
            dot_dims.rhs_batch_dimensions(),
            dot_dims.rhs_contracting_dimensions(),
            /*output_shape=*/instr.shape(), gemm_backend_config.alpha_real(),
            gemm_backend_config.alpha_imag(), gemm_backend_config.beta(),
            /*algorithm*/ se::blas::kDefaultAlgorithm, se::blas::kDefaultComputePrecision,
            se::gpu::BlasLt::Epilogue::kDefault));

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
      return false;
    }

    // The cublasLt API has two currently known limitations:
    // 1. Batch count must be <2^16.
    constexpr int64_t kMaxBatchCount = 65535;
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

    // if (auto isrocm = std::get_if<se::RocmComputeCapability>(&gpu_version_);
    //     isrocm) {
    //   if (!isrocm->has_hipblaslt()) {
    //     return false;
    //   }
    // }

    // 2. cublasLt does not support rhs col dimension size > 4194240 for
    // C64.
    constexpr int kMaxDimensionSize{4194240};
    if (output_shape.element_type() != C64) {
      // Does not match type in unsupported case.
      return true;
    }

    if (std::holds_alternative<se::CudaComputeCapability>(gpu_version_)) {
      auto cuda_compute_capability_ =
          std::get<se::CudaComputeCapability>(gpu_version_);
      if (cuda_compute_capability_.IsAtLeast(
              se::CudaComputeCapability::AMPERE)) {
        // cuBlasLt has an implementation for complex data with compute type
        // 32F_FAST_32TF that uses tensor cores and that is free from the
        // restriction. This implementation only works on Ampere
        // architecture though (where TF32 was introduced).
        return true;
      }
    }
    // Get the rhs non-contracting dimensions as they will eventually be at the
    // cublasLt level.
    std::vector<int64_t> rhs_non_contracting_dims;
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
          GetNonContractingDims(lhs->shape(), dot_dims.lhs_batch_dimensions(),
                                dot_dims.lhs_contracting_dimensions()));
    } else {
      TF_ASSIGN_OR_RETURN(
          rhs_non_contracting_dims,
          GetNonContractingDims(rhs->shape(), dot_dims.rhs_batch_dimensions(),
                                dot_dims.rhs_contracting_dimensions()));
    }

    const auto lhs_non_contracting_dimension_size = absl::c_accumulate(
        rhs_non_contracting_dims, 1, [&](int64_t size, int64_t dim) {
          return size * lhs->shape().dimensions(dim);
        });

    // Check that the size of the non-contracting dimension is not too large.
    return lhs_non_contracting_dimension_size <= kMaxDimensionSize;
  }

};

// Rewriter that adds a workspace to legacy cuBLAS custom calls. We run it
// separately after gemm rewriter, so that we can do pattern matching without
// having to match output tuples.
class GemmWorkspaceRewriteVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmWorkspaceRewriteVisitor(
      const GpuVersion &gpu_version)
      : gpu_version_(gpu_version) {}

  Status HandleCustomCall(HloInstruction *instr) override {
    bool has_aux_output = false;

    // add workspace only for cublas-lt calls
    if (instr->custom_call_target() != kCublasLtMatmulCallTarget) {
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(auto config,
                          instr->backend_config<GemmBackendConfig>());
    GemmBackendConfig_Epilogue epilogue = config.epilogue();
    TF_ASSIGN_OR_RETURN(
          has_aux_output,
          gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));

    if (!((instr->shape().IsTuple() &&
             instr->shape().tuple_shapes_size() ==
                 has_aux_output + /*config.damax_output()*/ + 1) ||
            instr->shape().IsArray())) {
      return absl::OkStatus();
    }

    auto *cuda_cc = std::get_if<se::CudaComputeCapability>(&gpu_version_);

    // Pass a user-managed workspace to legacy cuBLAS operations, as
    // otherwise cuBLAS will use its own internal pool which will be competing
    // with XLA allocator for device memory.
    int64_t workspace = cuda_cc == nullptr ? GemmConfig::kDefaultWorkspace
             : cuda_cc->IsAtLeast(se::CudaComputeCapability::HOPPER)
               ? GemmConfig::kHopperWorkspace
               : GemmConfig::kDefaultWorkspace;

    // We do not know the workspace size required by cuBLAS, but we can guess
    // that in a worst case cuBLAS will transpose all operands into tiled
    // layout optimal for the tensor cores. It doesn't make sense to allocate a
    // larger workspace.
    //
    // TODO(ezhulenev): This is not based on any measurement, just a common
    // sense, we should tweak it to find the minimal workspace size.
    if (instr->custom_call_target() == kGemmCallTarget) {
      int64_t operands_byte_size = 0;
      for (auto &operand : instr->operands()) {
        operands_byte_size += ShapeUtil::ByteSizeOf(operand->shape());
      }
      workspace = std::min(workspace, operands_byte_size);
    }

    // Append workspace buffer to instruction outputs.
    std::vector<Shape> output_shapes = instr->shape().IsArray()
                                           ? std::vector<Shape>{instr->shape()}
                                           : instr->shape().tuple_shapes();
    output_shapes.emplace_back(ShapeUtil::MakeShape(S8, {workspace}));
    Shape output_shape = ShapeUtil::MakeTupleShape(output_shapes);

    // Clone custom call with a new shape.
    HloInstruction *new_call = instr->AddInstruction(
        instr->CloneWithNewOperands(output_shape, instr->operands()));

    // Update operand aliasing if it was a fused gemm with aliased output.
    auto *custom_call = xla::Cast<HloCustomCallInstruction>(new_call);
    if (!custom_call->output_to_operand_aliasing().empty()) {
      custom_call->set_output_to_operand_aliasing({{{0}, {2, {}}}});
    }

    if (instr->shape().IsTuple()) {
      for (auto user : instr->users()) {
        auto user_get_tuple =
            dynamic_cast<HloGetTupleElementInstruction *>(user);
        TF_RET_CHECK(user_get_tuple);
        HloInstruction *get_output =
            instr->AddInstruction(HloInstruction::CreateGetTupleElement(
                new_call, user_get_tuple->tuple_index()));
        TF_RETURN_IF_ERROR(ReplaceInstruction(user_get_tuple, get_output));
      }
      return absl::OkStatus();
    } else {
      HloInstruction *get_output = instr->AddInstruction(
          HloInstruction::CreateGetTupleElement(new_call, 0));
      return ReplaceInstruction(instr, get_output);
    }
  }

 private:
  GpuVersion gpu_version_;
};

StatusOr<bool> RunOnComputation(HloComputation *computation,
                                      const GpuVersion& gpu_version) {
  GemmRewriterVisitor visitor(gpu_version);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  GemmWorkspaceRewriteVisitor workspace_visitor(gpu_version);
  TF_RETURN_IF_ERROR(computation->Accept(&workspace_visitor));
  return visitor.changed();
}

}  // anonymous namespace

GemmRewriter::GemmRewriter(GpuVersion gpu_version)
    : gpu_version_(gpu_version) {}

StatusOr<bool> GemmRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result,
                        RunOnComputation(computation, gpu_version_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
