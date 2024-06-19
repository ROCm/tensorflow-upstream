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

// This transformation pass converts operations in TensorFlow dialect into
// operations that are legal in the TensorFlow Lite dialect.  Operations that
// can be legalized to TensorFlow Lite dialect with simple replacements are part
// of this pass and other operations that may create extra ops should be part of
// the PrepareTF pass which should be run before this pass.  That way any
// constant folding opportunities from the extra ops can be exploited by the
// constant folding support for the TensorFlow ops.

#include <climits>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual LegalizeTF Pass.
namespace {
#define GEN_PASS_DEF_LEGALIZETFPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Legalize operations in functions.
struct LegalizeTF : public impl::LegalizeTFPassBase<LegalizeTF> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeTF)

  void runOnOperation() override;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_legalize_tf.inc"

#define DECL_CONVERT_OP(tf_op)                                             \
  struct ConvertTF##tf_op##Op : public RewritePattern {                    \
    explicit ConvertTF##tf_op##Op(MLIRContext* context)                    \
        : RewritePattern(TF::tf_op##Op::getOperationName(), 1, context) {} \
    LogicalResult matchAndRewrite(                                    \
        Operation* op, PatternRewriter& rewriter) const override;          \
  }

// TODO(antiagainst): Define this pattern in a table-driven manner once variadic
// operands are properly supported in declarative rewrite rule specification.

DECL_CONVERT_OP(Concat);
DECL_CONVERT_OP(ConcatV2);
DECL_CONVERT_OP(MatMul);
DECL_CONVERT_OP(Pack);
DECL_CONVERT_OP(Split);
DECL_CONVERT_OP(SplitV);
DECL_CONVERT_OP(Unpack);

#undef DECL_CONVERT_OP

LogicalResult ConvertTFConcatOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_concat_op = cast<TF::ConcatOp>(op);

  SmallVector<Value*, 4> values(tf_concat_op.getValues());
  auto output_type = tf_concat_op.getOutput()->getType();
  // Extract axis attribute from constant concat_dims tensor
  ElementsAttr axis;
  if (!matchPattern(tf_concat_op.getConcatDim(), m_Constant(&axis)))
    return failure();

  StringAttr fused_activation_function =
      StringAttr::get(rewriter.getContext(), "NONE");
  rewriter.replaceOpWithNewOp<TFL::ConcatenationOp>(
      op, output_type, values, mlir::TFL::ExtractSingleElementAsInteger(axis),
      fused_activation_function);
  return success();
}

LogicalResult ConvertTFConcatV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_concat_op = cast<TF::ConcatV2Op>(op);

  SmallVector<Value*, 4> values(tf_concat_op.getValues());
  auto output_type = tf_concat_op.getOutput().getType();
  // Extract axis attribute from constant axis tensor
  ElementsAttr axis;
  if (!matchPattern(tf_concat_op.getAxis(), m_Constant(&axis)))
    return failure();

  StringAttr fused_activation_function =
      StringAttr::get(rewriter.getContext(), "NONE");
  rewriter.replaceOpWithNewOp<ConcatenationOp>(
      op, output_type, values, ExtractSingleElementAsInteger(axis),
      fused_activation_function);
  return success();
}

// The following is effectively:
// def : Pat<
//   (TF_MatMulOp $a, $b, ConstBoolAttrFalse:$transpose_a,
//      ConstBoolAttrTrue:$transpose_b),
//   (TFL_FullyConnectedOp:$__0 $a, $b,
//     NoInput.pattern, TFL_AF_None, TFL_FCWO_Default, ConstBoolAttrFalse)>;
LogicalResult ConvertTFMatMulOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_matmul_op = cast<TF::MatMulOp>(op);
  if (tf_matmul_op.getTransposeA()) return failure();
  if (!tf_matmul_op.getTransposeB()) return failure();

  Type output_type = tf_matmul_op.getResult().getType();
  // TODO(jpienaar): Follow up post shuffle discussion.
  auto no_input = rewriter.create<TFL::NoValueOp>(
      op->getLoc(), rewriter.getNoneType(), rewriter.getUnitAttr());
  auto fc_op = rewriter.create<FullyConnectedOp>(
      op->getLoc(), ArrayRef<Type>{output_type}, op->getOperand(0),
      op->getOperand(1), no_input, rewriter.getStringAttr("NONE"),
      rewriter.getStringAttr("DEFAULT"), rewriter.getBoolAttr(false));
  rewriter.replaceOp(op, {fc_op.getResult(0)});
  return success();
}

LogicalResult ConvertTFPackOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_pack_op = cast<TF::PackOp>(op);

  SmallVector<Value*, 4> values(tf_pack_op.getValues());
  auto output_type = tf_pack_op.getOutput().getType();
  auto values_count = rewriter.getI32IntegerAttr(tf_pack_op.getN());
  // Axis can be negative.
  auto axis = rewriter.getI32IntegerAttr(tf_pack_op.getAxis());

  rewriter.replaceOpWithNewOp<PackOp>(op, output_type, values, values_count,
                                      axis);
  return success();
}

LogicalResult ConvertTFSplitOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_split_op = cast<TF::SplitOp>(op);

  auto output_types = functional::map([](Value* v) { return v->getType(); },
                                      tf_split_op.getOutput());
  // Number of splits cannot be negative.
  auto num_split =
      rewriter.getI32IntegerAttr(tf_split_op.getNumSplit());

  rewriter.replaceOpWithNewOp<TFL::SplitOp>(op, output_types,
                                            tf_split_op.getSplitDim(),
                                            tf_split_op.getValue(), num_split);
  return success();
}

LogicalResult ConvertTFSplitVOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_splitv_op = cast<TF::SplitVOp>(op);

  auto output_types = functional::map([](Value* v) { return v->getType(); },
                                      tf_splitv_op.getOutput());
  // Number of splits cannot be negative.
  auto num_split =
      rewriter.getI32IntegerAttr(tf_splitv_op.getNumSplit());

  rewriter.replaceOpWithNewOp<TFL::SplitVOp>(
      op, output_types, tf_splitv_op.value(), tf_splitv_op.getSizeSplits(),
      tf_splitv_op.getSplitDim(), num_split);
  return success();
}

LogicalResult ConvertTFUnpackOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_unpack_op = cast<TF::UnpackOp>(op);

  auto* input = tf_unpack_op.getValue();
  auto output_types = functional::map([](Value* v) { return v->getType(); },
                                      tf_unpack_op.getOutput());
  auto num = rewriter.getI32IntegerAttr(tf_unpack_op.getNum());
  // Axis can be negative.
  auto axis = rewriter.getI32IntegerAttr(tf_unpack_op.getAxis());

  rewriter.replaceOpWithNewOp<UnpackOp>(op, output_types, input, num, axis);
  return success();
}

void LegalizeTF::runOnOperation() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  // Add the generated patterns to the list.
  populateWithGenerated(ctx, &patterns);
  patterns.insert<ConvertTFConcatOp, ConvertTFConcatV2Op, ConvertTFMatMulOp,
                  ConvertTFPackOp, ConvertTFSplitOp, ConvertTFSplitVOp,
                  ConvertTFUnpackOp>(ctx);
  applyPatternsGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect LegalizeTF pass.
std::unique_ptr<mlir::Pass> CreateLegalizeTFPass() {
  return std::make_unique<LegalizeTF>();
}

}  // namespace TFL
}  // namespace mlir
