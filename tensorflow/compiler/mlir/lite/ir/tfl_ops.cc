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

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

#include <cstdint>
#include <cmath>
#include "Eigen/Core"  // from @eigen_archive
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/FoldUtils.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_dialect.cc.inc"

#define GET_ATTRDEF_CLASSES
#include "tensorflow/compiler/mlir/lite/ir/attributes.cc.inc"


namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// TensorFlowLiteDialect
//===----------------------------------------------------------------------===//

struct TensorFlowLiteInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(Operation* op, Region* dest, bool wouldBeCloned,
                       IRMapping&) const final {
    // No TFLite op restricts inlining today, revise as needed in the future.
    return true;
  }
  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       IRMapping& valueMapping) const final {
    return isa<WhileOp>(dest->getParentOp());
  }
};

struct TensorFlowLiteDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  // Registered hook to check if the given region, which is attached to an
  // operation that is *not* isolated from above (i.e. no internal regions
  // reference values defined in an enclosing region), should be used when
  // materializing constants.
  // In the TFLite dialect we materialize inside a while regions as slightly
  // more efficient computationally.
  bool shouldMaterializeInto(Region* region) const final {
    return isa<WhileOp>(region->getParentOp());
  }
};

void TFLDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (type.isa<ControlType>()) {
    os << "control";
    return;
  }
  os << "<unknown TFL type>";
}

Type TFLDialect::parseType(DialectAsmParser& parser) const {
  StringRef data_type;
  if (parser.parseKeyword(&data_type)) return Type();
  if (data_type == "control") return ControlType::get(getContext());
  parser.emitError(parser.getNameLoc()) << "unknown TFL type: " << data_type;
  return nullptr;
}

void TFLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tensorflow/compiler/mlir/lite/ir/attributes.cc.inc"
      >();
  addInterfaces<TensorFlowLiteInlinerInterface,
                TensorFlowLiteDialectFoldInterface>();
  addTypes<ControlType>();
}

//===----------------------------------------------------------------------===//
// Common support logic
//===----------------------------------------------------------------------===//

namespace {

// Returns true if the dimensions in `a` is a suffix of the ones in `b`.
// For example, dimensions {2}, {1, 2}, and {3, 1, 2} are all suffixes to
// {5, 4, 3, 1, 2}, while {1}, {5, 4}, and {1, 3, 2} are all not.
inline bool IsTrailingDimensions(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  if (a.size() > b.size()) return false;

  return std::equal(a.rbegin(), a.rend(), b.rbegin());
}

// Returns true if it is a shaped type of f32 elements.
inline bool IsF32ShapedType(Type t) {
  if (auto shaped_type = t.dyn_cast_or_null<ShapedType>()) {
    return shaped_type.getElementType().isF32();
  }
  return false;
}

// Performs const folding `calculate` with broadcast behavior on the two
// attributes `operand1` and `operand2` and returns the result if possible.
// The two operands are expected to both be scalar values.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              llvm::function_ref<ElementValueT(ElementValueT, ElementValueT)>>
Attribute ConstFoldBinaryOpScalarScalar(Type result_type, Attribute operand1,
                                        Attribute operand2,
                                        const CalculationT &calculate) {
  auto lhs = operand1.cast<AttrElementT>();
  auto rhs = operand2.cast<AttrElementT>();

  assert(lhs.getType() == result_type && rhs.getType() == result_type &&
         "values of incompatible types should be caught by op verification");

  // TODO: Need to handle overflow/underflow cases.
  return AttrElementT::get(result_type,
                           calculate(lhs.getValue(), rhs.getValue()));
}

/// Performs const folding `calculate` with broadcast behavior on the two
/// attributes `operand1` and `operand2` and returns the result if possible.
/// This function assumes the both operands are verified to have value
/// attributes of broadcastable types.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              llvm::function_ref<ElementValueT(ElementValueT, ElementValueT)>>
Attribute ConstFoldBinaryOpDenseDense(Type result_type, DenseElementsAttr lhs,
                                      DenseElementsAttr rhs,
                                      const CalculationT &calculate) {
  auto type = result_type.cast<ShapedType>();

  if (lhs.getType() != rhs.getType()) {
    // We only support the case that one of the operand's dimensions are
    // a perfect suffix of the other.
    // TODO: support the general broadcast behavior.
    auto lhs_shape = lhs.getType().getShape();
    auto rhs_shape = rhs.getType().getShape();
    if (IsTrailingDimensions(lhs_shape, rhs_shape)) {
      if (!type.hasStaticShape()) type = rhs.getType();
    } else if (IsTrailingDimensions(rhs_shape, lhs_shape)) {
      if (!type.hasStaticShape()) type = lhs.getType();
    } else {
      return {};
    }
  } else if (!type.hasStaticShape()) {
    type = lhs.getType();
  }

  const bool rhs_is_splat = rhs.isSplat();
  const bool lhs_is_splat = lhs.isSplat();

  // If both of them are splat, compute and return.
  if (lhs_is_splat && rhs_is_splat) {
    auto element_result = AttrElementT::get(
        type.getElementType(), calculate(lhs.getSplatValue<ElementValueT>(),
                                         rhs.getSplatValue<ElementValueT>()));
    if (!element_result) return {};

    return DenseElementsAttr::get(type, element_result);
  }

  auto lhs_num_elements = lhs.getType().getNumElements();
  auto rhs_num_elements = rhs.getType().getNumElements();
  auto num_elements = std::max(lhs_num_elements, rhs_num_elements);

  // We assume the arguments have broadcast-compatible types. Make sure again.
  assert(std::max(lhs_num_elements, rhs_num_elements) == num_elements);
  assert(num_elements % std::min(lhs_num_elements, rhs_num_elements) == 0);

  SmallVector<ElementValueT, 16> lhs_old_values;
  SmallVector<ElementValueT, 16> rhs_old_values;
  if (lhs_is_splat)
    lhs_old_values.push_back(lhs.getSplatValue<ElementValueT>());
  else
    lhs_old_values = llvm::to_vector<16>(lhs.getValues<ElementValueT>());
  if (rhs_is_splat)
    rhs_old_values.push_back(rhs.getSplatValue<ElementValueT>());
  else
    rhs_old_values = llvm::to_vector<16>(rhs.getValues<ElementValueT>());

  SmallVector<ElementValueT, 16> new_values;
  new_values.reserve(num_elements);

  // Add each pair of the corresponding values in the dense elements
  // attributes.
  for (int i = 0; i < num_elements; ++i) {
    // We only support a degenerated case here: the dimensions in one operand's
    // shape is a perfect suffix to the other operand. Then conceptually it's
    // similar to broadcasting a scalar to a 1-D vector.
    // TODO: support the general broadcast behavior.
    // We are tiling the operand with less elements an integral times to match
    // the operand with more elements. We don't care which operand has less
    // elements here because we are iterating its elements in circles, which can
    // be achieved using the result index modulo the element count. For the
    // operand with more elements, since the result has the same number of
    // elements, we are only going over its elements once. The modulo operation
    // also works for that.
    int lhs_index = lhs_is_splat ? 0 : (i % lhs_num_elements);
    int rhs_index = rhs_is_splat ? 0 : (i % rhs_num_elements);

    new_values.push_back(
        calculate(lhs_old_values[lhs_index], rhs_old_values[rhs_index]));
  }

  return DenseElementsAttr::get(type, ArrayRef<ElementValueT>(new_values));
}

/// Performs const folding `calculate` with broadcast behavior on the two
/// attributes `operand1` and `operand2` and returns the result if possible.
/// This function assumes the two operands are verified to have value
/// attributes of broadcastable types.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              llvm::function_ref<ElementValueT(ElementValueT, ElementValueT)>>
Attribute ConstFoldBinaryOp(Type result_type, Attribute operand1,
                            Attribute operand2, const CalculationT& calculate) {
  if (operand1.dyn_cast_or_null<DenseElementsAttr>() &&
      operand2.dyn_cast_or_null<DenseElementsAttr>()) {
    return ConstFoldBinaryOpDenseDense<AttrElementT, ElementValueT>(
        result_type, operand1.cast<DenseElementsAttr>(),
        operand2.cast<DenseElementsAttr>(), calculate);
  }

  // TODO: support other attribute kinds

  return {};
}

/// Performs const folding with broadcast behavior on the two attributes in
/// `operands` and returns the result if possible.
/// Depending on the given `resultType`, either `floatCalculate` or
/// `intCalculate` is chosen to conduct the calculate.
Attribute ConstFoldBinaryOp(
    Type result_type, ArrayRef<Attribute> operands,
    llvm::function_ref<APFloat(APFloat, APFloat)> float_calculate,
    llvm::function_ref<APInt(APInt, APInt)> int_calculate) {
  // Note: All types are wrapped in tensor types in TFlite. E.g., f32 is
  // represented as tensor<f32>. So we are only handling tensor types here.
  auto type = result_type.dyn_cast<ShapedType>();
  if (!type) return {};

  auto elemType = type.getElementType();

  if (elemType.isa<FloatType>())
    return ConstFoldBinaryOp<FloatAttr>(result_type, operands[0], operands[1],
                                        float_calculate);

  if (elemType.isa<IntegerType>())
    return ConstFoldBinaryOp<IntegerAttr>(result_type, operands[0], operands[1],
                                          int_calculate);

  return {};
}

/// Performs const folding a attributes `operand` and returns the result if
/// possible.
/// The function currently asserts that the `result_type` to be a f32 tensor
/// type.
/// TODO: Extend this function to handle integral tensor for ops like
/// "tfl.logical_not".
Attribute ConstFoldUnaryOp(Type result_type, Attribute operand,
                           llvm::function_ref<APFloat(APFloat)> calculate) {
  assert(IsF32ShapedType(result_type));
  auto result_shape_type = result_type.cast<ShapedType>();

  if (auto dense_elements = operand.dyn_cast_or_null<DenseElementsAttr>()) {
    SmallVector<APFloat, 16> new_values;
    const int num_elements = result_shape_type.getNumElements();
    new_values.reserve(num_elements);

    for (APFloat old_value : dense_elements.getValues<APFloat>()) {
      new_values.push_back(calculate(old_value));
    }

    return DenseElementsAttr::get(result_shape_type, new_values);
  }

  return {};
}

static constexpr int64_t kTFDynamicSize = -1;

llvm::SmallVector<int64_t> ConvertTFShapeToMlir(
    llvm::ArrayRef<int64_t> shapes) {
  return llvm::to_vector(llvm::map_range(shapes, [](int64_t shape) {
    return shape == kTFDynamicSize ? mlir::ShapedType::kDynamic : shape;
  }));
}

llvm::SmallVector<int64_t> ConvertMlirShapeToTF(
    llvm::ArrayRef<int64_t> shapes) {
  return llvm::to_vector(llvm::map_range(shapes, [](int64_t shape) {
    return mlir::ShapedType::isDynamic(shape) ? kTFDynamicSize : shape;
  }));
}

mlir::RankedTensorType GetTypeFromTFTensorShape(llvm::ArrayRef<int64_t> shape,
                                                mlir::Type elementType,
                                                mlir::Attribute encoding = {}) {
  return mlir::RankedTensorType::get(ConvertTFShapeToMlir(shape), elementType,
                                     encoding);
}

void buildComparisonBinOp(Builder* builder, OperationState& result, Value lhs,
                          Value rhs) {
  auto result_type =
      OpTrait::util::getBroadcastedType(lhs.getType(), rhs.getType());
  if (!result_type)
    emitError(result.location)
        << "non-broadcastable operands: " << lhs.getType() << " and "
        << rhs.getType();
  result.addOperands({lhs, rhs});
  // Comparison binary ops always return i1 tensor.
  if (auto shaped_type = result_type.dyn_cast<RankedTensorType>()) {
    auto result_shape = shaped_type.getShape();
    result.types.push_back(GetTypeFromTFTensorShape(
        result_shape, builder->getI1Type()));
  } else {
    result.types.push_back(UnrankedTensorType::get(builder->getI1Type()));
  }
}

void buildFusedBroadcastableBinOp(Builder* builder, OperationState& result,
                                  Value lhs, Value rhs,
                                  StringAttr fused_activation_function) {
  auto result_type =
      OpTrait::util::getBroadcastedType(lhs.getType(), rhs.getType());

  if (!result_type)
    emitError(result.location)
        << "non-broadcastable operands: " << lhs.getType() << " and "
        << rhs.getType();

  result.addOperands({lhs, rhs});
  result.addAttribute("fused_activation_function", fused_activation_function);
  result.types.push_back(result_type);
}

}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // TODO(b/142478136): Handle fused ops.
  if (getFusedActivationFunction() != "NONE") return {};
  return ConstFoldBinaryOp(
      getType(), operands, [](APFloat a, APFloat b) { return a + b; },
      [](APInt a, APInt b) { return a + b; });
}

//===----------------------------------------------------------------------===//
// ConcatenationOp
//===----------------------------------------------------------------------===//
// TODO(ashwinm): Implement shape inference for Concatenation

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

static void BuildGatherOp(OpBuilder *builder, OperationState &result,
                          Value params, Value indices, IntegerAttr axis) {
  auto params_type = params.getType().cast<TensorType>();
  auto indices_type = indices.getType().cast<TensorType>();

  // If params/indices is unranked, then output is unranked.
  if (!params_type.hasRank() || !indices_type.hasRank()) {
    TFL::GatherOp::build(
        *builder, result, UnrankedTensorType::get(params_type.getElementType()),
        params, indices, axis);
    return;
  }

  int64_t params_rank = params_type.getRank();
  int64_t indices_rank = indices_type.getRank();

  // params rank is guaranteed to be at least 1.
  // Produces an output tensor with shape:
  // params.shape[:axis] + indices.shape + params.shape[axis + 1:]
  std::vector<int64_t> shape(params_type.getShape());
  int64_t axis_i = axis.getInt();

  // For neg axis values, we wrap around params, e.g. axis = -1 => params[:-1]
  if (axis_i < 0) {
    axis_i += params_rank;
  }

  // params must be atleast rank axis + 1
  if (params_rank < axis_i + 1) {
    emitError(result.location, "params must be atleast rank axis + 1");
  }

  if (indices_rank == 0) {
    // Scalar indices (output is rank(params) - 1).
    // Erase shape[axis]
    shape.erase(shape.begin() + axis_i);
  } else if (indices_rank == 1) {
    // Vector indices (output is rank(params)).
    // Copy indices.shape into params.shape[axis]
    std::copy(std::begin(indices_type.getShape()),
              std::end(indices_type.getShape()), std::begin(shape) + axis_i);
  } else {
    // Higher rank indices (output is rank(params) + rank(indices) - 1).
    shape.resize(params_rank + indices_rank - 1);
    // Copy params.shape[axis + 1: ] into shape[axis + indices_rank:]
    std::copy(std::begin(params_type.getShape()) + axis_i + 1,
              std::end(params_type.getShape()),
              std::begin(shape) + axis_i + indices_rank);

    // Copy indices.shape into params.shape[axis]
    std::copy(std::begin(indices_type.getShape()),
              std::end(indices_type.getShape()), std::begin(shape) + axis_i);
  }

  TFL::GatherOp::build(
      *builder, result,
      GetTypeFromTFTensorShape(shape, params_type.getElementType()),
      params, indices, axis);
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // TODO(b/142478136): Handle fused ops.
  if (getFusedActivationFunction() != "NONE") return {};

  // This function is performance critical for op fusion patterns, e.g.
  // FuseBinaryOpToPrecedingAffine and FuseMulOrDivWithConv2dOrDepthwiseConv2d.
  // So a few specializations are provided to evaluate the math operation
  // more efficiently.

  // Specialization for f32 type.
  if (getType().cast<ShapedType>().getElementType().isF32()) {
    return ConstFoldBinaryOp<FloatAttr, float>(
        getType(), operands[0], operands[1],
        [](float a, float b) { return a * b; });
  }
#if 0
  // Specialization for bf16 type.
  if (getType().cast<ShapedType>().getElementType().isBF16()) {
    return ConstFoldBinaryOp<FloatAttr, Eigen::bfloat16>(
        getType(), operands[0], operands[1],
        [](Eigen::bfloat16 a, Eigen::bfloat16 b) { return a * b; });
  }
#endif
  // Generic fallback with APFloat
  return ConstFoldBinaryOp(
      getType(), operands, [](APFloat a, APFloat b) { return a * b; },
      [](APInt a, APInt b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

// TODO(b/133486129): Implement shape inference for pack

mlir::LogicalResult PackOp::verify() {
  PackOp op = *this;
  // TODO(antiagainst): Implement other checks as in
  // tensorflow/lite/kernels/pack.cc

  if (op.getOperation()->getNumOperands() != op.getValuesCount())
    return op.emitOpError("input count should match 'values_count' attribute");

  Value operand0 = op.getOperand(0);
  auto input_type = operand0.getType().cast<ShapedType>();

  // Check axis bounds.
  if (input_type.hasRank()) {
    int32_t axis_value = op.getAxis();
    if (axis_value < 0) axis_value += input_type.getRank() + 1;
    if (axis_value < 0 || axis_value >= input_type.getRank() + 1)
      return op.emitOpError()
             << "op attribute 'axis' should be in range [-rank - 1, rank + 1), "
             << "got rank = " << input_type.getRank()
             << ", and axis = " << op.getAxis();
  }

  // Make sure all inputs have the same shape and element type.
  // TODO(b/135032063): Simplify once fixed.
  for (Type operand_type : op.getOperandTypes()) {
    if (failed(mlir::verifyCompatibleShape(input_type, operand_type)))
      return op.emitOpError("operands should be of the same type. got ")
             << input_type << ", " << operand_type;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

namespace {
// This pattern matches and merges a tfl.reshape under the following
// condition:
// * The input's defining op is another tfl.reshape.
// TODO(antiagainst): This pattern probably should be moved to the peephole
// category, after we have the infra for peephole passes.
struct RemoveAdjacentReshape : public RewritePattern {
  explicit RemoveAdjacentReshape(MLIRContext* context)
      : RewritePattern(ReshapeOp::getOperationName(), 1, context) {}

  LogicalResult match(Operation* op) const override {
    auto thisOp = cast<ReshapeOp>(op);
    auto prevOp = thisOp.getOperand(0).getDefiningOp();
    return isa_and_nonnull<ReshapeOp>(prevOp) ? success() : failure();
  }

  void rewrite(Operation* op, PatternRewriter& rewriter) const override {
    auto thisOp = cast<ReshapeOp>(op);
    auto prevOp = cast<ReshapeOp>(thisOp.getOperand(0).getDefiningOp());

    // Replace
    //   %1 = "tfl.reshape"(%0, %shape0)
    //   %2 = "tfl.reshape"(%1, %shape1)
    // With
    //   %2 = "tfl.reshape"(%0, %shape1)
    rewriter.replaceOpWithNewOp<ReshapeOp>(
        op, thisOp.getType(), prevOp.getOperand(0), thisOp.getOperand(1));
  }
};

// The kernel expects an 1-D tensor for the shape operand if it presents. If all
// the dimensions are '1's except the last dimension, it will be reshaped to a
// 1-D tensor.
// Note that this pattern doesn't check or change the content of the shape
// tensor.
struct ConvertShapeTo1D : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshape,
                                PatternRewriter& rewriter) const override {
    if (!reshape.getShape().hasOneUse()) return failure();

    DenseIntElementsAttr shape;
    if (!matchPattern(reshape.getShape(), m_Constant(&shape))) {
      return failure();
    }
    // It is already a 1-D constant, no change.
    auto old_shape = shape.getShapedType().getShape();
    if (old_shape.size() == 1) {
      return failure();
    }
    // Verify all the leading dimensions are length one, except the last one.
    for (auto it = ++old_shape.rbegin(); it != old_shape.rend(); ++it) {
      if (*it != 1) {
        reshape->emitError(
            "Non-vector shape input is used, might cause runtime error");
        return failure();
      }
    }
    auto new_shape = shape.reshape(GetTypeFromTFTensorShape(
        {*old_shape.rbegin()}, shape.getShapedType().getElementType()));
    rewriter.replaceOpWithNewOp<TFL::ConstOp>(
        reshape.getShape().getDefiningOp(), new_shape);
    return success();
  }
};

bool InputOutputHasSameShape(mlir::Type input_type, mlir::Type output_type) {
  auto input_shaped_type = input_type.dyn_cast_or_null<ShapedType>();
  if (!input_shaped_type || !input_shaped_type.hasStaticShape()) return false;

  auto output_shaped_type = output_type.dyn_cast_or_null<ShapedType>();
  if (!output_shaped_type || !output_shaped_type.hasStaticShape()) return false;

  return input_shaped_type == output_shaped_type;
}

}  // end anonymous namespace

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // Remove identity reshape with both static result and input shape.
  auto result_type = getType().cast<ShapedType>();
  auto input_type = getOperand(0).getType().cast<ShapedType>();
  if (InputOutputHasSameShape(input_type, result_type)) return getInput();

  // Constant folding
  if (auto dense_elements = operands[0].dyn_cast_or_null<DenseElementsAttr>()) {
    // If the result type isn't static, tries to derive the result type from
    // the #2 operand.
    if (!result_type.hasStaticShape()) {
      auto shape_elements = operands[1].dyn_cast_or_null<DenseElementsAttr>();
      if (!shape_elements) return nullptr;

      SmallVector<int64_t, 4> shape_data;
      for (const auto& it : shape_elements.getValues<APInt>()) {
        shape_data.push_back(it.getSExtValue());
      }
      result_type = GetTypeFromTFTensorShape(
          shape_data, input_type.getElementType());
    }
    return dense_elements.reshape(result_type);
  }

  return nullptr;
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<RemoveAdjacentReshape, ConvertShapeTo1D>(context);
}

using ReshapeErrorHandler =
    llvm::function_ref<LogicalResult(const llvm::Twine&)>;

LogicalResult GetReshapeOutputType(Value input, Value shape,
                                   ReshapeErrorHandler error_handler,
                                   TensorType& output_ty) {
  auto input_ty = input.getType().cast<TensorType>();
  auto element_ty = input_ty.getElementType();
  output_ty = UnrankedTensorType::get(element_ty);

  auto shape_ty = shape.getType().dyn_cast<RankedTensorType>();
  if (!shape_ty) return success();
  if (shape_ty.getRank() != 1)
    return error_handler(llvm::formatv(
        "requires 'shape' to be rank 1, but got {0}", shape_ty.getRank()));

  DenseIntElementsAttr shape_attr;
  if (!matchPattern(shape, m_Constant(&shape_attr))) {
    // If only shape of `shape` is known, return ranked but dynamic output
    // shape.
    if (shape_ty.hasStaticShape()) {
      llvm::SmallVector<int64_t, 8> dynamic_shape(shape_ty.getDimSize(0),
                                                  ShapedType::kDynamic);
      output_ty =
          GetTypeFromTFTensorShape(dynamic_shape, element_ty);
    }
    return success();
  }

  // Detect if reshape output shape is folded.
  bool shape_ty_zero_dim = false;
  int unknown_index = -1;
  // The product of constant shape argument excluding unknown dimension.
  int64_t shape_ty_size = 1;
  llvm::SmallVector<int64_t, 8> output_ty_shape;
  output_ty_shape.reserve(shape_attr.getNumElements());
  for (const auto& dim : llvm::enumerate(shape_attr.getValues<APInt>())) {
    const int64_t size = dim.value().getSExtValue();
    if (size == kTFDynamicSize ||  // NOLINT
        size == ShapedType::kDynamic) {        // NOLINT
      if (unknown_index != -1)
        return error_handler(llvm::formatv(
            "requires 'shape' to have at most one dynamic dimension, but got "
            "multiple dynamic dimensions at indices {0} and {1}. You need to "
            "set up the unspecified size(s) to avoid this problem, for example,"
            "setting batch size in keras model or setting unspecified input "
            "size(s) with fixed ones.",
            unknown_index, dim.index()));

      unknown_index = dim.index();
    } else if (size == 0) {
      shape_ty_zero_dim = true;
    } else if (size > 0) {
      shape_ty_size *= size;
    } else {
      return error_handler(
          llvm::formatv("requires 'shape' to have dimensions greater than -1, "
                        "but got {0} at index {1}",
                        size, dim.index()));
    }
    output_ty_shape.push_back(size);
  }

  if (!input_ty.hasStaticShape()) {
    output_ty =
        GetTypeFromTFTensorShape(output_ty_shape, element_ty);
    return success();
  }

  // Compute the value of the unknown dimension.
  if (unknown_index != -1) {
    // Compute number of elements in tensor shape.
    int64_t input_ty_size = 1;
    bool input_ty_zero_dim = false;
    for (const auto& dim : input_ty.getShape()) {
      if (dim > 0 || !shape_ty_zero_dim) {
        input_ty_size *= dim;
      } else {
        input_ty_zero_dim = true;
      }
    }

    const int64_t missing_dim = input_ty_size / shape_ty_size;
    if (!input_ty_zero_dim && shape_ty_size * missing_dim != input_ty_size)
      return error_handler(
          llvm::formatv("requires 'input' number of elements be a multiple of "
                        "{0}, but got {1}",
                        shape_ty_size, input_ty_size));

    // Set the unknown dimension such that total number of elements remain
    // constant.
    output_ty_shape[unknown_index] = missing_dim;
  }

  output_ty = GetTypeFromTFTensorShape(output_ty_shape, element_ty);

  return success();
}

mlir::LogicalResult ReshapeOp::verify() {
  ReshapeOp op = *this;
  auto error_handler = [&op](const llvm::Twine& message) -> LogicalResult {
    return op.emitOpError() << message;
  };
  TensorType expected_ty;
  if (failed(GetReshapeOutputType(op.getInput(), op.getShape(), error_handler,
                                  expected_ty)))
    return failure();

  auto output_ty = op.getType().dyn_cast<RankedTensorType>();
  if (!output_ty) return success();
  auto input_ty = op.getInput().getType().cast<TensorType>();
  if (output_ty.hasStaticShape() && input_ty.hasStaticShape()) {
    const int64_t output_ty_size = output_ty.getNumElements();
    const int64_t input_ty_size = input_ty.getNumElements();
    if (input_ty_size != output_ty_size)
      return op.emitOpError() << "requires 'output' number of elements to "
                                 "match 'input' number of elements, but got "
                              << output_ty_size << " and " << input_ty_size;
  }

  if (!TF::AreCastCompatible({output_ty, expected_ty}))
    return op.emitOpError()
           << "requires 'output' type " << output_ty
           << " to be cast compatible with expected type " << expected_ty;

  return success();
}

LogicalResult ReshapeOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attr, OpaqueProperties properties, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ReshapeOpAdaptor op(operands, attr, properties);
  const Value input = op.getInput();
  const Value shape = op.getShape();

  auto error_handler = [&](const llvm::Twine& message) -> LogicalResult {
    // A dummy error handler.
    // Errors when computing the output shape will be raised in
    // ReshapeOp::verify call.
    return failure();
  };
  TensorType output_type;
  if (GetReshapeOutputType(input, shape, error_handler, output_type)
          .succeeded()) {
    inferredReturnTypes.assign({output_type});
    return success();
  }
  Type result_type;
  result_type = UnrankedTensorType::get(
      input.getType().cast<ShapedType>().getElementType());
  inferredReturnTypes.assign({result_type});
  return success();
}

bool ReshapeOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size() || lhs.size() != 1) return false;
  if (failed(mlir::verifyCompatibleShape(lhs[0], rhs[0]))) return false;
  return true;
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // TODO(b/142478136): Handle fused ops.
  if (getFusedActivationFunction() != "NONE") return {};
  return ConstFoldBinaryOp(
      getType(), operands, [](APFloat a, APFloat b) { return a - b; },
      [](APInt a, APInt b) { return a - b; });
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

static void BuildTopKOp(OpBuilder *builder, OperationState &result, Value input,
                        Value k) {
  // Output size is only known if k is constant value. A negative dimension is
  // considered dynamic so use -1 here if k is not a constant value.
  int const_k = -1;
  ElementsAttr cst;
  if (matchPattern(k, m_Constant(&cst)))
    // These casts should all be valid due to how Tensor constants are stored.
    // TODO(jpienaar): This should use a helper function.
    const_k = cst.getValues<IntegerAttr>()[0].getValue().getSExtValue();

  auto val_type = input.getType().cast<TensorType>();
  // If value is unranked, then so is results.
  if (!val_type.hasRank())
    return TFL::TopKV2Op::build(
        *builder, result, UnrankedTensorType::get(val_type.getElementType()),
        UnrankedTensorType::get(k.getType()), input, k);

  // Resultant shape is value.shape[:-1] + [k]
  std::vector<int64_t> shape(val_type.getShape());
  shape[shape.size() - 1] = const_k;
  TFL::TopKV2Op::build(
      *builder, result,
      GetTypeFromTFTensorShape(shape, val_type.getElementType()),
      GetTypeFromTFTensorShape(shape, k.getType()), input, k);
}

//===----------------------------------------------------------------------===//
// FakeQuantOp
//===----------------------------------------------------------------------===//

// Return true if the op has non-empty "minmax" attribute.
static inline bool HasValidMinMaxAttribute(Operation *op) {
  auto minmax = op->getAttrOfType<ArrayAttr>("minmax");
  return minmax && minmax.getValue().size() == 2;
}

namespace {

/// This pattern matches and remove a tfl.fake_quant if all the users of this op
/// and itself have "minmax" attribute set.
struct DropFakeQuant : public RewritePattern {
  explicit DropFakeQuant(MLIRContext* context)
      : RewritePattern(FakeQuantOp::getOperationName(), 1, context) {}

  LogicalResult match(Operation* op) const override {
    // We only match the op with valid "minmax" attribute.
    if (!HasValidMinMaxAttribute(op)) return failure();

    // If all the users of this op have valid "minmax" attributes, it is matched
    // and can be removed.
    auto fakeQuantOp = cast<FakeQuantOp>(op);
    for (auto* operand : fakeQuantOp.getResult().getUsers())
      if (!HasValidMinMaxAttribute(operand)) return failure();

    return success();
  }

  void rewrite(Operation* op, PatternRewriter& rewriter) const override {
    // Replace the matched FakeQuantOp by its primary operand.
    rewriter.replaceOp(op, op->getOperand(0));
  }
};
}  // end anonymous namespace

void FakeQuantOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<DropFakeQuant>(context);
}

//===----------------------------------------------------------------------===//
// UnpackOp
//===----------------------------------------------------------------------===//

LogicalResult UnpackOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  UnpackOpAdaptor op(operands, attributes, properties);
  // TODO(jpienaar): Refactor verify
  if (failed(op.verify(loc.has_value() ? *loc : UnknownLoc::get(context))))
    return failure();

  if (operands.size() != 1) {
    return emitOptionalError(loc, "input count should be equal to 1");
  }

  const int64_t num_value = op.getNumAttr().getInt();
  auto input_type = operands[0].getType().dyn_cast<ShapedType>();
  if (!input_type || !input_type.hasRank()) {
    // If input is unranked, then so is output.
    inferredReturnTypes.assign(
        num_value, UnrankedTensorType::get(input_type.getElementType()));
    return success();
  }

  if (input_type.hasStaticShape() && input_type.getNumElements() <= 0) {
    return emitOptionalError(
        loc, "number of elements in input should be larger than 0");
  }

  const int64_t rank = input_type.getRank();
  if (rank <= 0) {
    return emitOptionalError(loc, "input should be of rank larger than 0");
  }

  int64_t axis_value = op.getAxisAttr().getInt();
  if (axis_value < 0) {
    axis_value += rank;
  }
  if (axis_value < 0 || axis_value >= rank) {
    return emitOptionalError(
        loc, "attribute 'axis' should be in range [-rank, rank), got axis = ",
        op.getAxisAttr().getInt(), ", and rank = ", rank);
  }

  if (!ShapedType::isDynamic(input_type.getDimSize(axis_value)) &&
      input_type.getDimSize(axis_value) != num_value) {
    return emitOptionalError(loc, "output count should match 'num' attribute");
  }

  auto output_shape = llvm::to_vector<4>(input_type.getShape());
  output_shape.erase(output_shape.begin() + axis_value);

  auto output_type = GetTypeFromTFTensorShape(
      output_shape, input_type.getElementType());
  inferredReturnTypes.assign(num_value, output_type);

  return success();
}

bool UnpackOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (auto pair : llvm::zip(lhs, rhs)) {
    if (failed(
            mlir::verifyCompatibleShape(std::get<0>(pair), std::get<1>(pair))))
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

// TODO(b/133854225): Implement shape inference to Mean

//===----------------------------------------------------------------------===//
// LSTMOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(LSTMOp op) {
  auto operands = op.GetStatefulOperands();
  if (operands.size() == 2 && operands[0] == 18 && operands[1] == 19) {
    return success();
  }
  return op.emitError("LSTMOp expected to have two stateful operands");
}

//===----------------------------------------------------------------------===//
// UnidirectionalSequenceLSTMOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(UnidirectionalSequenceLSTMOp op) {
  auto operands = op.GetStatefulOperands();
  if (operands.size() == 2 && operands[0] == 18 && operands[1] == 19) {
    return success();
  }
  return op.emitError(
      "UnidirectionalSequenceLSTMOp expected to have two stateful operands");
}

//===----------------------------------------------------------------------===//
// UnidirectionalSequenceRNNOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(UnidirectionalSequenceRNNOp op) {
  auto operands = op.GetStatefulOperands();
  if (operands.size() == 1 && operands[0] == 4) {
    return success();
  }
  return op.emitError(
      "UnidirectionalSequenceRNNOp expected to have one stateful operand");
}

//===----------------------------------------------------------------------===//
// SvdfOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(SVDFOp op) {
  auto operands = op.GetStatefulOperands();
  if (operands.size() == 1 && operands[0] == 4) {
    return success();
  }
  return op.emitError("SvdfOp expected to have one stateful operand");
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

OpFoldResult AbsOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat { return llvm::abs(value); };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//

OpFoldResult SinOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::sin(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//

OpFoldResult CosOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::cos(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

OpFoldResult LogOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::log(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

OpFoldResult SqrtOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::sqrt(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//

OpFoldResult RsqrtOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = 1.f / std::sqrt(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// SquareOp
//===----------------------------------------------------------------------===//

OpFoldResult SquareOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat { return value * value; };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

OpFoldResult RankOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  assert(operands.size() == 1);
  auto result_type = getType().cast<ShapedType>();
  if (auto elements_attr = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    auto rank = static_cast<int32_t>(elements_attr.getShapedType().getRank());
    return DenseElementsAttr::get(result_type, {rank});
  }

  // Also fold if `input` has a known rank.
  auto input_type = getInput().getType().cast<ShapedType>();
  // Do not fold if rank is zero because the TFLite converter doesn't
  // distinguish between unranked input and scalar input due to b/138865275.
  // TODO(b/138865275): Remove `input_type.getRank() != 0` in the following
  // predicate and fold the op when rank is zero.
  if (input_type.hasRank() && input_type.getRank() != 0) {
    auto rank = static_cast<int32_t>(input_type.getRank());
    return DenseElementsAttr::get(result_type, {rank});
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return getValue();
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

namespace {

// Compute the length of a range (1-D) tensor given `start`, `limit`, `delta`.
// Template parameter `FloatOrInt` must be standard C integer or floating-point
// types.
template <typename FloatOrInt>
int GetLengthOfRange(FloatOrInt start, FloatOrInt limit, FloatOrInt delta) {
  // Refer to the implementation in
  // tensorflow/lite/kernels/range.cc.
  return std::is_integral<FloatOrInt>::value
             ? ((std::abs(limit - start) + std::abs(delta) - 1) /
                std::abs(delta))
             : std::ceil(std::abs((limit - start) / delta));
}

// Builds a constant range tensor of `result_elem_type` elements.
// Template parameter `FloatOrIntAtrr` must be mlir::IntegerAttr or
// mlir::FloatAttr.
template <typename FloatOrIntAtrr>
DenseElementsAttr BuildConstRangeTensor(Type result_elem_type, int num_elements,
                                        FloatOrIntAtrr start_attr,
                                        FloatOrIntAtrr delta_attr) {
  using ValueType = typename FloatOrIntAtrr::ValueType;  // APInt or APFloat
  ValueType start = start_attr.getValue();
  ValueType delta = delta_attr.getValue();

  SmallVector<ValueType, 16> new_values;
  new_values.reserve(num_elements);
  ValueType new_value = start;
  for (int i = 0; i < num_elements; ++i) {
    new_values.push_back(new_value);
    new_value = new_value + delta;
  }
  // Result is always a 1-D tensor.
  auto new_result_type =
      RankedTensorType::get({num_elements}, result_elem_type);
  return DenseElementsAttr::get(new_result_type, new_values);
}
}  // namespace

OpFoldResult RangeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  assert(operands.size() == 3);
  auto start_tensor = operands[0].dyn_cast_or_null<ElementsAttr>();
  auto limit_tensor = operands[1].dyn_cast_or_null<ElementsAttr>();
  auto delta_tensor = operands[2].dyn_cast_or_null<ElementsAttr>();
  if (start_tensor && limit_tensor && delta_tensor) {
    // Operands should all be scalars
    assert(start_tensor.getShapedType().getRank() == 0 &&
           limit_tensor.getShapedType().getRank() == 0 &&
           delta_tensor.getShapedType().getRank() == 0);
    Type elem_type = getType().cast<ShapedType>().getElementType();
    if (elem_type.isSignlessInteger()) {
      auto start_attr = start_tensor.getValues<IntegerAttr>()[0];
      auto limit_attr = limit_tensor.getValues<IntegerAttr>()[0];
      auto delta_attr = delta_tensor.getValues<IntegerAttr>()[0];
      const int num_elements = GetLengthOfRange(
          start_attr.getInt(), limit_attr.getInt(), delta_attr.getInt());
      return BuildConstRangeTensor(elem_type, num_elements, start_attr,
                                   delta_attr);
    } else if (elem_type.isa<FloatType>()) {
      auto start_attr = start_tensor.getValues<FloatAttr>()[0];
      auto limit_attr = limit_tensor.getValues<FloatAttr>()[0];
      auto delta_attr = delta_tensor.getValues<FloatAttr>()[0];
      const int num_elements = GetLengthOfRange(start_attr.getValueAsDouble(),
                                                limit_attr.getValueAsDouble(),
                                                delta_attr.getValueAsDouble());
      return BuildConstRangeTensor(elem_type, num_elements, start_attr,
                                   delta_attr);
    }
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

namespace {

// Computes the permutation of a constant `input_tensor` according to `perm`.
// The function recursively traverses the dimensions of the output tensor in
// a row-major order and writes the value in the output tensor into
// `new_values`.
void ComputePermutation(ElementsAttr input_tensor, ArrayRef<int32_t> perm,
                        ArrayRef<int64_t> output_shape, int num_dimensions,
                        int output_axis, std::vector<uint64_t> *input_indices,
                        std::vector<Attribute> *new_values) {
  // Refer to the implementation of `Transpose` function in
  // tensorflow/lite/kernels/internal/reference/reference_ops.h
  assert(output_axis < num_dimensions);
  const int input_axis = perm[output_axis];
  for (int i = 0; i < output_shape[output_axis]; ++i) {
    // Update the input indices on `input_axis`.
    input_indices->at(input_axis) = i;
    // Write the value from `input_tensor` if it is the last axis or
    // recurse into the next axis.
    const bool is_last_axis = output_axis == num_dimensions - 1;
    if (is_last_axis) {
      auto iter = input_tensor.getValues<IntegerAttr>();
      new_values->push_back(iter[*input_indices]);
    } else {
      ComputePermutation(input_tensor, perm, output_shape, num_dimensions,
                         output_axis + 1, input_indices, new_values);
    }
  }
}

}  // namespace

OpFoldResult TransposeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  assert(operands.size() == 2);
  auto input_tensor = operands[0].dyn_cast_or_null<ElementsAttr>();
  auto perm_tensor = operands[1].dyn_cast_or_null<ElementsAttr>();
  if (!input_tensor || !perm_tensor) return nullptr;

  // Do not try to fold elements attr of a quant type because
  // DenseElementsAttr does not support it.
  if (!getType().cast<ShapedType>().getElementType().isIntOrFloat())
    return nullptr;

  assert(perm_tensor.getShapedType().getRank() == 1);
  const int num_dimensions = input_tensor.getShapedType().getRank();
  assert(perm_tensor.getShapedType().getNumElements() == num_dimensions);

  ArrayRef<int64_t> input_shape = input_tensor.getShapedType().getShape();
  auto output_type = getType().cast<ShapedType>();

  SmallVector<int32_t, 4> perm;
  SmallVector<int64_t, 4> output_shape;
  auto attr_iter = perm_tensor.getValues<IntegerAttr>();
  for (int i = 0; i < num_dimensions; ++i) {
    perm.push_back(attr_iter[i].getInt());
    output_shape.push_back(input_shape[perm[i]]);
  }

  std::vector<Attribute> new_values;
  new_values.reserve(input_tensor.getShapedType().getNumElements());
  std::vector<uint64_t> input_indices(num_dimensions);
  ComputePermutation(input_tensor, perm, output_shape, num_dimensions,
                     /*output_axis=*/0, &input_indices, &new_values);
  auto result_type =
      RankedTensorType::get(output_shape, output_type.getElementType());
  return DenseElementsAttr::get(result_type, new_values);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

using namespace mlir::math;

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_interface.cc.inc"

Operation *TensorFlowLiteDialect::materializeConstant(OpBuilder &builder,
                                                      Attribute value,
                                                      Type type, Location loc) {
  // If this is a constant bytes attribute or the result type doesn't match the
  // attribute type, then generate a tfl.pseudo_const.
  if (value.isa<ConstBytesAttr>() ||
      (value.isa<ElementsAttr>() &&
       value.cast<ElementsAttr>().getType() != type))
    return builder.create<ConstOp>(loc, type, value.cast<ElementsAttr>());
  if (arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(value));
  return nullptr;
}

}  // namespace TFL
}  // namespace mlir


#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.cc.inc"
