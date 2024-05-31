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

// This is the base operation definition file for TensorFlow.
//
// This file includes the definition for the TensorFlow dialect, base TensorFlow
// op, and various commonly used TensorFlow traits, types, attributes, and
// builders.

#ifdef TF_OP_BASE
#else
#define TF_OP_BASE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/IR/AttrTypeBase.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.td"

//===----------------------------------------------------------------------===//
// TensorFlow dialect definitions
//===----------------------------------------------------------------------===//

def TF_Dialect : Dialect {
  let name = "tf";

  let description = [{
The TensorFlow dialect.

This dialect maps to TensorFlow operations.

Invariants:

* All values are of Tensor type (in particular, scalars are
  represented using zero-dimentional tensors);

TODO: Make invariants more structured so that we can reference them in ops.
  }];

  let cppNamespace = "::mlir::TF";
}

def TFTypeDialect : Dialect {
  let name = "tf_type";

  let summary = "This dialect defines TensorFlow Types.";
  let description = [{
    This dialect is only provide types and attributes used by various dialects
    in TensorFlow, but does not include any operation.
    It is separated from the other dialect for modularity and binary-size
    control.
  }];

  let cppNamespace = "::mlir::tf_type";
  let extraClassDeclaration = [{
     ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const;
     void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const;
  }];
  let useDefaultAttributePrinterParser = 1;
}


// All of the types will extend this class.
class TFType_Type<string name> : TypeDef<TFTypeDialect, name> { }

def ControlType : TFType_Type<"Control"> {
  let mnemonic = "control";
  let description = [{
    This type models a control dependency between two TensorFlow nodes.
  }];
}

def OpaqueTensorType : TFType_Type<"OpaqueTensor"> {
  let mnemonic = "tensor";
  let description = [{
    This type models an opaque tensor: the shape and dtypes are unknown. This is
    particularly useful in generic functions.
  }];
}

class TFGraph_OpaqueTensorOr<Type ConcreteType>
    : Type<Or<[ConcreteType.predicate, OpaqueTensorType.predicate]>,
           ConcreteType.description # " or opaque tensor type">;

def TFGraph_Tensor : TFGraph_OpaqueTensorOr<AnyTensor>;

// A value that is either a tensor or a control type.
def TFGraph_TensorOrControlType
    : Type<Or<[TFGraph_Tensor.predicate, ControlType.predicate]>,
           "any tensor or control type">;

//===----------------------------------------------------------------------===//
// TensorFlow traits
//===----------------------------------------------------------------------===//


// Specify this trait if the op requires all outputs to have the same type and
// the inputs either have the same type as result or a ref type corresponding to
// the result type.
def TF_OperandsSameAsResultsTypeOrRef : NativeOpTrait<
  "TF::OperandsSameAsResultsTypeOrRef">;

// Op has the same operand and result element types (or type itself, if scalar)
// after resolving reference types (i.e., after converting reference types to
// their corresponding TensorFlow or standard types).
def TF_SameOperandsAndResultElementTypeResolveRef : NativeOpTrait<
  "TF::SameOperandsAndResultElementTypeResolveRef">;

// Op has the same operand and result types after resolving reference types
// (i.e., after converting reference types to their corresponding TensorFlow or
// standard types). Also, this allows compatible types so it is legal to have
// tensor<*xf32> and tensor<4xf32> types.
def TF_SameOperandsAndResultTypeResolveRef : TraitList<
  InferTensorType.traits #
  [
    NativeOpTrait<"TF::SameOperandsAndResultTypeResolveRef">
  ]>;

// Layout agnostic operations do not depend on the operands data layout (data
// format), as an example all element wise operations are layout agnostic.
def TF_LayoutAgnostic : NativeOpTrait<"TF::LayoutAgnostic">;

// Trait to indicate operations that cannot be duplicated as they might carry
// certain state around within their implementations.
def TF_CannotDuplicate : NativeOpTrait<"TF::CannotDuplicate">;

// Trait to indicate an operation cannot be constant folded.
def TF_NoConstantFold : NativeOpTrait<"TF::NoConstantFold">;

// Coefficient wise binary operation with implicit broadcasting support, for
// example tf.Sub operation.
def TF_CwiseBinary : NativeOpTrait<"TF::CwiseBinary">;

// Coefficient wise unary operation, for example tf.Sqrt operation.
def TF_CwiseUnary : NativeOpTrait<"TF::CwiseUnary">;

// op op X == op X (unary) / X op X == X (binary)
// This version requires TF_SameOperandsAndResultTypeResolveRef trait unlike
// the core implementation requiring SameOperandsAndResultType.
//
// This shouldn't be used for side effecting ops.
def TF_Idempotent : NativeOpTrait<"TF::IsIdempotent">;

// op op X == X
// This version requires TF_SameOperandsAndResultTypeResolveRef trait unlike
// the core implementation requiring SameOperandsAndResultType.
//
// This shouldn't be used for side effecting ops.
def TF_Involution : NativeOpTrait<"TF::IsInvolution">;

// Variant of broadcastable trait that considers TF's subtype behavior.
class TF_OpIsBroadcastableToRes<int opId, int resId> : And<[
    TCOpResIsShapedTypePred<opId, resId>,
    CPred<"mlir::tf_type::BroadcastCompatible("
              "$_op.getOperand(" # opId # ").getType(), "
              "$_op.getResult(" # resId # ").getType())">]>;


class TF_AllTypesMatchPred<list<string> values> :
    CPred<"tf_type::AreCastCompatible(llvm::ArrayRef({" #
      !interleave(values, ", ") # "}))">;

class TF_AllTypesMatch<list<string> names> :
    PredOpTrait<
        "all of {" # !interleave(names, ", ") #
          "} have dynamically equal types ",
        TF_AllTypesMatchPred<
            !foreach(n, names, !subst("$_self", "$" # n, "$_self.getType()"))>>;

// This trait indicates that all returned resources are unique for a
// resource-allocating op (i.e. op with `MemAlloc` side effect).
//
// Note that if the trait is used where this invariant is not true, then this
// might lead to incorrect execution order, while if not used where it should
// be, it can only lead to reduced performance due to conservative ordering.
// Example op where the invariant is not true: `TF_VarHandleOp`.
def TF_UniqueResourceAllocation: TraitList<[
    TF_ResourceHandleAllocatorInterface,
    NativeOpTrait<"TF::UniqueResourceAllocation">
]>;

//===----------------------------------------------------------------------===//
// TensorFlow op definitions
//===----------------------------------------------------------------------===//

class TF_Op<string mnemonic, list<Trait> traits = []> :
    Op<TF_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// TensorFlow type definitions
//===----------------------------------------------------------------------===//

// Any tensor element type defined in the TensorFlow dialect
def TF_TFDialectType :
    Type<CPred<"$_self.isa<TensorFlowType>()">, "TensorFlow type">;

// Any tensor element type allowed in TensorFlow ops
def TF_ElementType : Type<Or<[AnyFloat.predicate, AnyInteger.predicate,
                                 TF_TFDialectType.predicate]>,
                          "tf.dtype">;

// Any TensorFlow tensor type
def TF_Tensor : TensorOf<[TF_ElementType]>;

//===----------------------------------------------------------------------===//
// Integer types

def TF_Bool : AnyTypeOf<[I<1>], "bool">;

def TF_I32Or64 : SignlessIntOfWidths<[32, 64]>;

def TF_I32OrI64Tensor : TensorOf<[TF_I32Or64]>;

def TF_Uint8 : Type<CPred<"$_self.isa<mlir::TF::Uint8Type>()">,
                    "TensorFlow uint8 type">,
               BuildableType<"getType<mlir::TF::Uint8Type>()">;

def TF_Uint16 : Type<CPred<"$_self.isa<mlir::TF::Uint16Type>()">,
                     "TensorFlow uint16 type">,
               BuildableType<"getType<mlir::TF::Uint16Type>()">;

def TF_Uint32 : Type<CPred<"$_self.isa<mlir::TF::Uint32Type>()">,
                     "TensorFlow uint32 type">,
               BuildableType<"getType<mlir::TF::Uint32Type>()">;

def TF_Uint64 : Type<CPred<"$_self.isa<mlir::TF::Uint64Type>()">,
                     "TensorFlow uint64 type">,
                BuildableType<"getType<mlir::TF::Uint64Type>()">;

def TF_Int16 : SignedIntOfWidths<[16]>;
def TF_Int32 : SignedIntOfWidths<[32]>;
def TF_Int64 : SignedIntOfWidths<[64]>;

// Any unsigned integer type
def TF_UInt : AnyTypeOf<[TF_Uint8, TF_Uint16, TF_Uint32, TF_Uint64]>;

// Any signed integer type
def TF_SInt : SignedIntOfWidths<[8, 16, 32, 64]>;

// Any integer type
def TF_Int : AnyTypeOf<[TF_SInt, TF_UInt]>;

// Any integer tensor types
def TF_Uint8Tensor: TensorOf<[TF_Uint8]>;
def TF_IntTensor : TensorOf<[TF_Int]>;
def TF_I32Tensor : TensorOf<[TF_Int32]>;
def TF_Int64Tensor : TensorOf<[TF_Int64]>;

//===----------------------------------------------------------------------===//
// Floating-point types

//===----------------------------------------------------------------------===//
// Floating-point types (including corresponding reference types)

def TF_Bfloat16 : AnyTypeOf<[BF16], "bfloat16">;
def TF_Float16 : AnyTypeOf<[F16], "16-bit float">;
def TF_Float32 : AnyTypeOf<[F32], "32-bit float">;
def TF_Float64 : AnyTypeOf<[F64], "64-bit float">;

def TF_F32Or64 : FloatOfWidths<[32, 64]>;

def TF_F32OrF64Tensor : TensorOf<[TF_F32Or64]>;

// Any floating-point tensor types
def TF_FpTensor : TensorOf<[AnyFloat]>;
def TF_Bfloat16Tensor : TensorOf<[TF_Bfloat16]>;
def TF_FloatTensor : TensorOf<[AnyFloat]>;
def TF_Float32Tensor : TensorOf<[TF_Float32]>;

def TF_BoolTensor : TensorOf<[TF_Bool]>;

//===----------------------------------------------------------------------===//
// Complex types

def TF_Complex64 :
    Type<CPred<"$_self.isa<TF::Complex64Type>()">, "complex64 type">;
def TF_Complex64Tensor : TensorOf<[TF_Complex64]>;

def TF_Complex128 :
    Type<CPred<"$_self.isa<TF::Complex128Type>()">, "complex128 type">;
def TF_Complex128Tensor : TensorOf<[TF_Complex128]>;

def TF_AnyComplex : AnyTypeOf<[TF_Complex64, TF_Complex128],
                              "64/128-bit complex type">;

def TF_ComplexTensor : TensorOf<[TF_AnyComplex]>;

//===----------------------------------------------------------------------===//
// String/variant/resource types

def TF_Str : Type<CPred<"$_self.isa<mlir::TF::StringType>()">,
                  "TensorFlow string type">,
             BuildableType<"getType<mlir::TF::StringType>()">;
def TF_StrTensor : TensorOf<[TF_Str]>;

def TF_Variant : Type<CPred<"$_self.isa<mlir::TF::VariantType>()">,
                      "TensorFlow variant type">,
                 BuildableType<"getType<mlir::TF::VariantType>()">;
def TF_VariantTensor : TensorOf<[TF_Variant]>;

def TF_Resource : Type<CPred<"$_self.isa<mlir::TF::ResourceType>()">,
                       "TensorFlow variant type">,
                  BuildableType<"getType<mlir::TF::ResourceType>()">;
def TF_ResourceTensor : TensorOf<[TF_Resource]>;

//===----------------------------------------------------------------------===//
// Multi-category type constraints

def TF_IntOrF32OrF64Tensor: TensorOf<[TF_Int, TF_F32Or64]>;

def TF_FpOrI32OrI64Tensor : TensorOf<[AnyFloat, TF_I32Or64]>;

// Any integer or floating-point tensor types
def TF_IntOrFpTensor : TensorOf<[TF_Int, AnyFloat]>;

def TF_FpOrComplexTensor : TensorOf<[AnyFloat, TF_AnyComplex]>;

def TF_AnyNumber : AnyTypeOf<[TF_Int, AnyFloat, TF_AnyComplex], "number">;

def TF_NumberTensor : TensorOf<[TF_AnyNumber]>;

def TF_NumberOrStrTensor : TensorOf<[TF_AnyNumber, TF_Str]>;

//===----------------------------------------------------------------------===//
// TensorFlow attribute definitions
//===----------------------------------------------------------------------===//

class TF_TensorFlowAttr <string name, string description> :
    Attr<CPred<"$_self.isa<mlir::TF::" # name # "Attr>()">,
         "TensorFlow " # description # " attribute">;

def TF_ShapeAttr : TF_TensorFlowAttr<"Shape", "shape"> {
  let returnType = "std::optional<llvm::ArrayRef<int64_t>>";
  let convertFromStorage = "$_self.cast<mlir::TF::ShapeAttr>().getValue()";

  // Create a ranked shape attr by default.
  let constBuilderCall = "mlir::TF::ShapeAttr::get($_builder.getContext(), $0)";
}

def TF_ShapeAttrArray :
    TypedArrayAttrBase<TF_ShapeAttr, "tensorflow shape attribute array">;

//===----------------------------------------------------------------------===//
// String attribute constraints

// A string attribute whose value are one of the values in `cases`.
class TF_AnyStrAttrOf<list<string> cases> : StringBasedAttr<
  CPred<!foldl(
      "$_self.cast<StringAttr>().getValue() == \"" # !head(cases) # "\"",
      !foreach(case, !tail(cases),
               "$_self.cast<StringAttr>().getValue() == \"" # case # "\""),
      prev, cur, prev # " || " # cur)>,
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;

// TODO: Use EnumAttr to define the common attribute cases

def TF_ConvnetDataFormatAttr : StringBasedAttr<
    CPred<"$_self.cast<StringAttr>().getValue() == \"NHWC\" || " #
          "$_self.cast<StringAttr>().getValue() == \"NCHW\"">,
    "'NHWC' or 'NCHW' convnet data format">;

//===----------------------------------------------------------------------===//
// Type attributes

// A derived attribute that returns the size of `idx`-th ODS-declared variadic
// operand.
class TF_DerivedOperandSizeAttr<int idx> : DerivedAttr<
  "size_t",
  "auto range = getODSOperands(" # idx # ");\n"
  "return std::distance(range.begin(), range.end());",
  [{ $_builder.getI64IntegerAttr($_self) }]>;

// A derived attribute that returns the element type of `idx`-th ODS-declared
// operand. If the `idx`-th operand is a variadic operand, then this attribute
// just returns the element type of its first tensor, which is only meaningful
// when the variadic operand has at least one tensor and the tensors all have
// the same element type.
class TF_DerivedOperandTypeAttr<int idx> : DerivedTypeAttr<
  "return mlir::getElementTypeOrSelf(*getODSOperands(" # idx # ").begin());">;

// A derived attribute that returns the element types of the tensors in the
// dynamic value pack that corresponds to the `idx`-th ODS-declared variadic
// operand. This returns a list of element types so it is used for variadic
// operands that can have different element types.
class TF_DerivedOperandTypeListAttr<int idx> : DerivedAttr<
  "mlir::OperandElementTypeRange",
  "auto values = getODSOperands(" # idx # ");\n"
  "return {mlir::OperandElementTypeIterator(values.begin()), "
          "mlir::OperandElementTypeIterator(values.end())};"
>;

// A derived attribute that returns the size of `idx`-th ODS-declared variadic
// result.
class TF_DerivedResultSizeAttr<int idx> : DerivedAttr<
  "size_t",
  "auto range = getODSResults(" # idx # ");\n"
  "return std::distance(range.begin(), range.end());",
  [{ $_builder.getI64IntegerAttr($_self) }]>;

// A derived attribute that returns the element type of `idx`-th ODS-declared
// result. If the `idx`-th result is a variadic result, then this attribute
// just returns the element type of its first tensor, which is only meaningful
// when the variadic result has at least one tensor and the tensors all have
// the same element type.
class TF_DerivedResultTypeAttr<int idx> : DerivedTypeAttr<
  "return mlir::getElementTypeOrSelf(*getODSResults(" # idx # ").begin());">;

// A derived attribute that returns the element types of the tensors in the
// dynamic value pack that corresponds to the `idx`-th ODS-declared variadic
// result. This returns a list of element types so it is used for variadic
// results that can have different element types.
class TF_DerivedResultTypeListAttr<int idx> : DerivedAttr<
  "mlir::ResultElementTypeRange",
  "auto values = getODSResults(" # idx # ");\n"
  "return {mlir::ResultElementTypeIterator(values.begin()), "
          "mlir::ResultElementTypeIterator(values.end())};"
>;

// A derived attribute that returns the shapes of the tensors in the actual
// value pack that corresponds to the `idx`-th ODS-declared variadic result.
// This returns a list of shapes so it is used for variadic results that
// can have different shapes.
class TF_DerivedResultShapeListAttr<int idx> : DerivedAttr<
  "mlir::TF::ResultShapeRange",
  "auto values = getODSResults(" # idx # ");\n"
  "return {mlir::TF::ResultShapeIterator(values.begin()), "
          "mlir::TF::ResultShapeIterator(values.end())};",
  [{
    ArrayAttr::get($_ctxt,
      [&](){
        llvm::SmallVector<Attribute, 4> ret;
        for (auto shape : $_self)
          ret.push_back(mlir::TF::ShapeAttr::get($_ctxt, shape));
        return ret;
      }())
  }]
>;

// A derived attribute that returns the shape of the first result type.
def TF_DerivedResultShapeAttr : DerivedAttr<"ShapedType",
  "return (*getOperation()->result_type_begin()).cast<ShapedType>();",
  [{ mlir::TF::ShapeAttr::get($_ctxt, $_self) }]>;

def TF_IntTypeAttr : TypeAttrBase<"IntegerType", "integer type"> {
  let returnType = "Type";
}

//===----------------------------------------------------------------------===//
// TensorFlow common builders
//===----------------------------------------------------------------------===//

// Mixin class defining a builder for binary ops supporting broadcast
// behavior. The result type has the same element type as both operands.
class WithBroadcastableBinOpBuilder {
  list<OpBuilder> builders = [
    OpBuilder<(ins "Value":$x, "Value":$y),
    [{
  auto resultType =
      OpTrait::util::getBroadcastedType(x.getType(), y.getType());
  if (!resultType) {
    mlir::emitError($_state.location, "non-broadcastable operands");
    resultType = $_builder.getNoneType();
  }
  return build($_builder, $_state, resultType, x, y);
}]>];
}

// Mixin class defining a builder for comparison ops supporting broadcast
// behavior. The result type has bool element type.
class WithBroadcastableCmpOpBuilder {
  list<OpBuilder> builders = [
    OpBuilder<(ins "Value":$x, "Value":$y),
    [{
  Type resultType;
  if (x.getType().isa<UnrankedTensorType>() ||
      y.getType().isa<UnrankedTensorType>()) {
    resultType = UnrankedTensorType::get($_builder.getI1Type());
  } else {
    SmallVector<int64_t, 4> resultShape;
    if (!OpTrait::util::getBroadcastedShape(
            x.getType().cast<ShapedType>().getShape(),
            y.getType().cast<ShapedType>().getShape(), resultShape)) {
      mlir::emitError($_state.location,
                      "operands have no broadcastable shapes");
    }

    resultType = RankedTensorType::get(resultShape, $_builder.getI1Type());
  }
  return build($_builder, $_state, resultType, x, y);
}]>];
}


class TF_ResourceBase<string resourceKind> :
  Resource<!strconcat("::mlir::TF::ResourceEffects::", resourceKind)> {
}

// Resource types
def TF_VariableResource : TF_ResourceBase<"Variable">;
def TF_StackResource : TF_ResourceBase<"Stack">;
def TF_TensorArrayResource : TF_ResourceBase<"TensorArray">;
def TF_SummaryResource : TF_ResourceBase<"Summary">;
def TF_LookupTableResource : TF_ResourceBase<"LookupTable">;
def TF_DatasetSeedGeneratorResource : TF_ResourceBase<"DatasetSeedGenerator">;
def TF_DatasetMemoryCacheResource : TF_ResourceBase<"DatasetMemoryCache">;
def TF_DatasetIteratorResource : TF_ResourceBase<"DatasetIterator">;
def TF_TPUEmbeddingResource : TF_ResourceBase<"TPUEmbedding">;
def TF_GeneratorOpResource : TF_ResourceBase<"GeneratorOp">;
def TF_SendResource : TF_ResourceBase<"Send">;
def TF_RecvResource : TF_ResourceBase<"Recv">;
def TF_TPUExecuteResource : TF_ResourceBase<"TPUExecute">;
def TF_RandomGeneratorResource : TF_ResourceBase<"RandomGenerator">;
def TF_XlaHostComputeResource : TF_ResourceBase<"XlaHostCompute">;
def TF_WriteTrainingPredictionsResource : TF_ResourceBase<"WriteTrainingPredictions">;
def TF_CollectiveReduceOrderingResource : TF_ResourceBase<"CollectiveReduceOrdering">;
def TF_NcclAllReduceOrderingResource : TF_ResourceBase<"NcclAllReduceOrdering">;
def TF_GlobalIterIdResource : TF_ResourceBase<"GlobalIterId">;
def TF__XlaRunResource : TF_ResourceBase<"_XlaRun">;
// Fake resource, see `TF_MustExecute` below.
def TF_MustExecuteResource : TF_ResourceBase<"MustExecute">;

// Value-based side effects
//
// Value-based side effect traits are attached to op operands or results to
// signal what type of resource is accessed and in which way.
def TF_VariableRead : MemRead<TF_VariableResource>;
def TF_StackRead : MemRead<TF_StackResource>;
def TF_TensorArrayRead : MemRead<TF_TensorArrayResource>;
def TF_LookupTableRead : MemRead<TF_LookupTableResource>;
def TF_DatasetSeedGeneratorRead : MemRead<TF_DatasetSeedGeneratorResource>;
def TF_DatasetMemoryCacheRead : MemRead<TF_DatasetMemoryCacheResource>;
def TF_DatasetIteratorRead : MemRead<TF_DatasetIteratorResource>;

def TF_VariableWrite : MemWrite<TF_VariableResource>;
def TF_StackWrite : MemWrite<TF_StackResource>;
def TF_TensorArrayWrite : MemWrite<TF_TensorArrayResource>;
def TF_SummaryWrite : MemWrite<TF_SummaryResource>;
def TF_LookupTableWrite : MemWrite<TF_LookupTableResource>;
def TF_DatasetSeedGeneratorWrite : MemWrite<TF_DatasetSeedGeneratorResource>;
def TF_DatasetMemoryCacheWrite : MemWrite<TF_DatasetMemoryCacheResource>;
def TF_DatasetIteratorWrite : MemWrite<TF_DatasetIteratorResource>;

def TF_VariableAlloc : MemAlloc<TF_VariableResource>;
def TF_StackAlloc : MemAlloc<TF_StackResource>;
def TF_TensorArrayAlloc : MemAlloc<TF_TensorArrayResource>;
def TF_SummaryAlloc : MemAlloc<TF_SummaryResource>;
def TF_LookupTableAlloc : MemAlloc<TF_LookupTableResource>;
def TF_DatasetSeedGeneratorAlloc : MemAlloc<TF_DatasetSeedGeneratorResource>;
def TF_DatasetMemoryCacheAlloc : MemAlloc<TF_DatasetMemoryCacheResource>;
def TF_DatasetIteratorAlloc : MemAlloc<TF_DatasetIteratorResource>;

def TF_StackFree : MemFree<TF_StackResource>;
def TF_TensorArrayFree : MemFree<TF_TensorArrayResource>;
def TF_SummaryFree : MemFree<TF_SummaryResource>;
def TF_DatasetSeedGeneratorFree : MemFree<TF_DatasetSeedGeneratorResource>;
def TF_DatasetMemoryCacheFree : MemFree<TF_DatasetMemoryCacheResource>;
def TF_DatasetIteratorFree : MemFree<TF_DatasetIteratorResource>;

// Op-based side effects

// Op-based side effect traits can be used to enforce certain execution order
// constraints, in particular for ops that don't use resource handles (those
// typically have value-based side effects). For a `read` effect, all instances
// of ops with the trait keep their order to all ops with unknown side effects
// (e.g. `stateful` ops). For a `write` effect, all instances of ops with the
// trait stay in order, and they also keep their order to all unknown side-
// effecting ops. Note that for `read` effects ops might be pruned if nothing
// depends on them.
def TF_GeneratorOpSideEffect : MemoryEffects<[MemWrite<TF_GeneratorOpResource>]>;

def TF_TPUEmbeddingWriteEffect : MemoryEffects<[MemWrite<TF_TPUEmbeddingResource>]>;
def TF_TPUEmbeddingReadEffect : MemoryEffects<[MemRead<TF_TPUEmbeddingResource>]>;

def TF_SendSideEffect : MemoryEffects<[MemWrite<TF_SendResource>]>;
def TF_RecvSideEffect : MemoryEffects<[MemWrite<TF_RecvResource>]>;
def TF_XlaHostComputeSideEffect : MemoryEffects<[MemWrite<TF_XlaHostComputeResource>]>;

def TF_WriteTrainingPredictions : MemoryEffects<[MemWrite<TF_WriteTrainingPredictionsResource>]>;
def TF_RandomGeneratorSideEffect : MemoryEffects<[MemWrite<TF_RandomGeneratorResource>]>;

// Special effect for keeping `CollectiveReduce` ops in order.
def TF_CollectiveReduceOrderingEffect : MemoryEffects<[MemWrite<TF_CollectiveReduceOrderingResource>]>;

// Special effect for keeping `NcclAllReduce` ops on the same device in order.
def TF_NcclAllReduceOrderingEffect : MemoryEffects<[MemWrite<TF_NcclAllReduceOrderingResource>]>;

def TF_GlobalIterIdEffect : MemoryEffects<[MemRead<TF_GlobalIterIdResource>]>;

// Trait for enforcing that a side-effecting op is executed, even if it would be
// considered dead by MLIR (see b/195782952).
// The trait is implemented as a write effect for a fake resource which is
// ignored by side effect analysis, so it does not affect execution order
// constraints and control dependencies at all (for example, multiple ops with
// this trait do not have to execute in order).
def TF_MustExecute : MemoryEffects<[MemWrite<TF_MustExecuteResource>]>;

#endif // TF_OP_BASE
