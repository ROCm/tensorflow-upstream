include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/Quant/QuantOpsBase.td"

def TFL_Dialect : Dialect {
  let name = "tfl";

  let description = [{
    The TensorFlow Lite dialect.

    This dialect maps to TensorFlow Lite operations.

    Invariants:

    * All values are of Tensor type (in particular, scalars are
      represented using zero-dimensional tensors);
  }];

  let cppNamespace = "::mlir::TFL";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    ParseResult parseOneResultSameOperandTypeOp(OpAsmParser &parser,
                                                OperationState &result);

    void printOneResultOp(Operation *op, OpAsmPrinter &p);

    // Registered hook to materialize a constant operation from a given
    // attribute value with the desired resultant type.
    Operation *materializeConstant(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) override;
  }];
}

// A string attribute whose value are one of the values in `cases`.
// Referred TF_AnyStrAttrOf in tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td
class TFL_AnyStrAttrOf<list<string> cases> : StringBasedAttr<
  CPred<!foldl(
      "$_self.cast<StringAttr>().getValue() == \"" # !head(cases) # "\"",
      !foreach(case, !tail(cases),
               "$_self.cast<StringAttr>().getValue() == \"" # case # "\""),
      prev, cur, prev # " || " # cur)>,
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;


// LSTM Kernel Type attributes
def TFL_LSTM_KT_FULL  : I32EnumAttrCase<"FULL", 0>;
def TFL_LSTM_KT_BASIC  : I32EnumAttrCase<"BASIC", 1>;
def TFL_LSTMKernelType : I32EnumAttr<"LSTMKernelType", "lstm_kernel_type",
   [
     TFL_LSTM_KT_FULL,  TFL_LSTM_KT_BASIC
   ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::TFL";
}
def TFL_LSTMKernelTypeAttr : EnumAttr<TFL_Dialect, TFL_LSTMKernelType,
    "lstm_kernel_type_attr">;

//===----------------------------------------------------------------------===//
// Padding enum definitions.
//===----------------------------------------------------------------------===//

// Allowed padding cases
// These should match the padding enum in TFLite schema.
def TFL_PADEnum_Same  : I32EnumAttrCase<"SAME", 0>;
def TFL_PADEnum_Valid : I32EnumAttrCase<"VALID", 1>;
def TFL_PaddingAttr : TFL_AnyStrAttrOf<[
      TFL_PADEnum_Same.symbol, TFL_PADEnum_Valid.symbol
    ]>;
def TFL_PAD_Same  : ConstantStrAttr<TFL_PaddingAttr, TFL_PADEnum_Same.symbol>;
def TFL_PAD_Valid : ConstantStrAttr<TFL_PaddingAttr, TFL_PADEnum_Valid.symbol>;

// MirrorPadding type attributes
def TFL_MIRRORPAD_Reflect : I32EnumAttrCase<"REFLECT", 0>;
def TFL_MIRRORPAD_Symmetric : I32EnumAttrCase<"SYMMETRIC", 1>;
def TFL_MirrorPaddingType : I32EnumAttr<"MirrorPaddingType", "mirror_pad_enum", [
      TFL_MIRRORPAD_Reflect, TFL_MIRRORPAD_Symmetric
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::TFL";
}
def TFL_MirrorPaddingAttr : EnumAttr<TFL_Dialect, TFL_MirrorPaddingType,
    "mirror_pad_attr">;

//===----------------------------------------------------------------------===//
// TFL runtime type verification of operand/result types.

def TFL_RuntimeVerification : OpInterface<"TflRuntimeVerifyOpInterface"> {
  let description = [{
    Interface to verify TFLite runtime op verification.

    This verifies that the converted TFLite ops has operand/result type
    supported by the TFLite runtime.
  }];

  let methods = [
    StaticInterfaceMethod<
      [{Returns whether the op's operands/results are supported by runtime.}],
      "LogicalResult", "VerifyTflRuntimeConstraints",
      (ins "Operation*":$op, "bool":$emit_error_on_verify_fail)
    >,
  ];
}

def TFL_ConstBytesAttr : AttrDef<TFL_Dialect, "ConstBytes"> {
  let summary = "A string attribute representation of compiled bytes";
  let description = [{
    Syntax Examples:

    ```mlir
    #tfl<const_bytes : "0xDEADBEEF">
    ```
  }];
  let mnemonic = "const_bytes";
  let parameters = (ins StringRefParameter<"">:$value);
  let hasCustomAssemblyFormat = 1;
}

// Specify this trait if the op does have quantizable output. Quantizers will
// apply quantization on this op.
def QuantizableResult : NativeOpTrait<"::mlir::quant::QuantizableResult">;

