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

// This file defines the operations used in the standard MLIR TensorFlow dialect
// after control dependences are raise to the standard form.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {

class TensorFlowRegistryEffectInterfaceFallback;

class TensorFlowDialect final : public Dialect {
 public:
  explicit TensorFlowDialect(MLIRContext *context);
  ~TensorFlowDialect() override;

  static StringRef getDialectNamespace() { return "tf"; }

  // Overrides to redirect to tf_type dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
  Type parseType(DialectAsmParser &parser) const override;

  // Gradient attribute ("tf.gradient") in the list of NamedAttributes in a
  // function references to its gradient function. This attribute in TensorFlow
  // Dialect is used to model TF GradientDef. GetGradientAttrName() returns the
  // string description of gradient attribute.
  static StringRef GetGradientAttrName() { return "tf.gradient"; }

  // This attribute marks if a function is stateful.
  // Returns the string description of stateful attribute.
  static StringRef GetStatefulAttrName() { return "tf.signature.is_stateful"; }

  // Returns true if the op can be duplicated during transformations.
  static bool CanDuplicate(Operation *op);

  // Returns true if the op can have side effects.
  static bool CanHaveSideEffects(Operation *op);

  // Registered hook to materialize a constant operation from a given attribute
  // value with the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  typedef std::function<void(TensorFlowDialect &dialect)> AdditionalOpFunction;

  // Register an op registration hook which is invoked during construction.
  //
  // A hook may use the public addOperations() method to add additional
  // operations to the dialect. Hooks will only apply to subsequent
  // instantations of the Dialect/MLIRContext.
  static void RegisterAdditionalOperationHook(TypeID uniqueId,
                                              AdditionalOpFunction fn);

  // Re-define publicly the protected addOperations() method from the Dialect
  // class, usually used in a Dialect constructor. This allows hook
  // functions to register operations on the TensorFlow dialect using the
  // same interface.
  template <typename... Args>
  void addOperations() {
    Dialect::addOperations<Args...>();
  }

  using ConstantFoldHook = LogicalResult (*)(Operation *, ArrayRef<Attribute>,
                                             SmallVectorImpl<OpFoldResult> &);
  static void RegisterConstantFoldHook(ConstantFoldHook fn) {
    constant_fold_hook_ = std::move(fn);
  }

  static LogicalResult constantFold(Operation *op, ArrayRef<Attribute> operands,
                                    SmallVectorImpl<OpFoldResult> &results) {
    if (constant_fold_hook_) return constant_fold_hook_(op, operands, results);
    return failure();
  }

  static bool HasConstantFoldHook() { return constant_fold_hook_; }

  // Provides a hook for op interface.
  void *getRegisteredInterfaceForOp(mlir::TypeID interface,
                                    mlir::OperationName opName) override;

 private:
  static ConstantFoldHook constant_fold_hook_;

  // Storage for a custom fallback interface.
  TensorFlowRegistryEffectInterfaceFallback *fallback_effect_op_interface_;
};

// TODO(b/131258166): TensorFlow's mutex.h defines a `mutex_lock` macro, whose
// purpose is to catch bug on `tensorflow::mutex_lock`. We don't use
// `tensorflow::mutex_lock` here but we have ops (`tf.MutexLock` and
// `tf.ConsumeMutexLock`) with getter methods named as `mutex_lock()`. Need to
// undefine here to avoid expanding the getter symbol as macro when including
// both mutex.h and this header file.
#undef mutex_lock

}  // namespace TF
}  // namespace mlir

#define TF_DIALECT_REGISTER_ADDITIONAL_OPERATIONS(hookFn)           \
  {                                                                 \
    static bool key;                                                \
    ::mlir::TF::TensorFlowDialect::RegisterAdditionalOperationHook( \
        ::mlir::TypeID::getFromOpaquePointer(&key), hookFn);        \
  }

using namespace mlir;
#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_H_
