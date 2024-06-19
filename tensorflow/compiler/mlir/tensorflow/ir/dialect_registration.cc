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

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
//#include "mlir/Dialect/MLProgram/IR/MLProgram.h"  // from @llvm-project
//#include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/control_flow_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
/*
// Static initialization for TF dialect registration.
static DialectRegistration<TFControlFlow::TFControlFlowDialect>
    tf_control_flow_ops;
static DialectRegistration<TF::TensorFlowDialect> tf_ops;
static DialectRegistration<tf_executor::TensorFlowExecutorDialect>
    tf_excutor_dialect;
static DialectRegistration<tf_device::TensorFlowDeviceDialect>
    tf_device_dialect;
*/

// Inserts all the TensorFlow dialects in the provided registry. This is
// intended for tools that need to register dialects before parsing .mlir files.
// If include_extensions is set (default), also registers extensions. Otherwise
// it is the responsibility of the caller, typically required when the registry
// is appended to the context in a parallel context, which does not allow for
// extensions to be added.
inline void RegisterAllTensorFlowDialectsImpl(DialectRegistry &registry,
                                              bool include_extensions = true) {
  registry
      .insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
//              mlir::ml_program::MLProgramDialect,
              TF::TensorFlowDialect,
              TFControlFlow::TFControlFlowDialect,
              tf_device::TensorFlowDeviceDialect,
              tf_executor::TensorFlowExecutorDialect>();
  if (include_extensions) {
    mlir::func::registerAllExtensions(registry);
  }
}

// FIXME: this is currently not called 

// Inserts all the TensorFlow dialects in the provided registry. This is
// intended for tools that need to register dialects before parsing .mlir files.
inline void RegisterAllTensorFlowDialects(DialectRegistry &registry) {
  RegisterAllTensorFlowDialectsImpl(registry, true);
}

}  // namespace mlir
