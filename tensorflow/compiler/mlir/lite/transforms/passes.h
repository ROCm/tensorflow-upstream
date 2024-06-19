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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace mlir {
class FunctionPassBase;
class ModulePassBase;

namespace TFL {

// Creates an instance of the TensorFlow Lite dialect LegalizeTF pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeTFPass();

// Creates an instance of the TensorFlow Lite dialect Optimize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateOptimizePass();

// Creates an instance of the TensorFlow Lite dialect PrepareTF pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareTFPass();

// Creates an instance of the TensorFlow Lite dialect LowerStaticTensorList
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateLowerStaticTensorListPass();

// Creates an instance of the TensorFlow Lite dialect Quantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass();

// Creates an instance of the TensorFlow Lite dialect PrepareQuantize pass.
// When `quantize_sign` is true, constant tensors will use int8 quantization
// scheme.
// TODO(fengliuai): make the bit width configurable.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass();
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass(bool quantize_sign);

// Creates a instance of the TensorFlow Lite dialect PostQuantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizePass();
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizePass(
    bool emit_quant_adaptor_ops);

// Creates an instance of the TensorFlow Lite dialect TrimFunctions
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateTrimFunctionsPass();
std::unique_ptr<OperationPass<ModuleOp>> CreateTrimFunctionsPass(
    llvm::ArrayRef<std::string> trim_funcs_whitelist);

// Creates an instance of the TensorFlow Lite dialect PrepareCompositeFunctions
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePrepareCompositeFunctionsPass();

// Creates a instance of the TensorFlow Lite dialect ExtractOphint pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateExtractOphintPass();

#define GEN_PASS_DECL_DEFAULTQUANTPARAMSPASS
#define GEN_PASS_DECL_DENSETOSPARSEPASS
#define GEN_PASS_DECL_LEGALIZETFPASS
#define GEN_PASS_DECL_MODIFYIONODESPASS
#define GEN_PASS_DECL_OPTIMIZEPASS
#define GEN_PASS_DECL_POSTQUANTIZEPASS
#define GEN_PASS_DECL_PREPARECOMPOSITEFUNCTIONSPASS
#define GEN_PASS_DECL_PREPAREDYNAMICRANGEQUANTIZEPASS
#define GEN_PASS_DECL_PREPAREQUANTIZEPASS
#define GEN_PASS_DECL_PREPARETFPASS
#define GEN_PASS_DECL_QUANTIZEPASS
#define GEN_PASS_DECL_RAISECUSTOMOPSPASS
#define GEN_PASS_DECL_TRIMFUNCTIONSPASS
#define GEN_PASS_DECL_EXTRACTOPHINTPASS
#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"
}  // namespace TFL

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_
