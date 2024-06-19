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

#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/decode_constant.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"

namespace mlir {
/// Create a pass to convert from the TFExecutor to the TF control dialect.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFExecutorToControlDialectConversion();
}  // namespace mlir

namespace tensorflow {

bool ShouldRunQuantizePasses(mlir::ModuleOp m) {
  if (mlir::func::FuncOp main_fn = m.lookupSymbol<mlir::func::FuncOp>("main")) {
    for (int i=0; i<main_fn.getNumArguments(); i++)   {
      if ( main_fn.getArgAttrOfType<mlir::UnitAttr>(i, "tf.quantize") !=
           mlir::Attribute() )
        return true;
    }
  }
  return false;
}

// TODO: API has both addPass and addNestedPass; 2.x uses addNestedPass almost exclusively; what's the difference?
void AddTFToTFLConversionPasses(bool emit_builtin_tflite_ops, bool run_quantize,
                                bool emit_quant_adaptor_ops,
                                bool lower_tensor_list_ops,
                                mlir::OpPassManager& pass_manager) {
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::CreateTFExecutorToControlDialectConversion());
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::TFControlFlow::CreateRaiseTFControlFlowPass());
  // Ophint extraction will happen after island extraction pass.
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::TFL::CreateExtractOphintPass());

  if (lower_tensor_list_ops) {
    // Execute this pass before `CanonicalizerPass` in case some TensorList
    // ops are constant folded into variant types.
    // TODO(b/137125056): Move this pass after `CanonicalizerPass` after we
    // handle constant ops that produce `TensorList`.
    // TODO(haoliang): Add this pass by default.
    pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::TFL::CreateLowerStaticTensorListPass());
  }

  // TODO(jpienaar): Revise post dialect constants.
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::TF::CreateDecodeConstantPass());
  // Canonicalization includes const folding, which is utilized here to optimize
  // away ops that can't get constant folded after PrepareTF pass. For example,
  // tf.Conv2D is split into tf.Transpose and tfl.Conv2D.
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

  // The below passes only make sense if Builtin TFLite ops are enabled
  // for emission.
  if (emit_builtin_tflite_ops) {
    // Prepare for TFLite dialect, rerun canonicalization, and then legalize to
    // the TFLite dialect.
    pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::TFL::CreatePrepareTFPass());
    pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::TFL::CreateLegalizeTFPass());
    pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::TFL::CreateOptimizePass());
    if (run_quantize) {
      pass_manager.addPass(mlir::TFL::CreatePrepareQuantizePass(
          /*quantize_sign=*/false));
      pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::TFL::CreateQuantizePass());
      pass_manager.addNestedPass<mlir::func::FuncOp>(
          mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops));
    }
    pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  }
}

}  // namespace tensorflow
