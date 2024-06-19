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

// This transformation pass applies quantization propagation on TFLite dialect.

#include "absl/memory/memory.h"
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

//===----------------------------------------------------------------------===//
// The prepare-quantize Pass.
//
namespace mlir {
namespace TFL {

namespace {
#define GEN_PASS_DEF_PREPAREQUANTIZEPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Applies prepare quantization on the model in TFL dialect. This pass runs
// before the quantization pass and propagate the quantization parameters
// across ops. This step is necessary for post-training quantization and also
// making the quantization rule for some operations in the quantization-aware
// training quantization simpler.
class PrepareQuantizePass : public impl::PrepareQuantizePassBase<PrepareQuantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareQuantizePass)
  // Constructor used by the PassRegistration.
  explicit PrepareQuantizePass() : quantize_sign_(false) {}

  // Constructor used by manually creating the pass.
  explicit PrepareQuantizePass(bool quantize_sign)
      : quantize_sign_(quantize_sign) {}

  void runOnOperation() override;

 private:
  bool quantize_sign_;
};

#include "tensorflow/compiler/mlir/lite/utils/generated_op_quant_spec_getters.inc"

void PrepareQuantizePass::runOnOperation() {
  ApplyQuantizationParamsPropagation(getFunction(), quantize_sign_,
                                     GetOpQuantSpec);
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PrepareQuantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass(
    bool quantize_sign) {
  return std::make_unique<PrepareQuantizePass>(quantize_sign);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass() {
    return CreatePrepareQuantizePass(true);
}

}  // namespace TFL
}  // namespace mlir
