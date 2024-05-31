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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project 
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project 
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project 
#include "mlir/IR/MLIRContext.h"  // from @llvm-project 
#include "mlir/IR/Operation.h"  // from @llvm-project 
#include "mlir/IR/PatternMatch.h"  // from @llvm-project 
#include "mlir/Pass/Pass.h"  // from @llvm-project 
#include "mlir/Pass/PassOptions.h"  // from @llvm-project 
#include "mlir/Support/LLVM.h"  // from @llvm-project 
#include "mlir/Support/LogicalResult.h"  // from @llvm-project  

namespace mlir {
namespace TF {

using FunctionPassBase = OperationPass<func::FuncOp>;
using ModulePassBase = OperationPass<mlir::ModuleOp>;

// Transforms functional control flow operations in the standard TensorFlow
// dialect to MLIR Control Flow Graph (CFG) form.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFFunctionalControlFlowToCFG();

// Optimizes Tensorflow graph.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFOptimizePass();

// Creates a pass that canonicalizes legacy compilation and replication
// attributes.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateCanonicalizeCompileAndReplicateAttributesPass();

// Creates a pass that drops `shape_invariant` attribute from While/WhileRegion
// ops.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateDropWhileShapeInvariantPass();

// Creates a pass that drops `shape_invariant` attribute from While/WhileRegion
// ops within device cluster.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateDropWhileShapeInvariantInDeviceClusterPass();

// Creates a pass that moves writes to replicate invariant resource variables
// outside tf_device.replicate op.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateHoistReplicateInvariantResourceWritesPass();

// Transforms functional control flow operations in the TensorFlow dialect to
// MLIR Control Flow Graph (CFG) form.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTFFunctionalControlFlowToCFG();

// Transforms functional control flow operations in the TensorFlow dialect to
// their region based counterparts.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTFFunctionalControlFlowToRegions();
std::unique_ptr<OperationPass<ModuleOp>> CreateTFFunctionalControlFlowToRegions(
    bool allow_passthrough_args);

// Transforms region bases control flow operations in the TensorFlow dialect to
// their functional counterparts.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTFRegionControlFlowToFunctional();

// Materialize the MlirPassthroughOp by replacing it with the MLIR module
// attached as an attribute.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateMaterializePassthroughOpPass();

// Replicates the TensorList init op by undoing some CSE needed for correct
// shape assignment in shape_inference.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateReplicateTensorListInitOpsPass();

// Performs Shape Inference on the TensorFlow dialect using the global registry.
std::unique_ptr<OperationPass<ModuleOp>> CreateTFShapeInferencePass(
    ArrayRef<ArrayRef<int64_t>> input_shapes = {});

// Performs TF.data optimizations.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFDataOptimizationPass();

std::unique_ptr<OperationPass<func::FuncOp>> CreateMoveTransposesPass();
std::unique_ptr<OperationPass<func::FuncOp>> CreateLayoutAssignmentPass();

// Guarantee that all FuncOp's have a single use.
std::unique_ptr<OperationPass<ModuleOp>> CreateGuaranteeAllFuncsOneUsePass();

// Optional pass which will unroll BatchMatMul and use only MatMul
std::unique_ptr<OperationPass<func::FuncOp>> CreateUnrollBatchMatMulPassPass();

// Optional pass which will map TF BatchMatMul to TF Einsum
std::unique_ptr<OperationPass<func::FuncOp>> CreateBatchMatMulToEinsumPass();

// Pass that transform Einsum to other TF Ops for the supported variants.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTransformEinsumPass();

// Optimizes Tensorflow graph.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFOptimizePass();
void RegisterTFOptimizePassPipeline();

// Creates pass to rewrite RecvTPUEmbeddingActivationsOp and
// SendTPUEmbeddingGradients ops to internal variants.
std::unique_ptr<OperationPass<func::FuncOp>> CreateRewriteTPUEmbeddingOpsPass();

// Performs specific fusion for GPU targets.
std::unique_ptr<OperationPass<func::FuncOp>> CreateGpuOpFusionPass();

// Creates a pass that decomposes to be compiled ReduceDataset ops into a while
// loop that iterates the dataset and calls the reduction function.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDecomposeReduceDatasetPass();

// Create a pass that convert ops that copy tensors between devices, e.g.
// tf.Identity.
std::unique_ptr<OperationPass<mlir::func::FuncOp>>
CreateTensorDeviceCopyConversionPass();

// Returns a pass that folds tf.BroadcastTo nodes with subsequent nodes if they
// have built in broadcasting support.
std::unique_ptr<OperationPass<func::FuncOp>> CreateBroadcastFoldPass();

void populateTfControlFlowToScfPatterns(MLIRContext* context,
                                        RewritePatternSet* patterns);
// Create a pass to convert TensorFlow control flow to SCF.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTfControlFlowToScfPass();

struct LayoutOptimizationPipelineOptions
    : public PassPipelineOptions<LayoutOptimizationPipelineOptions> {
  Option<std::string> force_data_format{
      *this, "force-data-format",
      llvm::cl::desc("Force data format for all layout sensitive ops")};
  Option<bool> skip_fold_transpose_in_ops{
      *this, "skip-fold-transpose-in-ops",
      llvm::cl::desc("Skip folding transpose operands in Ops which can support "
                     "different layouts.")};
};

// Layout optimization assigns optimal data layout for layout sensitive
// operations, and cancels all redundant transposes.
void CreateLayoutOptimizationPipeline(
    OpPassManager& pm,  // NOLINT - MLIR contract is pass by mutable reference.
    const LayoutOptimizationPipelineOptions& options);

struct StandardPipelineOptions
    : public PassPipelineOptions<StandardPipelineOptions> {
  Option<bool> enable_inliner{*this, "enable-inliner",
                              llvm::cl::desc("Enable inliner."),
                              llvm::cl::init(false)};
  Option<bool> form_clusters{*this, "form-clusters",
                             llvm::cl::desc("Enable Cluster Formation pass."),
                             llvm::cl::init(false)};
};

// Propagates the pass manager with the passes involved in transforming or
// optimizing an MLIR graph without any target specialization.
// NOLINTNEXTLINE - MLIR contract is pass by mutable reference.
void CreateTFStandardPipeline(OpPassManager& pm,
                              const StandardPipelineOptions& options);

}  // namespace TF

namespace TFControlFlow {
// Raises from the "TensorFlow Control Flow" dialect to the standard TensorFlow
// dialect.
std::unique_ptr<OperationPass<func::FuncOp>> CreateRaiseTFControlFlowPass();

}  // namespace TFControlFlow

namespace tf_executor {
class GraphOp;

// Create a pass to merge IslandOps from TFExecutor dialect.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFExecutorIslandCoarseningPass();

// Create a pass to prune tf_executor.graph from dead nodes.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFExecutorGraphPruningPass(
    llvm::ArrayRef<std::string> ops_to_preserve = {});

// Prune a tf_executor.graph operation from dead nodes.
void prune_graph(GraphOp graph);

}  // namespace tf_executor

namespace TFDevice {
// Creates a pass that forms clusters from instructions that are assigned to
// same device.
std::unique_ptr<OperationPass<func::FuncOp>> CreateClusterFormationPass();

// Creates a pass that outlines regions of tf_device.launch operations.
std::unique_ptr<OperationPass<mlir::ModuleOp>> CreateClusterOutliningPass();
}  // namespace TFDevice

namespace detail {
#define GEN_PASS_DECL_CLUSTEROUTLININGPASS
#define GEN_PASS_DECL_LAUNCHOUTLININGPASS
#define GEN_PASS_DECL_EXECUTORGRAPHPRUNINGPASS
#define GEN_PASS_DECL_FUNCTIONALCONTROLFLOWTOCFGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"
}
using namespace detail;

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_
