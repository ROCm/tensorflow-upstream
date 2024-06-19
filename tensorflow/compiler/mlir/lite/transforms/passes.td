/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/Pass/PassBase.td"

def LegalizeTFPass : Pass<"tfl-legalize-tf", "mlir::func::FuncOp"> {
  let summary = "Legalize from TensorFlow to TensorFlow Lite dialect.";
  let constructor = "CreateLegalizeTFPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect" ,
    "quant::QuantizationDialect",
    "quantfork::QuantizationForkDialect"
  ];
  let options = [
  ];
}

def LowerStaticTensorListPass : Pass<"tfl-lower-static-tensor-list", "mlir::ModuleOp"> {
  let summary = "Lower TensorList ops within TensorFlow Lite dialect.";
  let constructor = "CreateLowerStaticTensorListPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect",
    "quant::QuantizationDialect",
    "quantfork::QuantizationForkDialect"
  ];
  let options = [
  ];
}

def OptimizePass : Pass<"tfl-optimize", "mlir::func::FuncOp"> {
  let summary = "Optimize within the TensorFlow Lite dialect";
  let constructor = "CreateOptimizePass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
  let options = [
  ];
}

def PostQuantizePass : Pass<"tfl-post-quantize", "mlir::func::FuncOp"> {
  let summary = "Apply post quantization clean up after quantization.";
  let constructor = "CreatePostQuantizePass()";
  let options = [
      Option<"emit_quant_adaptor_ops_", "emit-quant-adaptor-ops",
             "bool", "false",
             "Enable canonicalization during optimization pass.">,
  ];
}

def PrepareCompositeFunctionsPass : Pass<"tfl-prepare-composite-funcs-tf", "mlir::ModuleOp"> {
  let summary = "Prepares composite functions in Tensorflow dialect of MLIR.";
  let constructor = "CreatePrepareCompositeFunctionsPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
  let description = [{
      This pass uses mechanisms listed in RFC:
      https://github.com/tensorflow/community/pull/113
      It prepares composite functions that are attributed to indicate
      a specific interface (LSTM, SVDF, Embedding lookup etc.) by replacing the
      body with the corresponding fused TFLite op. The replacement need not
      always be a fused op, though that is the primary use case.
  }];
  let options = [
      Option<"tfl_fuse_tftext_", "fuse-tftext", "bool", "false",
             "Fuse TF.Text API ops when it's true">
  ];

}

def PrepareQuantizePass : Pass<"tfl-prepare-quantize", "mlir::func::FuncOp"> {
  let summary = "Remove qdq from input and output nodes after quantization.";
  let constructor = "CreatePrepareQuantizePass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect",
    "quant::QuantizationDialect",
    "quantfork::QuantizationForkDialect"
  ];
  let options = [
      Option<"quantize_sign_", "quantize-sign", "bool", "false",
             "When `quantize_sign` is true, constant tensors will use int8 quantization scheme.">,
  ];
}

def PrepareTFPass : Pass<"tfl-prepare-tf", "mlir::func::FuncOp"> {
  let summary = "Prepare TF for legalization to TensorFlow Lite dialect.";
  let constructor = "CreatePrepareTFPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect",
    "mlir::quant::QuantizationDialect",
    "mlir::quantfork::QuantizationForkDialect",
    "mhlo::MhloDialect"
  ];
  let options = [
  ];
}

// keep
def QuantizePass : Pass<"tfl-quantize", "mlir::func::FuncOp"> {
  let summary = "Apply quantization on models in TensorFlow Lite dialect.";
  let constructor = "CreateQuantizePass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect",
    "mlir::quant::QuantizationDialect",
    "mlir::quantfork::QuantizationForkDialect"
  ];

  let options = [
  ];
}

// keep
def TrimFunctionsPass : Pass<"tfl-trim-funcs-tf", "mlir::ModuleOp"> {
  let summary = "Trim functions to restrict them to a specified allowlist prior to legalization to TensorFlow lite dialect";
  let constructor = "CreateTrimFunctionsPass()";
  let options = [
      ListOption<"trim_funcs_allowlist_", "trim-funcs-allowlist", "std::string",
                 "comma separated list of allowlisted functions. The first "
                 "function specified will be used as main">
  ];
}

def ExtractOphintPass: Pass<"tfl-extract-ophint", "mlir::ModuleOp"> {
  let summary = "Extract Ophint for TfLite dialect";
  let constructor = "CreateExtractOphintPass();";
}