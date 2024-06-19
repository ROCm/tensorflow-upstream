include "tensorflow/compiler/mlir/lite/ir/tfl_op_base.td"

// A blank .td that exists to generate attribute type definitions (TFL_LSTMKernelTypeAttr, etc.);
// we can't run tablegen directly on tfl_ops.td, because it results in double definitions
// of MLIR's internal mlir::arith::... entities.