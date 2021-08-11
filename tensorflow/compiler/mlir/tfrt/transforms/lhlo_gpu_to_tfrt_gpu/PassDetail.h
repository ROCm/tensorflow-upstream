// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LHLO_GPU_TO_TFRT_GPU_PASSDETAIL_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LHLO_GPU_TO_TFRT_GPU_PASSDETAIL_H_

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"  // from @tf_runtime

namespace tensorflow {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/gpu_passes.h.inc"

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LHLO_GPU_TO_TFRT_GPU_PASSDETAIL_H_
