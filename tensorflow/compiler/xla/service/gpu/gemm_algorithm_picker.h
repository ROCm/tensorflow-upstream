/* Copyright 2019 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_GEMM_ALGORITHM_PICKER_H_
#define XLA_SERVICE_GPU_GEMM_ALGORITHM_PICKER_H_

#include <functional>
#include <optional>
#include <string_view>

// #include "absl/container/flat_hash_set.h"
// #include "absl/status/statusor.h"
// #include "absl/strings/string_view.h"
// #include "absl/types/span.h"
#include "tensorflow/core/protobuf/autotune_results.pb.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/gpu/autotuner_util.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// GemmAlgorithmPicker supports two modes: device and deviceless.
// In device mode, we run autotuning on the device and store autotune results.
// In deviceless mode, we pass in some information related to the device and
// use stored autotune results to rewrite Gemm instructions. If the required
// autotune result is not stored, then algorithm is set to kRuntimeAutotuning.
class GemmAlgorithmPicker : public HloModulePass {
 public:
  explicit GemmAlgorithmPicker(AutotuneConfig config) : config_(config) {}

  absl::string_view name() const override { return "gemm-algorithm-picker"; }

  size_t num_algorithms_left() const {
    return num_algorithms_left_;
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(HloModule* module) override;

 private:
  AutotuneConfig config_;
  // The number of valid algorithms used for autotuning (from the last call),
  // to be used for testing purposes.
  size_t num_algorithms_left_ = 0; 
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GEMM_ALGORITHM_PICKER_H_
