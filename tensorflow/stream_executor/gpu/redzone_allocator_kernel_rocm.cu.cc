/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>

#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator_kernel.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace {
__global__ void redzone_checker_kernel(uint8_t* input_buffer,
                                       uint8_t redzone_pattern,
                                       uint64_t buffer_length,
                                       uint32_t* out_mismatched_ptr) {
  uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;
  if (input_buffer[idx] != redzone_pattern) atomicAdd(out_mismatched_ptr, 1);
}
}  // namespace

namespace stream_executor {

port::StatusOr<const ComparisonKernel*> GetComparisonKernel(
    StreamExecutor* executor, GpuAsmOpts /*gpu_asm_opts*/) {

  static auto kernel =
      executor->CreateTypedKernel<DeviceMemory<uint8>, uint8, uint64,
                                 DeviceMemory<uint64>>(
           "redzone_checker", reinterpret_cast<void*>(
                                          redzone_checker_kernel));

  if (!kernel.ok()) return kernel.status();
  return kernel.ValueOrDie().get();
}

}  // namespace stream_executor
