/*  Copyright 2023 The OpenXLA Authors.
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *     ==============================================================================*/
#define LEGACY_HIPBLAS_DIRECT
#include <hip/hip_runtime.h>
#include "rocm/rocm_config.h"
#include "rocm/include/hipblaslt/hipblaslt-ext.hpp"
#include "rocm/include/hipblaslt/hipblaslt.h"

namespace stream_executor {
namespace rocm {

using namespace hipblaslt_ext;

namespace {

__global__ void CopyUserArgsKernel(UserArguments *dest_args,
    const void **a, const void **b, const void **c, void **d,
    uint32_t num_gemms)
{
  uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < num_gemms) {
  // writing ArrayOfStructs is not optimal..
    auto arg = dest_args[idx];
    arg.a = const_cast< void *>(a[idx]);
    arg.b = const_cast< void *>(b[idx]);
    arg.c = const_cast< void *>(c[idx]);
    arg.d = d[idx];
  //printf("idx: %d %p %p %p %p\n", idx, arg.a, arg.b, arg.c, arg.d);
  }
}
} // namespace

void GroupGemmUpdateArgs(hipStream_t stream,
      UserArguments *dev_args,
      //const gpu::GroupedGemmConfig& cfg
      const void **a, const void **b, const void **c, void **d,
      uint32_t num_gemms) {

  const uint32_t block_sz = 128,
  n_blocks = (num_gemms + block_sz - 1)/block_sz;
  hipLaunchKernelGGL(CopyUserArgsKernel, n_blocks,
  std::min(block_sz, num_gemms), 0,
  stream,
  dev_args,
  //static_cast< UserArguments *>(device_args_.opaque()),
  a, b, c, d, num_gemms);
}
}  // namespace rocm
}  // namespace stream_executor
