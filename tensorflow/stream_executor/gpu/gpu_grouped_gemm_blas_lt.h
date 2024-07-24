/* Copyright 2023 The OpenXLA Authors.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_GROUPED_GEMM_BLAS_LT_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_GROUPED_GEMM_BLAS_LT_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/stream_executor/gpu/gpu_blas_lt.h"

namespace stream_executor {
  
namespace  gpu {

using GroupedGemmConfig = GroupedGemmConfig;

namespace {

auto AsTuple(const GroupedGemmConfig& p) {
    // NOTE: alpha, beta and data pointers are not included in cache !!
  return std::make_tuple(p.m, p.n, p.k, p.batch_count,
                         p.trans_a, p.trans_b,
                         p.type_a, p.type_b, p.type_c, p.type_d,
                         p.lda, p.ldb, p.ldc, p.ldd, 
                         p.compute_type);
}

} // namespace

bool operator ==(const GroupedGemmConfig& rhs,
        const GroupedGemmConfig& lhs);

template <typename H>
H AbslHashValue(H h, const GroupedGemmConfig& params) {
  return H::combine(std::move(h), AsTuple(params));
}

struct GroupedGemmRunner {

  GroupedGemmRunner() {}

  Stream& operator()(Stream& stream, blas::Transpose transa, 
      blas::Transpose transb, uint64 m, uint64 n, uint64 k, 
      const void *alpha, blas::DataType type_a, const void** a, int lda, 
      blas::DataType type_b, const void** b, int ldb, const void *beta,
      blas::DataType type_c, void** c, int ldc, int batch_count);

private:
  absl::flat_hash_map<GroupedGemmConfig, BlasLt::GroupedMatmulPlanPtr> map_;
};

} // namespace gpu

} // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_GROUPED_GEMM_BLAS_LT_H_
