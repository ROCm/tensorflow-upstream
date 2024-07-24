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

#include "tensorflow/stream_executor/gpu/gpu_grouped_gemm_blas_lt.h"
#include "tensorflow/stream_executor/stream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

namespace stream_executor {

namespace gpu {

bool operator ==(const GroupedGemmConfig& rhs,
        const GroupedGemmConfig& lhs) {
  return AsTuple(rhs) == AsTuple(lhs);
}

Stream& GroupedGemmRunner::operator()(Stream& stream, 
      blas::Transpose transa, blas::Transpose transb, 
      uint64 m, uint64 n, uint64 k, 
      const void *alpha, blas::DataType type_a, const void** a, int lda, 
      blas::DataType type_b, const void** b, int ldb, const void *beta,
      blas::DataType type_c, void** c, int ldc, int batch_count) {

  auto compute_type = gpu::GetBlasComputationType(type_a, type_c, 0).ValueOrDie();

  GroupedGemmConfig cfg{
    .m = (int64)m,
    .n = (int64)n,
    .k = (int64)k,
    .batch_count = (int64)batch_count,
    .trans_a = transa,
    .trans_b = transb,
    .alpha = alpha,
    .beta = beta,
    .type_a = type_a,
    .type_b = type_b,
    .type_c = type_c,
    .type_d = type_c, 
    .lda = (int64)lda,
    .ldb = (int64)ldb,
    .ldc = (int64)ldc,
    .ldd = (int64)ldc,
    .compute_type = compute_type,
    .a = a,
    .b = b,
    .c = const_cast< const void **>(c),
    .d = c,
  };

  auto res = map_.find(cfg);
  if(res == map_.end()) {
    auto plan_res = gpu::BlasLt::GetGroupedMatmulPlan(&stream, nullptr, cfg);
    if(!plan_res.ok()) {
      stream.CheckStatus(plan_res.status());
      return stream;
    }
    res = map_.emplace(cfg, std::move(plan_res.ValueOrDie())).first;

    auto& plan = res->second;
    auto algos = plan->GetAlgorithms().ValueOrDie();

    VLOG(0) << "++++++++++++++++++++++++++++++++++ added new config: " << map_.size() << 
        " algos found: " << algos.size();
    if(algos.empty()) {
      stream.SetError();
      return stream;
    }
    plan->SetAlgorithm(algos[0]);
  }

  auto& plan = res->second;
  auto status = plan->ExecuteOnStream(&stream, cfg);
  if(!status.ok()) {
    stream.CheckStatus(status);
  }
  return stream;
}

}  // namespace gpu

}  // namespace stream_executor
