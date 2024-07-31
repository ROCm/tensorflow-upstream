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

#include "tensorflow/stream_executor/gpu/gpu_blas_lt_gemm_runner.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/compiler/xla/util.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

namespace stream_executor {
namespace gpu {

bool operator ==(const GroupedGemmConfig& rhs, const GroupedGemmConfig& lhs) {
  return AsTuple(rhs) == AsTuple(lhs);
}

bool operator ==(const StridedGemmConfig& rhs, const StridedGemmConfig& lhs) {
  return AsTuple(rhs) == AsTuple(lhs);
}

BlasLtGemmRunner::BlasLtGemmRunner() :
    mutex_(std::make_unique< absl::Mutex >()) { }

/*static*/ BlasLtGemmRunner& BlasLtGemmRunner::i(const Stream *stream) {
    static absl::Mutex m(absl::kConstInit);
    // Each GPU gets different cache instance
    static absl::flat_hash_map<void *, BlasLtGemmRunner> meta;
    absl::MutexLock lock(&m);
    auto res = meta.find(stream->parent());
    if(res != meta.end()) return res->second;
    BlasLtGemmRunner r;
    return meta.emplace(stream->parent(), std::move(r)).first->second;
}

xla::Status BlasLtGemmRunner::RunBatched(Stream& stream, blas::Transpose transa, 
      blas::Transpose transb, uint64 m, uint64 n, uint64 k, 
      const void *alpha, blas::DataType type_a, const void** a, int lda, 
      blas::DataType type_b, const void** b, int ldb, const void *beta,
      blas::DataType type_c, void** c, int ldc, int batch_count) {

  TF_ASSIGN_OR_RETURN(auto compute_type, 
            gpu::GetBlasComputationType(type_a, type_c, 0));

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

  absl::MutexLock lock(mutex_.get());

  auto res = grouped_gemm_map_.find(cfg);
  if(res == grouped_gemm_map_.end()) {
    TF_ASSIGN_OR_RETURN(auto plan_res, 
            gpu::BlasLt::GetGroupedMatmulPlan(&stream, nullptr, cfg));
    res = grouped_gemm_map_.emplace(cfg, std::move(plan_res)).first;

    TF_ASSIGN_OR_RETURN(auto algos, res->second->GetAlgorithms());
    VLOG(2) << "++++++++++++++++ added new config: " << grouped_gemm_map_.size() << 
        " algos found: " << algos.size();
    if(algos.empty()) {
      return xla::InternalError("No valid algorithms found!");
    }
    res->second->SetAlgorithm(algos[0]);
  }
  return res->second->ExecuteOnStream(&stream, cfg);
}

xla::Status BlasLtGemmRunner::RunStridedBatchedImpl(Stream& stream, 
      blas::Transpose trans_a, blas::Transpose trans_b, int64 m, int64 n, int64 k, 
      xla::complex128 alpha, 
      blas::DataType type_a, const DeviceMemoryBase& a, int64 lda, int64 stride_a,
      blas::DataType type_b, const DeviceMemoryBase& b, int64 ldb, int64 stride_b,
      double beta,
      blas::DataType type_c, DeviceMemoryBase *c, int64 ldc, int64 stride_c, 
      int64 batch_count)
{
  TF_ASSIGN_OR_RETURN(auto compute_type, 
            gpu::GetBlasComputationType(type_a, type_c, 0));

  StridedGemmConfig scfg{
    .m = m,
    .n = n,
    .k = k,
    .batch_count = (int64)batch_count,
    .trans_a = trans_a,
    .trans_b = trans_b,
    .alpha_re = alpha.real(),
    .alpha_im = alpha.imag(),
    .beta = beta,
    .type_a = type_a,
    .type_b = type_b,
    .type_c = type_c,
    .type_d = type_c, 
    .lda = lda,
    .ldb = ldb,
    .ldc = ldc,
    .ldd = ldc,
    .stride_a = stride_a,
    .stride_b = stride_b,
    .stride_c = stride_c,
    .stride_d = stride_c,
    .compute_type = compute_type,
  };

  absl::MutexLock lock(mutex_.get());

  auto res = strided_gemm_map_.find(scfg);
  if(res == strided_gemm_map_.end()) {

    int64 row_a = m, col_a = k, row_b = k, col_b = n;
    if (trans_a == blas::Transpose::kTranspose) std::swap(row_a, col_a);
    if (trans_b == blas::Transpose::kTranspose) std::swap(row_b, col_b);

    auto order = MatrixLayout::Order::kColumnMajor;
    GemmConfig cfg = {
        .lhs_layout = MatrixLayout(type_a, row_a, col_a, order, batch_count,
                                  lda, stride_a, trans_a),

        .rhs_layout = MatrixLayout(type_b, row_b, col_b, order, batch_count,
                                  ldb, stride_b, trans_b),

        .c_layout = MatrixLayout(type_c, m, n, order, batch_count,
                                  ldc, stride_c),
        .output_layout = MatrixLayout(type_c, m, n, order, batch_count,
                                  ldc, stride_c),
        .alpha = alpha,
        .beta = beta,
        .algorithm = {},
        .grad_x = false,
        .grad_y = false,
        .compute_type = compute_type,
    };

    TF_ASSIGN_OR_RETURN(auto plan_res, 
            gpu::BlasLt::GetMatmulPlan(&stream, cfg, gpu::BlasLt::Epilogue::kDefault));
    res = strided_gemm_map_.emplace(scfg, std::move(plan_res)).first;

    TF_ASSIGN_OR_RETURN(auto algos, res->second->GetAlgorithms());
    VLOG(2) << "+++++++++++++ added new config: " << strided_gemm_map_.size() << 
        " algos found: " << algos.size();
    if(algos.empty()) {
      return xla::InternalError("No valid algorithms found!");
    }
    res->second->SetAlgorithm(algos[0]);
  }
  return res->second->ExecuteOnStream(
      &stream, a, b, *c, *c,
      DeviceMemoryBase{}, // bias
      DeviceMemoryBase{}, // aux
      DeviceMemoryBase{}, // a_scale
      DeviceMemoryBase{}, // b_scale
      DeviceMemoryBase{}, // c_scale
      DeviceMemoryBase{}, // d_scale
      DeviceMemoryBase{}, // d_amax
      DeviceMemoryBase{}); // workspace
}


}  // namespace gpu

}  // namespace stream_executor
