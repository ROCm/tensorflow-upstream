/* Copyright 2022 The OpenXLA Authors.

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

#include <utility>
#include "tensorflow/compiler/xla/service/gpu/gpublas_lt_matmul_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/autotuner_util.h"

namespace xla {
namespace gpu {

struct MatmulPlanCache {

  static MatmulPlanCache& i(const se::Stream *stream) {
    static absl::Mutex m(absl::kConstInit);
    // Each GPU gets different cache instance
    static absl::flat_hash_map<void *, MatmulPlanCache> meta;
    absl::MutexLock lock(&m);
    auto res = meta.find(stream->parent());
    if(res != meta.end()) return res->second;
    MatmulPlanCache r;
    return meta.emplace(stream->parent(), std::move(r)).first->second;
  }

  template < class Func >
  StatusOr<se::gpu::BlasLt::MatmulPlan *> 
          GetOrCreate(const std::string& key, Func&& create) {
    // each GPU has a different mutex => hence different GPU instances can
    // create matmul plans in parallel
    absl::MutexLock lock(mutex_.get()); 
    auto res = map_.emplace(key, se::gpu::BlasLt::MatmulPlanPtr{});
    if(res.second) { // new entry inserted
      TF_ASSIGN_OR_RETURN(res.first->second, create());
    } 
    return res.first->second.get();
  }

private:
  MatmulPlanCache() : mutex_(std::make_unique< absl::Mutex >()) { }

private:
  std::unique_ptr< absl::Mutex > mutex_;
  absl::flat_hash_map<std::string, se::gpu::BlasLt::MatmulPlanPtr> map_;
};

CublasLtMatmulThunk::CublasLtMatmulThunk(
    const HloInstruction *hlo_instruction,
    GemmBackendConfig&& backend_config,
    se::gpu::BlasLt::Epilogue epilogue, int64_t algorithm_idx,
    BufferAllocation::Slice a_buffer, BufferAllocation::Slice b_buffer,
    BufferAllocation::Slice c_buffer, BufferAllocation::Slice d_buffer,
    BufferAllocation::Slice bias_buffer, BufferAllocation::Slice aux_buffer,
    BufferAllocation::Slice a_scale, BufferAllocation::Slice b_scale,
    BufferAllocation::Slice c_scale, BufferAllocation::Slice d_scale,
    BufferAllocation::Slice d_amax,
    absl::optional<const BufferAllocation::Slice> workspace_buffer)
    : Thunk(Kind::kCublasLtMatmul, hlo_instruction),
      backend_config_(std::move(backend_config)),
      epilogue_(epilogue),
      algorithm_idx_(algorithm_idx),
      canonical_hlo_(AutotuneCacheKey("none", *hlo_instruction).GetHlo()),
      a_buffer_(a_buffer),
      b_buffer_(b_buffer),
      c_buffer_(c_buffer),
      d_buffer_(d_buffer),
      bias_buffer_(bias_buffer),
      aux_buffer_(aux_buffer),
      a_scale_buffer_(a_scale),
      b_scale_buffer_(b_scale),
      c_scale_buffer_(c_scale),
      d_scale_buffer_(d_scale),
      d_amax_buffer_(d_amax),
      workspace_buffer_(workspace_buffer) {}

Status CublasLtMatmulThunk::Initialize(const GpuExecutable& executable,
                            se::StreamExecutor* executor) {
  if (!executor->AsBlas()) {
    return InternalError("Failed to initialize BLASLT support");
  }
  return Status::OK();
}

Status CublasLtMatmulThunk::ExecuteOnStream(const ExecuteParams& params) {

  TF_ASSIGN_OR_RETURN(auto *plan, GetCachedMatmulPlan(params.stream));

  VLOG(3) << "Running cublas_lt matmul thunk for instr: " << hlo_instruction()->ToString();
  const BufferAllocations& allocs = *params.buffer_allocations;

  se::DeviceMemoryBase bias, a_scale, b_scale, c_scale, d_scale, d_amax;
  if (bias_buffer_.allocation() != nullptr) {
    bias = allocs.GetDeviceAddress(bias_buffer_);
  }
  if (a_scale_buffer_.allocation() != nullptr) {
    a_scale = allocs.GetDeviceAddress(a_scale_buffer_);
  }
  if (b_scale_buffer_.allocation() != nullptr) {
    b_scale = allocs.GetDeviceAddress(b_scale_buffer_);
  }
  if (c_scale_buffer_.allocation() != nullptr) {
    c_scale = allocs.GetDeviceAddress(c_scale_buffer_);
  }
  if (d_scale_buffer_.allocation() != nullptr) {
    d_scale = allocs.GetDeviceAddress(d_scale_buffer_);
  }
  if (d_amax_buffer_.allocation() != nullptr) {
    d_amax = allocs.GetDeviceAddress(d_amax_buffer_);
  }

  se::DeviceMemoryBase aux;
  if (aux_buffer_.allocation() != nullptr) {
    aux = allocs.GetDeviceAddress(aux_buffer_);
  }

  absl::optional<se::DeviceMemoryBase> workspace;
  if (workspace_buffer_.has_value()) {
    workspace = allocs.GetDeviceAddress(workspace_buffer_.value());
  }

  return plan->ExecuteOnStream(
      params.stream, allocs.GetDeviceAddress(a_buffer_),
      allocs.GetDeviceAddress(b_buffer_), allocs.GetDeviceAddress(c_buffer_),
      allocs.GetDeviceAddress(d_buffer_), bias, aux, a_scale, b_scale, c_scale,
      d_scale, d_amax, workspace);
}

auto CublasLtMatmulThunk::GetCachedMatmulPlan(
    const se::Stream* stream) -> StatusOr<se::gpu::BlasLt::MatmulPlan *> {

  auto& cache = MatmulPlanCache::i(stream);

  auto create = [&]() ->  StatusOr<se::gpu::BlasLt::MatmulPlanPtr>  {
    VLOG(2) << this << ": Adding new MatmulPlan for stream: " << stream << 
                       " instr: " << hlo_instruction()->ToString();
    TF_ASSIGN_OR_RETURN(
       auto gemm_config,
       GemmConfig::For(hlo_instruction(), backend_config_));
    
    TF_ASSIGN_OR_RETURN(auto plan, se::gpu::BlasLt::GetMatmulPlan(
                                  stream, gemm_config, epilogue_));
    
    int64_t max_workspace = workspace_buffer_.has_value()
                          ? workspace_buffer_.value().size() : 0;
    TF_ASSIGN_OR_RETURN(auto algorithms,
       plan->GetAlgorithms(/*max_algorithm_count*/GemmConfig::kMaxCublasLtAlgorithms,
                           /*max_workspace_size*/ max_workspace));
    TF_RET_CHECK(algorithm_idx_ >= 0 && algorithm_idx_ < algorithms.size());
    plan->SetAlgorithm(algorithms[algorithm_idx_]);

    return std::move(plan);
  };
  return cache.GetOrCreate(canonical_hlo_, create);
}

}  // namespace gpu
}  // namespace xla
