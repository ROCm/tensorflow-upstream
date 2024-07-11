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

#include "tensorflow/compiler/xla/service/gpu/gemm_algorithm_picker.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// #include "absl/container/flat_hash_set.h"
// #include "absl/status/status.h"
// #include "absl/status/statusor.h"
// #include "absl/strings/str_cat.h"
// #include "absl/strings/string_view.h"
// #include "absl/synchronization/mutex.h"
// #include "absl/types/span.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/autotuner_compile_util.h"
#include "tensorflow/compiler/xla/service/gpu/autotuner_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
// #include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
// #include "tensorflow/compiler/xla/service/gpu/variant_visitor.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_description.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#include "tensorflow/compiler/xla/util.h"
// #include "tsl/platform/errors.h"
// #include "tsl/platform/logging.h"
// #include "tsl/platform/statusor.h"
// #include "tsl/profiler/lib/scoped_annotation.h"

namespace xla {
namespace gpu {
namespace {

using se::gpu::BlasLt;

StatusOr<BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::GELU_AUX:
      return BlasLt::Epilogue::kGELUWithAux;
    case GemmBackendConfig::BIAS:
      return BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return BlasLt::Epilogue::kBiasThenGELU;
    case GemmBackendConfig::BIAS_GELU_AUX:
      return BlasLt::Epilogue::kBiasThenGELUWithAux;
    default:
      return Internal("Unsupported Epilogue.");
  }
}

class GemmAutotuner {
  const AutotuneConfig& autotune_config_;
  RedzoneBuffers rz_buffers_;
  std::unique_ptr< se::Stream > stream_;
  bool deterministic_ops_ = false;
  size_t solutions_limit_ = 0;
  size_t num_algorithms_left_ = 0;
  float gemm_relative_tol_ = 0.1f;
  std::string gemm_str_; // needed for PickBestResult only

 public:
  explicit GemmAutotuner(const AutotuneConfig& autotune_config)
      : autotune_config_(autotune_config) {}

  size_t num_algorithms_left() const { return num_algorithms_left_; }

  StatusOr<tensorflow::AutotuneResult> operator()(
          const std::string& gemm_str,
          const GemmConfig& gemm_config, GemmBackendConfig::Epilogue epilogue,
          std::vector< Shape >&& input_shapes, const Shape& output_shape,
          const DebugOptions& debug_options) {
  
    num_algorithms_left_ = 0;
    gemm_str_ = gemm_str;
    VLOG(3) << "Starting autotune of GemmThunk " << gemm_str_;

    if(!stream_) {
      stream_ = std::make_unique< se::Stream >(autotune_config_.GetExecutor());
      stream_->Init();
    }

    deterministic_ops_ = false ;//debug_options.xla_gpu_deterministic_ops() ||
                                //debug_options.xla_gpu_exclude_nondeterministic_ops();
    solutions_limit_ = 0; //debug_options.xla_gpu_autotune_max_solutions();
    gemm_relative_tol_ = debug_options.xla_gpu_autotune_gemm_rtol();

    // Don't run autotuning concurrently on the same GPU.
    tensorflow::mutex_lock gpu_lock = LockGpu(stream_->parent());

    TF_ASSIGN_OR_RETURN(rz_buffers_, RedzoneBuffers::FromShapes(
         std::move(input_shapes), output_shape, autotune_config_, stream_.get(), 
          debug_options, RedzoneBuffers::kAllInputsAllOutputs));
    
    return TuneGpuBlasLt(output_shape, gemm_config, epilogue);
  }

  StatusOr<tensorflow::AutotuneResult> operator()(const HloInstruction* gemm) {

    num_algorithms_left_ = 0;
    gemm_str_ = gemm->ToString();
    VLOG(3) << "Starting autotune of GemmThunk " << gemm_str_;

    if(!stream_) {
      stream_ = std::make_unique< se::Stream >(autotune_config_.GetExecutor());
      stream_->Init();
    }

    const DebugOptions& debug_options =
        gemm->GetModule()->config().debug_options();
    deterministic_ops_ = false ;//debug_options.xla_gpu_deterministic_ops() ||
                                //debug_options.xla_gpu_exclude_nondeterministic_ops();
    solutions_limit_ = 0; //debug_options.xla_gpu_autotune_max_solutions();
    gemm_relative_tol_ = debug_options.xla_gpu_autotune_gemm_rtol();

    GemmBackendConfig backend_config = gemm->
              backend_config<GemmBackendConfig>().ValueOrDie();
    TF_ASSIGN_OR_RETURN(auto gemm_config, GemmConfig::For(gemm, backend_config));

    // Don't run autotuning concurrently on the same GPU.
    tensorflow::mutex_lock gpu_lock = LockGpu(stream_->parent());

    TF_ASSIGN_OR_RETURN(rz_buffers_, RedzoneBuffers::FromInstruction(
                        *gemm, autotune_config_, stream_.get(), debug_options,
                        RedzoneBuffers::kAllInputsAllOutputs));
    
    return IsCublasLtMatmul(*gemm)
           ? TuneGpuBlasLt(gemm->shape(), gemm_config, backend_config.epilogue())
           : TuneGpuBlas(gemm->shape(), gemm_config);
  }

 private:
  se::DeviceMemoryBase LhsBuffer() { return rz_buffers_.input_buffers().at(0); }
  se::DeviceMemoryBase RhsBuffer() { return rz_buffers_.input_buffers().at(1); }
  se::DeviceMemoryBase OutputBuffer() {
    return rz_buffers_.output_buffers().at(0);
  }

  StatusOr<tensorflow::AutotuneResult> TuneGpuBlasLt(const Shape& out_shape,
         const GemmConfig& gemm_config, GemmBackendConfig::Epilogue epilogue) {
    
    auto workspace_buffer = 
      rz_buffers_.output_buffers().at(out_shape.tuple_shapes_size() - 1);

    bool has_matrix_bias = gemm_config.beta != 0.;

    TF_ASSIGN_OR_RETURN(
        bool has_vector_bias,
        gpublas_lt::EpilogueAddsVectorBias(epilogue));

    TF_ASSIGN_OR_RETURN(
        bool has_aux_output,
        gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));

    TF_ASSIGN_OR_RETURN(auto blas_epilogue,
                        AsBlasLtEpilogue(epilogue));

    se::DeviceMemoryBase a_scale_buffer, b_scale_buffer, c_scale_buffer,
        d_scale_buffer, d_amax_buffer, bias_buffer, aux_buffer;

    if (has_vector_bias) {
      bias_buffer = rz_buffers_.input_buffers().at(has_matrix_bias ? 3 : 2);
    }
    if (has_aux_output) {
      aux_buffer = rz_buffers_.output_buffers().at(1);
    }

    TF_ASSIGN_OR_RETURN(auto plan,
              BlasLt::GetMatmulPlan(stream_.get(), gemm_config, blas_epilogue));

    TF_ASSIGN_OR_RETURN(
        auto algorithms,
        plan->GetAlgorithms(/*max_algorithm_count*/ GemmConfig::kMaxCublasLtAlgorithms,
                             /*max_workspace_size*/ workspace_buffer.size()));

    auto tuned_func = [&](const BlasLt::MatmulAlgorithm& algorithm)
        -> StatusOr<se::blas::ProfileResult> {
      plan->SetAlgorithm(algorithm);
      // Run a warmup iteration without the profiler active.
      TF_RETURN_IF_ERROR(plan->ExecuteOnStream(
          stream_.get(), LhsBuffer(), RhsBuffer(), OutputBuffer(), OutputBuffer(),
          bias_buffer, aux_buffer, a_scale_buffer, b_scale_buffer,
          c_scale_buffer, d_scale_buffer, d_amax_buffer, 
          workspace_buffer));

      se::blas::ProfileResult profile_result;
      TF_RETURN_IF_ERROR(plan->ExecuteOnStream(
          stream_.get(), LhsBuffer(), RhsBuffer(), OutputBuffer(), OutputBuffer(),
          bias_buffer, aux_buffer, a_scale_buffer, b_scale_buffer,
          c_scale_buffer, d_scale_buffer, d_amax_buffer, 
          workspace_buffer, absl::nullopt, &profile_result));
      return std::move(profile_result);
    };

    const auto& shape = out_shape.IsTuple() ? out_shape.tuple_shapes(0) 
                                            : out_shape;
    return GetBestAlgorithm<BlasLt::MatmulAlgorithm>(
         shape, algorithms, gemm_config.beta, true, tuned_func);
  }

  StatusOr<tensorflow::AutotuneResult> TuneGpuBlas(const Shape& out_shape,
                                             const GemmConfig& gemm_config) {
#if 0
    auto workspace_buffer = rz_buffers_.output_buffers().at(1);

    std::vector<se::blas::AlgorithmType> algorithms;
    TF_ASSIGN_OR_RETURN(GemmConfig::DescriptorsTuple desc,
                        gemm_config.GetMatrixDescriptors(
                            LhsBuffer(), RhsBuffer(), OutputBuffer()));

    auto blas = stream_->parent()->AsBlas();
    if (blas == nullptr) {
      return xla::InternalError("No BLAS support for stream");
    }
    blas->GetBlasGemmAlgorithms(stream_.get(), desc.lhs, desc.rhs, &desc.output,
                                &gemm_config.alpha, &gemm_config.beta,
                                &algorithms);

    auto tuned_func = [&](const se::blas::AlgorithmType& algorithm)
        -> StatusOr<se::blas::ProfileResult> {
      // Do a warm-up run first, without a profile result. RunGemm swallows
      // error codes when profile_result is passed, as it is in the measurement
      // below, but not otherwise. It is, therefore, consistent to ignore the
      // error code here.
      static_cast<void>(RunGemm(gemm_config, LhsBuffer(), RhsBuffer(),
                                OutputBuffer(), workspace_buffer,
                                deterministic_ops_, stream_.get(), algorithm));
      se::blas::ProfileResult profile_result;
      // Allow GpuTimer to use its delay kernel implementation to improve
      // accuracy.
      profile_result.set_warmup_run_executed(true);
      // We expect GemmWithAlgorithm to fail sometimes -- in fact, it will fail
      // for all algorithms if we're targeting < sm_50. But because we pass a
      // non-null ProfileResult, DoGemmWithAlgorithm should always return true,
      // and the actual success-ness is returned in ProfileResult::is_valid.
      TF_RETURN_IF_ERROR(RunGemm(gemm_config, LhsBuffer(), RhsBuffer(),
                                 OutputBuffer(), workspace_buffer,
                                 deterministic_ops_, stream_.get(), algorithm,
                                 &profile_result));
      return std::move(profile_result);
    };

    const auto& shape = out_shape.IsTuple() ? out_shape.tuple_shapes(0) 
                                            : out_shape;
    return GetBestAlgorithm<se::blas::AlgorithmType>(
         shape, algorithms, gemm_config.beta, false, tuned_func);
#else
  return tensorflow::AutotuneResult{};
#endif
  }

  // Returns the index (into `algorithms`) of the fastest algorithm.
  template <typename AlgoT, typename TunedFunc>
  StatusOr<tensorflow::AutotuneResult> GetBestAlgorithm(
      const Shape& output_shape, absl::Span<const AlgoT> algorithms,
      double beta, bool return_algo_index, TunedFunc&& run_benchmark) {
    // static_assert(std::is_invocable_r_v<StatusOr<se::blas::ProfileResult>,
    //                                     TunedFunc, const AlgoT&>,
    //               "Tuned function has incorrect prototype!");

    if (!stream_->parent()->SynchronizeAllActivity()) {
      return Internal("Failed to synchronize GPU for autotuning.");
    }

    se::DeviceMemoryBase reference_buffer;
    if (autotune_config_.should_check_correctness()) {
      TF_ASSIGN_OR_RETURN(reference_buffer,
                          rz_buffers_.RedzoneAllocator().AllocateBytes(
                              ShapeUtil::ByteSizeOf(output_shape)));
    }

    // Do not print error messages if should_skip_wrong_results() is ON.
    BufferComparator comparator(output_shape, gemm_relative_tol_,
        /* verbose */!autotune_config_.should_skip_wrong_results()
    );
    std::vector<tensorflow::AutotuneResult> results;
    results.reserve(algorithms.size());
    absl::optional<int64_t> reference_algorithm;

    auto num = algorithms.size();
    if (solutions_limit_ > 0) num = std::min(num, solutions_limit_);
    for (size_t i = 0; i < num; i++) {
      const AlgoT& algorithm = algorithms[i];
      // Make sure the output buffer always has the same value if we use
      // the bias parameter.
      if (autotune_config_.should_reinit_output_buffer() && beta != 0) {
        int64 rng_state = 0;
        InitializeBuffer(stream_.get(), output_shape.element_type(), &rng_state,
                         OutputBuffer());
      }
      TF_ASSIGN_OR_RETURN(auto profile_result, run_benchmark(algorithm));

      results.emplace_back();
      tensorflow::AutotuneResult& result = results.back();
      result.mutable_gemm()->set_algorithm(profile_result.algorithm());

      if (!profile_result.is_valid()) {  // Unsupported algorithm.
        result.mutable_failure()->set_kind(tensorflow::AutotuneResult::DISQUALIFIED);
        continue;
      }

      VLOG(2) << "gemm algorithm " << profile_result.algorithm() << " took "
              << profile_result.elapsed_time_in_ms() << "ms";

      *result.mutable_run_time() = tensorflow::proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));

      if (!autotune_config_.should_check_correctness()) {
        num_algorithms_left_++;
        continue;
      }
      TF_ASSIGN_OR_RETURN(
          se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
          rz_buffers_.RedzoneAllocator().CheckRedzones());

      if (!rz_check_status.ok()) {
        result.mutable_failure()->set_kind(tensorflow::AutotuneResult::REDZONE_MODIFIED);
        *result.mutable_failure()->mutable_msg() =
            rz_check_status.RedzoneFailureMsg();
        LOG(ERROR) << "Detected out-of-bounds write in gemm buffer";
        CHECK(!autotune_config_.should_crash_on_check_failure());
        continue;
      }

      num_algorithms_left_++; 
      if (!reference_algorithm) {
        stream_->ThenMemcpy(&reference_buffer, OutputBuffer(),
                                           OutputBuffer().size());
        reference_algorithm = profile_result.algorithm();
        continue;
      } 
      // Perform the comparison versus the reference algorithm.
      TF_ASSIGN_OR_RETURN(
          bool outputs_match,
          comparator.CompareEqual(stream_.get(), /*current=*/OutputBuffer(),
                                    /*expected=*/reference_buffer));
      if (!outputs_match) {
        LOG(ERROR) << "Results mismatch between different GEMM algorithms. "
                   << "This is likely a bug/unexpected loss of precision.";
        CHECK(!autotune_config_.should_crash_on_check_failure());

        // By default, autotuner does NOT really skip wrong results, but 
        // merely prints out the above error message: this may lead to a 
        // great confusion. When should_skip_wrong_results() is set to true,
        // solutions with accuracy problems will be disqualified.
        auto kind = tensorflow::AutotuneResult::WRONG_RESULT;
        if (autotune_config_.should_skip_wrong_results()) {
          kind = tensorflow::AutotuneResult::DISQUALIFIED;
          num_algorithms_left_--; // Decrement again since we disqualified it.
        }
        result.mutable_failure()->set_kind(kind);
        result.mutable_failure()->mutable_reference_gemm()->set_algorithm(
              *reference_algorithm);
      }
    }  // for algorithms

    StatusOr<tensorflow::AutotuneResult> best_res =
        PickBestResult(results, gemm_str_);
    if (best_res.ok()) {
      auto best = std::move(best_res.ValueOrDie());
      // Return a real algorithm ID if return_algo_index is false: 
      // e.g., in case of legacy cublas tuning.
      if (!return_algo_index) return best; 
      // Otherwise, map a real algorithm ID to its index among the results.
      for (size_t i = 0; i < results.size(); ++i) {
        if (best.gemm().algorithm() == results[i].gemm().algorithm()) {
          best.mutable_gemm()->set_algorithm(i);
          return best;
        }
      }
      return Internal("unknown best algorithm");
    }
    LOG(WARNING) << "Failed to find best cuBLAS algorithm, GEMM performance "
                    "might be suboptimal: "
                 << best_res.status();
    return tensorflow::AutotuneResult{};
  }  // GetBestAlgorithm
};  // GemmAutotuner

// Do Gemm Autotune without stream executor. Use results from autotune cache
// only.
StatusOr<bool> RunOnInstruction(HloInstruction* gemm,
                                const AutotuneConfig& config,
                                size_t *num_algorithms_left) {
  VLOG(3) << "Loading the autotune result of GemmThunk " << gemm->ToString();
  GemmBackendConfig backend_config = gemm->
              backend_config<GemmBackendConfig>().ValueOrDie();

  *num_algorithms_left = 0;
  // Degenerate gemms replaced with memzero operation, no need to auto tune it.
  if (backend_config.alpha_real() == 0.0 &&
      backend_config.alpha_imag() == 0.0 && backend_config.beta() == 0.0) {
    VLOG(3) << "Skip degenerate gemm instruction auto tuning";
    return false;
  }

  GemmAutotuner autotuner(config);
  TF_ASSIGN_OR_RETURN(tensorflow::AutotuneResult algorithm,
       AutotunerUtil::Autotune(AutotunerUtil::GetKey(gemm, config), 
                    config, [&] { return autotuner(gemm); }));

  *num_algorithms_left = autotuner.num_algorithms_left();
  auto old_algorithm = backend_config.selected_algorithm();
  bool update_algorithm = true;
      // std::visit(VariantVisitor{[](const se::CudaComputeCapability& cc) {
      //                             // We only set the 'algorithm' field on
      //                             // non-Ampere architectures, as for Ampere
      //                             // it's ignored in any case.
      //                             return !cc.IsAtLeast(
      //                                 se::CudaComputeCapability::AMPERE);
      //                           },
      //                           [](const se::RocmComputeCapability&) {
      //                             return true;  // TODO: not decided yet
      //                           }},
      //            config.GetGpuComputeCapability());

  if (update_algorithm) {
    int64_t new_algorithm{};
    if (algorithm.has_gemm()) {
      new_algorithm = algorithm.gemm().algorithm();
    } else {
      // NOTE: runtime autotuning is no longer available => set to default
      new_algorithm = se::blas::kDefaultAlgorithm;
    }

    if (new_algorithm == old_algorithm) {
      // We don't need to update the backend config if
      // the algorithm hasn't changed unless previously
      // the algorithm wasn't set explicitly.
      return false;
    }

    backend_config.set_selected_algorithm(new_algorithm);
    TF_RETURN_IF_ERROR(gemm->set_backend_config(backend_config));
    return true;  // We changed `gemm`
  }

  return false;  // No change to `gemm`
}

StatusOr<bool> RunOnComputation(HloComputation* computation,
               AutotuneConfig config, size_t *num_algorithms_left) {
  bool changed = false;

  for (HloInstruction* instr : computation->instructions()) {
    if (IsCublasGemm(*instr)) {
      size_t num_left;
      TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(instr, config, &num_left));
      // Gathering statistics on the algorithms left after tuning (for testing)
      *num_algorithms_left = std::max(*num_algorithms_left, num_left);
      changed |= result;
    }
  }
  return changed;
}

}  // namespace

StatusOr<tensorflow::AutotuneResult> GemmAlgorithmPicker::RunStandalone(
     const std::string& gemm_canonical_str, const GemmConfig& gemm_config, 
     GemmBackendConfig::Epilogue epilogue,
     std::vector< Shape >&& input_shapes, const Shape& output_shape,
     const DebugOptions& debug_options) {

  GemmAutotuner autotuner(config_);
  return AutotunerUtil::Autotune(
     AutotuneCacheKey(config_.GetModelStr(), gemm_canonical_str),
     config_, [&]{ 
            return autotuner(gemm_canonical_str, gemm_config, 
                epilogue, std::move(input_shapes), output_shape, debug_options); 
  });
}

StatusOr<bool> GemmAlgorithmPicker::Run(
    HloModule* module) {
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrCat("GemmAlgorithmPicker for ", module->name()));

  num_algorithms_left_ = 0;
  if (module->config().debug_options().xla_gpu_autotune_level() == 0) {
    VLOG(2) << "GEMM auto-tuning disabled, GemmAlgorithmPicker returning early";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, 
                RunOnComputation(computation, config_, &num_algorithms_left_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
