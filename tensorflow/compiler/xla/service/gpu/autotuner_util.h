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
#ifndef XLA_SERVICE_GPU_AUTOTUNER_UTIL_H_
#define XLA_SERVICE_GPU_AUTOTUNER_UTIL_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/types/variant.h"

#include "tensorflow/core/protobuf/autotune_results.pb.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/stream_executor/device_description.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {
namespace gpu {

struct DeviceConfig {
  se::StreamExecutor* stream_exec;  // never null

  // If the `allocator` parameter is not null, we will use it to allocate temp
  // memory while timing the various convolution algorithms.  If it's null,
  // we'll use the default allocator on the StreamExecutor.
  se::DeviceMemoryAllocator* allocator = nullptr;  // may be null
};

class AutotuneCacheKey {
 public:
  AutotuneCacheKey(absl::string_view model_str,
                   const HloInstruction& instruction);

  explicit AutotuneCacheKey(absl::string_view model_str,
                            absl::string_view hlo_canonical)
      : model_str_(model_str), hlo_canonical_(hlo_canonical) {}

  absl::string_view GetModelStr() const { return model_str_; }

  absl::string_view GetHlo() const { return hlo_canonical_; }

  template <typename H>
  friend H AbslHashValue(H h, const AutotuneCacheKey& w) {
    return H::combine(std::move(h), w.model_str_, w.hlo_canonical_);
  }

  bool operator==(const AutotuneCacheKey& w) const {
    return model_str_ == w.model_str_ && hlo_canonical_ == w.hlo_canonical_;
  }

  std::string ToString() const {
    return absl::StrFormat("<key model='%s', hlo='%s'>", model_str_,
                           hlo_canonical_);
  }

 private:
  std::string model_str_;
  std::string hlo_canonical_;
};

class AutotuneConfig {
 public:
  bool should_init_buffers() const { return autotune_level_ >= 2; }
  bool should_reinit_output_buffer() const { return autotune_level_ >= 3; }
  bool should_check_correctness() const { return autotune_level_ >= 4; }
  bool should_skip_wrong_results() const { return autotune_level_ >= 5; }
  bool should_crash_on_check_failure() const {
    return should_crash_on_check_failure_;
  }

  AutotuneConfig(const AutotuneConfig& right)
      : config_(right.config_),
        autotune_level_(right.autotune_level_),
        should_crash_on_check_failure_(right.should_crash_on_check_failure_)
         {}

  AutotuneConfig(const DeviceConfig& config,
                 const DebugOptions& debug_options)
      : config_(config),
        autotune_level_(debug_options.xla_gpu_autotune_level()),
        should_crash_on_check_failure_(
            debug_options.xla_gpu_crash_on_verification_failures())
   {}

   std::string autotune_cache_dir() const { return ""; }

  absl::string_view GetModelStr() const {
    // NOTE: model_str is not available !
    return GetExecutor()->GetDeviceDescription().name();
  }

  se::StreamExecutor* GetExecutor() const {
    CHECK(config_.stream_exec != nullptr);
    return config_.stream_exec;
  }

  se::DeviceMemoryAllocator* GetAllocator() const {
    if (config_.allocator != nullptr) {
      return config_.allocator;
    }
    if (allocator_ == nullptr) {
      allocator_ =
          std::make_unique<se::StreamExecutorMemoryAllocator>(GetExecutor());
    }
    return allocator_.get();
  }

  // const se::GpuComputeCapability& GetGpuComputeCapability() const {
  //   if (auto c = absl::get_if<DeviceConfig>(&config_)) {
  //     return c->stream_exec->GetDeviceDescription().gpu_compute_capability();
  //   }
  //   return absl::get<DevicelessConfig>(config_).gpu_compute_capability;
  // }

 private:
  DeviceConfig config_;
  int32_t autotune_level_;
  bool should_crash_on_check_failure_;
  mutable std::unique_ptr<se::DeviceMemoryAllocator> allocator_;
};

using AutotuneNoCacheFn = std::function<StatusOr<tensorflow::AutotuneResult>()>;

struct AutotunerUtil {
  // Create a buffer for a given operation using redzone checker, initialize
  // based on a given rng state.
  static StatusOr<se::DeviceMemoryBase> CreateBuffer(
      se::RedzoneAllocator& allocator, const Shape& shape,
      const AutotuneConfig& config, int64& rng_state);

  static StatusOr<tensorflow::AutotuneResult> Autotune(
    const AutotuneCacheKey& key, const AutotuneConfig& config,
    const AutotuneNoCacheFn& autotune_fn);

  // Returns the same cache key that would be used inside Autotune().
  //
  // Normally, we don't have to use this low level method.
  static AutotuneCacheKey GetKey(const HloInstruction* instr,
                                 const AutotuneConfig& config);

  // Checks if the key is in the autotune cache.
  //
  // Normally, we don't have to use this low level method.
  static StatusOr<bool> IsInCache(const AutotuneCacheKey& key,
                                        const AutotuneConfig& config);

  // Adds the result to the autotune cache.
  //
  // Returns true if the entry is inserted.
  //
  // Normally, we don't have to use this low level method.
  static StatusOr<bool> AddResult(const AutotuneCacheKey& key,
                                        tensorflow::AutotuneResult result,
                                        const AutotuneConfig& config);

  // Creates a RedzoneAllocator from a given config.
  static StatusOr<se::RedzoneAllocator> CreateRedzoneAllocator(
      const AutotuneConfig& config, se::Stream *stream, 
      const DebugOptions& opts);

  // Functions to save/load XLA's autotuning results.
  //
  // This is used for ahead-of-time autotuning.  Specifically:
  //
  // When XLA calls cublas (for matmuls, aka "gemm" or "dot") or cudnn (for
  // convolutions), it usually has to choose an "algorithm" for the particular
  // dot/conv.  XLA queries cublas/cudnn for a list of candidate algorithms.
  // Then it runs all of them and picks the fastest one.  This is what we call
  // "autotuning". It happens in GemmAlgorithmPicker and GpuConvAlgorithmPicker.
  //
  // Autotuning is necessary to get good performance for dot/conv.  But it also
  // has some disadvantages.
  //
  //  - Because it relies on timing data, it is fundamentally nondeterministic.
  //    But even if two algorithms have similar runtimes, our choice of
  //    algorithm may be visible to the user: Different algorithms can have
  //    different numerics, and sometimes they can even have different bugs!
  //
  //  - Trying all the candidate algorithms can be slow, especially if when some
  //    of the candidates are "very bad" and run especially slowly compared to
  //    the optimal candidate.  This slows down compilation.
  //
  // To address the disadvantages above, we allow users to save/restore the
  // autotuning choices that XLA has made, using the functions below.
  //
  // Loading autotuning results does not erase existing autotuning choices, but
  // in the event of a disagreement between the existing data and the new data,
  // the new algorithm is chosen.
  //
  // Note that even if you call LoadAutotuneResults(), if XLA encounters a
  // dot/conv that is *not* covered by the loaded data, it will go ahead and
  // autotune it like normal.  In other words, the behavior of XLA should be
  // identical with or without ahead-of-time autotuning, modulo nondeterminism.
  //
  // This is important if you want to be able to use the same autotuning file
  // with different versions of XLA, because as XLA changes, exactly which
  // dots/convs it wants to run can also change.  For example, XLA might change
  // the conv padding heuristics it uses, and we don't want that to mean that
  // all users of ahead-of-time autotuning are broken.
  static StatusOr<std::string> SerializeAutotuneResults(
      bool as_textproto = false);

  // Serializes autotune results into the given proto.
  static Status SerializeAutotuneResults(tensorflow::AutotuneResults* results);

  // Loads autotune results from the given string of bytes.
  //
  // Warning: The results are only loaded to the in-memory cache.
  static Status LoadAutotuneResults(absl::string_view data,
                                          bool as_textproto = false);

  // Loads autotune results from the given proto.
  //
  // Warning: The results are only loaded to the in-memory cache.
  static Status LoadAutotuneResults(const tensorflow::AutotuneResults& results);

  // Serializes autotune results into a file.
  //
  // If `file_path` ends with ".txt" or ".textproto", then the textproto format
  // is used, otherwise the binary protobuf format.
  static Status SerializeAutotuneResultsToFile(
      absl::string_view file_path);

  // As above, but if you already called SerializeAutotuneResults to get a
  // proto.
  static Status SerializeAutotuneResultsToFile(
      const tensorflow::AutotuneResults& results, absl::string_view file_path);

  // Loads autotune results from a file.
  //
  // If `file_path` ends with ".txt" or ".textproto", then the file is
  // considered to be in the textproto format, otherwise the binary protobuf
  // format.
  //
  // Warning: The results are only loaded to the in-memory cache.
  static Status LoadAutotuneResultsFromFile(absl::string_view file_path);

  // Warning: This only clears the in-memory cache. If you use a file based
  // cache you're responsible for clearing the cache directory when you want to.
  static void ClearAutotuneResults();

  // Warning: This only checks the in-memory cache. If you use a file based
  // cache, you're responsible for checking whether the cache directory is
  // empty.
  static bool ResultCacheIsEmpty();
};

StatusOr<std::string> AutotuneResultsToString(
    const tensorflow::AutotuneResults& results, bool as_textproto);

// Exposed only for testing. Returns the SHA-256 hash of the input string,
// encoded in base64.
//
// SHA-256 was chosen to follow industry best practices and avoid collisions.
// Git is also transitioning to SHA-256. This is probably better than
// Fingerprint128.
StatusOr<std::string> GetBase64EncodedSha256Hash(absl::string_view s);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_AUTOTUNER_UTIL_H_
