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

#include "tensorflow/tsl/protobuf/autotuning.pb.h"
//#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
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
  explicit AutotuneCacheKey(const std::string& s) : key_(s) { }

  absl::string_view Get() const { return key_; }

  template <typename H>
  friend H AbslHashValue(H h, const AutotuneCacheKey& w) {
    return H::combine(std::move(h), w.key_);
  }

  bool operator==(const AutotuneCacheKey& w) const {
    return key_ == w.key_;
  }

 private:
  std::string key_;
};

using AutotuneConvCacheKey =
    std::tuple<std::string /* stream_exec->GetDeviceDescription().model_str()*/,
               std::string /* instr->ToString(HloPrintOptions::Canonical()) */>;

// using AutotuneConvCacheMap =
//     absl::flat_hash_map<AutotuneConvCacheKey, tensorflow::AutotuneResult>;


class AutotuneConfig {
 public:
  bool should_init_buffers() const { return autotune_level_ >= 2; }
  bool should_reinit_output_buffer() const { return autotune_level_ >= 3; }
  bool should_check_correctness() const { return autotune_level_ >= 4; }
  bool should_skip_wrong_results() const { return autotune_level_ >= 5; }
  bool should_crash_on_check_failure() const {
    return should_crash_on_check_failure_;
  }
  
  absl::string_view dump_path() const { return dump_path_; }
  absl::string_view load_path() const { return load_path_; }

  AutotuneConfig(const AutotuneConfig& right)
      : config_(right.config_),
        autotune_level_(right.autotune_level_),
        should_crash_on_check_failure_(right.should_crash_on_check_failure_),
        dump_path_(right.dump_path_),
        load_path_(right.load_path_)
         {}

  AutotuneConfig(const DeviceConfig& config,
                 const DebugOptions& debug_options)
      : config_(config),
        autotune_level_(debug_options.xla_gpu_autotune_level()),
        should_crash_on_check_failure_(
            debug_options.xla_gpu_crash_on_verification_failures()),
        dump_path_(debug_options.xla_gpu_dump_autotune_results_to()),
        load_path_(debug_options.xla_gpu_load_autotune_results_from())
   {}

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

 private:
  DeviceConfig config_;
  int32_t autotune_level_;
  bool should_crash_on_check_failure_;
  std::string dump_path_, load_path_;
  mutable std::unique_ptr<se::DeviceMemoryAllocator> allocator_;
};

struct AutotunerUtil {

  using CacheValue = int64_t; // algorithm ID
  using AutotuneNoCacheFn = std::function<StatusOr<CacheValue>()>;

  // Create a buffer for a given operation using redzone checker, initialize
  // based on a given rng state.
  static StatusOr<se::DeviceMemoryBase> CreateBuffer(
      se::RedzoneAllocator& allocator, const Shape& shape,
      const AutotuneConfig& config, int64_t& rng_state);

  static StatusOr<CacheValue> Autotune(
    const std::string& gemm_config, const AutotuneConfig& config,
    const AutotuneNoCacheFn& autotune_fn);

  static absl::optional<CacheValue> TryToFindInInMemoryCache(
    const AutotuneCacheKey& key);

  static const CacheValue& AddResultToInMemoryCache(
          const AutotuneCacheKey& key, CacheValue result,
          const AutotuneConfig& cfg);

  // Creates a RedzoneAllocator from a given config.
  static StatusOr<se::RedzoneAllocator> CreateRedzoneAllocator(
      const AutotuneConfig& config, se::Stream *stream, 
      const DebugOptions& opts);

  // Loads autotune results from a file.
  //
  // Warning: The results are only loaded to the in-memory cache.
  static Status LoadAutotuneResultsFromFile(const AutotuneConfig& config);
  // Same as above but do it only once!
  static Status LoadAutotuneResultsFromFileOnce(const AutotuneConfig& config);

  // Warning: This only clears the in-memory cache. If you use a file based
  // cache you're responsible for clearing the cache directory when you want to.
  static void ClearAutotuneResults();

  static AutotuneConvCacheKey ConvCacheKeyFromInstruction(
    const HloInstruction* instr, absl::string_view model_str);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_AUTOTUNER_UTIL_H_

