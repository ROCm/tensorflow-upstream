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

#include "tensorflow/compiler/xla/service/gpu/autotuner_util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <fstream>

#include "absl/base/call_once.h"
//#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/core/lib/strings/base64.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep


#define kCsvSep ','
#define kCsvComment '#'


namespace xla {
namespace gpu {

using AutotuneCacheMap = absl::flat_hash_map<AutotuneCacheKey, AutotunerUtil::CacheValue>;

static absl::Mutex autotune_cache_mu(absl::kConstInit);
static auto& autotune_cache ABSL_GUARDED_BY(autotune_cache_mu) =
    *new AutotuneCacheMap();

namespace {

void CSVLegend(std::ostream& os) {
  
  os << kCsvComment << " m" << kCsvSep << "n" << kCsvSep << "k" << kCsvSep
        << "batch_count" << kCsvSep << "trans_a" << kCsvSep 
        << "trans_b" << kCsvSep 
        << "type_a" << kCsvSep << "type_b" << kCsvSep 
        << "type_c" << kCsvSep << "lda" << kCsvSep << "ldb" << kCsvSep
        << "ldc" << kCsvSep << "stride_a" << kCsvSep
        << "stride_b" << kCsvSep << "stride_c" << kCsvSep
        << "alg_index" << std::endl;
}

}  // namespace


/*static*/ auto AutotunerUtil::AddResultToInMemoryCache(
      const AutotuneCacheKey& key, CacheValue result, 
      const AutotuneConfig& cfg) -> const CacheValue& 
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {

  static std::unique_ptr< std::ofstream > s_dump_fs;
  absl::MutexLock lock(&autotune_cache_mu);
  auto res = autotune_cache.emplace(key, std::move(result));
  auto it = res.first;

  auto dump_path = cfg.dump_path();
  if (res.second && !dump_path.empty()) {
    if (!s_dump_fs)
    {
      s_dump_fs = std::make_unique< std::ofstream >(std::string(dump_path));
      if (!s_dump_fs->is_open()) {
        LOG(WARNING) << "Unable to open: " << dump_path << " for writing!";
      } 
      CSVLegend(*s_dump_fs);
    }
    *s_dump_fs << key.Get() << kCsvSep << it->second << std::endl;
  }
  return it->second;
}

/*static*/ auto AutotunerUtil::TryToFindInInMemoryCache(
    const AutotuneCacheKey& key) -> absl::optional<CacheValue> 
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  absl::MutexLock lock(&autotune_cache_mu);
  auto it = autotune_cache.find(key);
  if (it == autotune_cache.end()) {
    return absl::nullopt;
  }
  return it->second;
}

/*static*/ void AutotunerUtil::ClearAutotuneResults() {
  absl::MutexLock lock(&autotune_cache_mu);
  autotune_cache.clear();
}

/* static*/ StatusOr<se::DeviceMemoryBase> AutotunerUtil::CreateBuffer(
    se::RedzoneAllocator& allocator, const Shape& shape,
    const AutotuneConfig& config, int64_t& rng_state) {
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase buffer,
                      allocator.AllocateBytes(ShapeUtil::ByteSizeOf(shape)));
  if (config.should_init_buffers()) {
    InitializeBuffer(allocator.stream(), shape.element_type(), &rng_state,
                     buffer);
  }
  return buffer;
}

namespace {
std::string ToCanonicalString(const HloInstruction* instr) {
  auto options = HloPrintOptions::Canonical();
  if (instr->opcode() != HloOpcode::kFusion) {
    options.set_print_backend_config(true);
    return instr->ToString(options);
  }
  options.set_print_subcomputation_mode(
      HloPrintOptions::PrintSubcomputationMode::kOff);
  // options.set_print_infeed_outfeed_config(false);
  // options.set_print_only_essential_constants(true);
  options.set_print_operand_shape(true);
  options.set_print_ids(false);
  // options.set_canonicalize_computations(true);

  // TODO(b/266210099): This is unsound. We should probably do the fingerprint
  // of the HLO computation proto instead.
  return instr->called_computations()[0]->ToString(options);
}

}  // namespace

/*static*/ auto AutotunerUtil::Autotune(
    const std::string& str_key, const AutotuneConfig& cfg,
    const AutotuneNoCacheFn& autotune_fn) -> StatusOr<CacheValue> {

  AutotuneCacheKey key(str_key);
  auto opt_res = TryToFindInInMemoryCache(key);
  if (opt_res.has_value()) {
    VLOG(1) << "In-memory autotune cache hit: key = " << key.Get();
    return *opt_res;
  }
  VLOG(1) << "Autotuning for key = " << key.Get() << " needed";
  TF_ASSIGN_OR_RETURN(auto result, autotune_fn());
  return AddResultToInMemoryCache(key, result, cfg);
}

/*static*/ Status AutotunerUtil::LoadAutotuneResultsFromFileOnce(
    const AutotuneConfig& cfg) {

  auto status = OkStatus();
  static absl::once_flag once;
  absl::call_once(once, [&cfg, &status] {
    status = LoadAutotuneResultsFromFile(cfg);
  });
  TF_RETURN_IF_ERROR(status);
  return status;
}

/*static*/ Status AutotunerUtil::LoadAutotuneResultsFromFile(
    const AutotuneConfig& cfg) {

  auto file_path = cfg.load_path();
  if (file_path.empty()) return OkStatus();

  std::ifstream ifs{std::string(file_path)};
  if (!ifs.is_open()) {
    LOG(WARNING) << "Unable to open autotune file for reading: " << file_path;
    return OkStatus();
  }

  std::vector< std::pair< AutotuneCacheKey, CacheValue >> vec;
  vec.reserve(256);

  std::string line;
  while(std::getline(ifs, line)) 
  {
    line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
    if (line.empty() || line[0] == '#') continue;
    
    std::istringstream iss(line);
    auto pos = line.find_last_of(kCsvSep);
    if (pos == std::string::npos) {
      LOG(WARNING) << "Unable to parse CSV row: " << line;
      continue;
    }
    auto key = line.substr(0, pos), sval = line.substr(pos + 1);
    char* p_end{};
    auto ival = std::strtol(sval.c_str(), &p_end, 10);
    if (p_end == sval.c_str()) {
      LOG(WARNING) << "Unable to parse CSV row: " << line;
      continue;
    }
    vec.emplace_back(AutotuneCacheKey{key}, CacheValue{});
    VLOG(1) << "Read autotune cache line: " << key << " -> " << ival;
    vec.back().second = ival;
  }
  for(const auto& p : vec) {
    AddResultToInMemoryCache(p.first, p.second, cfg);
  }

  LOG(INFO) << "Autotune results loaded from file: " << file_path;
  return OkStatus();
}

/*static*/ StatusOr<se::RedzoneAllocator>
AutotunerUtil::CreateRedzoneAllocator(const AutotuneConfig& config,
                                      se::Stream *stream,
                                      const DebugOptions& opts) {
  return se::RedzoneAllocator(
      stream, config.GetAllocator(), PtxOptsFromDebugOptions(opts),
      /*memory_limit=*/std::numeric_limits<int64_t>::max(),
      /*redzone_size=*/config.should_check_correctness()
          ? opts.xla_gpu_redzone_padding_bytes()
          : 0);
}

/*static*/ AutotuneConvCacheKey AutotunerUtil::ConvCacheKeyFromInstruction(
    const HloInstruction* instr, absl::string_view model_str) {
  
   auto options = HloPrintOptions::Canonical();
   options.set_print_backend_config(true);
   return std::make_tuple(std::string(model_str), instr->ToString(options));
}

}  // namespace gpu
}  // namespace xla

