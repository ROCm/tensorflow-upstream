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
#include <openssl/sha.h>

#include "tensorflow/core/protobuf/autotune_results.pb.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/base64.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace gpu {
namespace {

// Bump this version whenever you change the structure of the results.
// LINT.IfChange(version)
constexpr int kVersion = 3;
// LINT.ThenChange()

}  // namespace

using AutotuneCacheMap = absl::flat_hash_map<AutotuneCacheKey, tensorflow::AutotuneResult>;

static absl::Mutex autotune_cache_mu(absl::kConstInit);
static auto& autotune_cache ABSL_GUARDED_BY(autotune_cache_mu) =
    *new AutotuneCacheMap();

StatusOr<std::string> GetBase64EncodedSha256Hash(absl::string_view s) {
  
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, s.data(), s.size());

    //Aws::Utils::ByteBuffer hash(SHA256_DIGEST_LENGTH);
    std::array<uint8_t, 32> hash;
    SHA256_Final(hash.data(), &sha256);
  
  // llvm::SHA256 sha256;
  // sha256.update(llvm::StringRef(s));
  // std::array<uint8_t, 32> hash = sha256.final();
  // C++ strict aliasing rules allow reinterpret casting to (const) char*.
  absl::string_view hash_view(reinterpret_cast<const char*>(hash.data()),
                              hash.size());
  std::string base64_encoded_hash;
  TF_RETURN_IF_ERROR(tensorflow::Base64Encode(hash_view, &base64_encoded_hash));
  return base64_encoded_hash;
}

namespace {

// Get the path corresponding to the given key.
StatusOr<std::string> GetCacheFilePath(absl::string_view cache_dir,
                                             const AutotuneCacheKey& key) {
  if (cache_dir.empty()) {
    return xla::InvalidArgument("autotune_cache_dir should not be empty");
  }

  TF_ASSIGN_OR_RETURN(std::string key_hash,
                      GetBase64EncodedSha256Hash(key.ToString()));
  return tensorflow::io::JoinPath(cache_dir, absl::StrCat(key_hash, ".textproto"));
}

struct ResultAndInserted {
  // The result that ended up in the cache. This is the existing result if
  // inserted is false, and the new result if inserted is true.
  //
  // We return a value, not a pointer, for thread safety reasons.
  tensorflow::AutotuneResult result;
  // Did we insert the given result into the cache?
  bool inserted;
};

ResultAndInserted AddResultToInMemoryCache(const AutotuneCacheKey& key,
                                           tensorflow::AutotuneResult result)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  absl::MutexLock lock(&autotune_cache_mu);
  auto [it, inserted] = autotune_cache.emplace(key, std::move(result));
  return {it->second, inserted};
}

Status AddResultToFileBasedCacheIfEnabled(const AutotuneCacheKey& key,
                                                tensorflow::AutotuneResult result,
                                                absl::string_view cache_dir)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  if (cache_dir.empty()) {
    return Status::OK();
  }

  TF_ASSIGN_OR_RETURN(const std::string file_path,
                      GetCacheFilePath(cache_dir, key));

  VLOG(1) << "Writing autotune result to file: " << file_path;

  std::string result_str;
  if (!tensorflow::protobuf::TextFormat::PrintToString(result, &result_str)) {
    return xla::InternalError("Failed to serialize autotune result.");
  }

  // Rename trick: Write to a temporary file, then rename it to the final file
  // to avoid mingled files when multiple processes are writing to the same
  // file. Also avoids reading incomplete files. (This may not work on all file
  // systems.)
  std::string temp_file_path = tensorflow::io::GetTempFilename(".textproto");
  auto* default_env = tensorflow::Env::Default();
  TF_RETURN_IF_ERROR(
      tensorflow::WriteStringToFile(default_env, temp_file_path, result_str));
  return default_env->RenameFile(temp_file_path, file_path);
}

StatusOr<ResultAndInserted> AddResultToCaches(const AutotuneCacheKey& key,
                                                    tensorflow::AutotuneResult result,
                                                    absl::string_view cache_dir)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  ResultAndInserted result_and_inserted = AddResultToInMemoryCache(key, result);
  if (result_and_inserted.inserted) {
    TF_RETURN_IF_ERROR(AddResultToFileBasedCacheIfEnabled(
        key, result_and_inserted.result, cache_dir));
  }
  return result_and_inserted;
}

absl::optional<tensorflow::AutotuneResult> TryToFindInInMemoryCache(
    const AutotuneCacheKey& key) ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  absl::MutexLock lock(&autotune_cache_mu);
  auto it = autotune_cache.find(key);
  if (it == autotune_cache.end()) {
    return absl::nullopt;
  }
  return it->second;
}

StatusOr<absl::optional<tensorflow::AutotuneResult>>
TryToFindInFileBasedCacheIfEnabled(const AutotuneCacheKey& key,
                                   absl::string_view cache_dir)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  if (cache_dir.empty()) {
    return absl::optional<tensorflow::AutotuneResult>{};
  }

  TF_ASSIGN_OR_RETURN(const std::string file_path,
                      GetCacheFilePath(cache_dir, key));
  if (!tensorflow::Env::Default()->FileExists(file_path).ok()) {
    VLOG(1) << "Autotune result file not found: " << file_path;
    return absl::optional<tensorflow::AutotuneResult>{};
  }

  VLOG(1) << "Autotune result file found: " << file_path;
  std::string autotune_result_str;
  TF_RETURN_IF_ERROR(ReadFileToString(tensorflow::Env::Default(), file_path,
                                           &autotune_result_str));
  tensorflow::AutotuneResult result;
  if (!tensorflow::protobuf::TextFormat::ParseFromString(autotune_result_str,
                                                  &result)) {
    return xla::InvalidArgument("Failed to parse autotune result.");
  }
  return absl::optional<tensorflow::AutotuneResult>{result};
}

// Sort the results so that they're deterministic.
void SortAutotuneResults(tensorflow::AutotuneResults* results) {
  std::sort(results->mutable_results()->pointer_begin(),
            results->mutable_results()->pointer_end(),
            [](const auto* a, const auto* b) {
              return std::make_pair(absl::string_view(a->device()),
                                    absl::string_view(a->hlo())) <
                     std::make_pair(absl::string_view(b->device()),
                                    absl::string_view(b->hlo()));
            });
}

}  // namespace

// Serialize `results` to string as a proto.
StatusOr<std::string> AutotuneResultsToString(
    const tensorflow::AutotuneResults& results, bool as_textproto) {
  if (as_textproto) {
    std::string textproto;
    if (tensorflow::protobuf::TextFormat::PrintToString(results, &textproto)) {
      return textproto;
    } else {
      return Internal("Failed to serialize autotune results.");
    }
  }
  return results.SerializeAsString();
}

namespace {
// Serialize a single entry to `results`.
void SerializeAutotuneEntry(tensorflow::AutotuneResults* results, const AutotuneCacheKey& k,
                            const tensorflow::AutotuneResult* res) {
  auto& entry = *results->add_results();
  entry.set_device(std::string(k.GetModelStr()));
  entry.set_hlo(std::string(k.GetHlo()));
  *entry.mutable_result() = *res;
}
}  // namespace

/*static*/ Status AutotunerUtil::SerializeAutotuneResults(
    tensorflow::AutotuneResults* results) {
  absl::MutexLock lock(&autotune_cache_mu);
  for (const auto& [k, result] : autotune_cache) {
    SerializeAutotuneEntry(results, k, &result);
  }

  results->set_version(kVersion);
  SortAutotuneResults(results);

  return Status::OK();
}

/*static*/ Status AutotunerUtil::LoadAutotuneResults(
    const tensorflow::AutotuneResults& results) {
  absl::MutexLock lock(&autotune_cache_mu);
  for (const tensorflow::AutotuneResults::Entry& result : results.results()) {
    if (auto [it, inserted] = autotune_cache.emplace(
            AutotuneCacheKey(result.device(), result.hlo()), result.result());
        !inserted) {
      return xla::InternalError(
          "Duplicate autotuning result for %s", it->first.ToString().c_str());
    }
  }
  return Status::OK();
}

/*static*/ void AutotunerUtil::ClearAutotuneResults() {
  absl::MutexLock lock(&autotune_cache_mu);
  autotune_cache.clear();
}

/*static*/ bool AutotunerUtil::ResultCacheIsEmpty() {
  absl::MutexLock lock(&autotune_cache_mu);
  return autotune_cache.empty();
}

/* static*/ StatusOr<se::DeviceMemoryBase> AutotunerUtil::CreateBuffer(
    se::RedzoneAllocator& allocator, const Shape& shape,
    const AutotuneConfig& config, int64& rng_state) {
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

AutotuneCacheKey::AutotuneCacheKey(absl::string_view model_str,
                                   const HloInstruction& instr)
    : AutotuneCacheKey(model_str, ToCanonicalString(&instr)) {}

namespace {
StatusOr<absl::optional<tensorflow::AutotuneResult>> TryFindInCache(
    const AutotuneCacheKey& key, absl::string_view cache_dir)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  absl::optional<tensorflow::AutotuneResult> opt_result = TryToFindInInMemoryCache(key);
  if (opt_result.has_value()) {
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "In-memory autotune cache hit";
    } else if (VLOG_IS_ON(2)) {
      LOG(INFO) << "In-memory autotune cache hit: key = " << key.ToString();
    }
    return opt_result;
  }

  TF_ASSIGN_OR_RETURN(opt_result,
                      TryToFindInFileBasedCacheIfEnabled(key, cache_dir));
  if (opt_result.has_value()) {
    AddResultToInMemoryCache(key, opt_result.value());

    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "File-based autotune cache hit";
    } else if (VLOG_IS_ON(2)) {
      LOG(INFO) << "File-based autotune cache hit: key = " << key.ToString();
    }
    return opt_result;
  }

  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "Autotune cache miss";
  } else if (VLOG_IS_ON(2)) {
    LOG(INFO) << "Autotune cache miss: key = " << key.ToString();
  }
  return absl::optional<tensorflow::AutotuneResult>{};
}
}  // namespace

/*static*/ AutotuneCacheKey AutotunerUtil::GetKey(
    const HloInstruction* instr, const AutotuneConfig& config) {
  return AutotuneCacheKey(config.GetModelStr(), *instr);
}

/*static*/ StatusOr<bool> AutotunerUtil::IsInCache(
    const AutotuneCacheKey& key, const AutotuneConfig& config) {
  TF_ASSIGN_OR_RETURN(absl::optional<tensorflow::AutotuneResult> opt_res,
                      TryFindInCache(key, config.autotune_cache_dir()));
  return opt_res.has_value();
}

/*static*/ StatusOr<bool> AutotunerUtil::AddResult(
    const AutotuneCacheKey& key, tensorflow::AutotuneResult result,
    const AutotuneConfig& config) {
  TF_ASSIGN_OR_RETURN(
      ResultAndInserted result_and_inserted,
      AddResultToCaches(key, std::move(result), config.autotune_cache_dir()));
  return result_and_inserted.inserted;
}

/*static*/ StatusOr<tensorflow::AutotuneResult> AutotunerUtil::Autotune(
    const HloInstruction* instr, const AutotuneConfig& config,
    const AutotuneNoCacheFn& autotune_fn) {
  const AutotuneCacheKey key = GetKey(instr, config);
  TF_ASSIGN_OR_RETURN(absl::optional<tensorflow::AutotuneResult> opt_res,
                      TryFindInCache(key, config.autotune_cache_dir()));
  if (opt_res.has_value()) {
    return opt_res.value();
  }

  TF_ASSIGN_OR_RETURN(tensorflow::AutotuneResult autotune_result, autotune_fn());

  TF_ASSIGN_OR_RETURN(ResultAndInserted result_and_inserted,
                      AddResultToCaches(key, std::move(autotune_result),
                                        config.autotune_cache_dir()));
  return result_and_inserted.result;
}

namespace {

bool IsTextProtoPath(absl::string_view file_path) {
  return absl::EndsWith(file_path, ".txt") ||
         absl::EndsWith(file_path, ".textproto") ||
         absl::EndsWith(file_path, ".prototxt") ||
         absl::EndsWith(file_path, ".pbtxt");
}

}  // anonymous namespace

/*static*/ Status AutotunerUtil::LoadAutotuneResults(
    absl::string_view data, bool as_textproto) {
  tensorflow::AutotuneResults results;
  // The cast here is necessary for MacOS builds.
  bool parse_success =
      as_textproto ? tensorflow::protobuf::TextFormat::ParseFromString(
                         std::string(data), &results)             // NOLINT
                   : results.ParseFromString(std::string(data));  // NOLINT
  if (!parse_success) {
    return xla::InvalidArgument(
        "Failed to parse autotune results string.");
  }
  if (results.version() != kVersion) {
    return xla::InvalidArgument(
        "Version mismatch in autotune results. Expected %d but was %d",
        kVersion, results.version());
  }

  TF_RETURN_IF_ERROR(LoadAutotuneResults(results));
  return Status::OK();
}

/*static*/ StatusOr<std::string> AutotunerUtil::SerializeAutotuneResults(
    bool as_textproto) {
  tensorflow::AutotuneResults results;
  TF_RETURN_IF_ERROR(SerializeAutotuneResults(&results));
  return AutotuneResultsToString(results, as_textproto);
}

/*static*/ Status AutotunerUtil::SerializeAutotuneResultsToFile(
    const tensorflow::AutotuneResults& results, absl::string_view file_path) {
  TF_RET_CHECK(!file_path.empty());
  TF_RET_CHECK(results.version() > 0)
      << "Did you call SerializeAutotuneResults to get this tensorflow::AutotuneResults?";

  std::string resolved_path{file_path};
  // if (!tensorflow::io::ResolveTestPrefixes(file_path, resolved_path)) {
  //   return FailedPrecondition("File path can not be resolved: %s", file_path);
  // }

  TF_ASSIGN_OR_RETURN(
      std::string autotune_results_str,
      AutotuneResultsToString(results, IsTextProtoPath(resolved_path)));
  TF_RETURN_IF_ERROR(tensorflow::WriteStringToFile(tensorflow::Env::Default(), resolved_path,
                                            autotune_results_str));
  LOG(INFO) << "Autotune results serialized to file: " << resolved_path;

  return Status::OK();
}

/*static*/ Status AutotunerUtil::SerializeAutotuneResultsToFile(
    absl::string_view file_path) {
  tensorflow::AutotuneResults results;
  TF_RETURN_IF_ERROR(SerializeAutotuneResults(&results));
  return SerializeAutotuneResultsToFile(results, file_path);
}

/*static*/ Status AutotunerUtil::LoadAutotuneResultsFromFile(
    absl::string_view file_path) {
  TF_RET_CHECK(!file_path.empty());

  std::string resolved_path{file_path};
  // if (!tensorflow::io::ResolveTestPrefixes(file_path, resolved_path)) {
  //   return FailedPrecondition("File path can not be resolved: %s", file_path);
  // }

  if (!tensorflow::Env::Default()->FileExists(resolved_path).ok()) {
    return FailedPrecondition("Autotune results file does not exist: %s",
                              resolved_path);
  }
  std::string autotune_results_str;
  TF_RETURN_IF_ERROR(ReadFileToString(tensorflow::Env::Default(), resolved_path,
                                           &autotune_results_str));

  TF_RETURN_IF_ERROR(LoadAutotuneResults(autotune_results_str,
                                         IsTextProtoPath(resolved_path)));

  LOG(INFO) << "Autotune results loaded from file: " << resolved_path;

  return Status::OK();
}

se::GpuAsmOpts PtxOptsFromDebugOptions(const DebugOptions& debug_options) {
  return se::GpuAsmOpts(
      debug_options.xla_gpu_disable_ptxas_optimizations(),
      debug_options.xla_gpu_cuda_data_dir());
}

/*static*/ StatusOr<se::RedzoneAllocator>
AutotunerUtil::CreateRedzoneAllocator(const AutotuneConfig& config,
                                      se::Stream *stream,
                                      const DebugOptions& opts) {
  return se::RedzoneAllocator(
      stream, config.GetAllocator(), PtxOptsFromDebugOptions(opts),
      /*memory_limit=*/std::numeric_limits<int64>::max(),
      /*redzone_size=*/config.should_check_correctness()
          ? opts.xla_gpu_redzone_padding_bytes()
          : 0);
}

}  // namespace gpu
}  // namespace xla
