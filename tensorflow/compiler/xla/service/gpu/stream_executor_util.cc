/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"

// #include "llvm/Support/SHA256.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#include "tensorflow/core/platform/cuda_libdevice_path.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/stream_executor/gpu_asm_opts.h"
#include "tensorflow/stream_executor/kernel_spec.h"


namespace xla {
namespace gpu {

using se::dnn::DataLayout;
using se::dnn::DataLayoutString;
using se::dnn::FilterLayout;
using se::dnn::FilterLayoutString;


namespace {
std::vector<tensorflow::AutotuneResult> KeepNonFailures(
    absl::Span<tensorflow::AutotuneResult const> profile_results) {
  // Filter out all failures except WRONG_RESULT, because false-positives are
  // possible (e.g. perhaps the reference algorithm is the one that's
  // incorrect!). Other failures can be detected with high accuracy. E.g.
  // REDZONE_MODIFIED which is also quite severe.
  std::vector<tensorflow::AutotuneResult> filtered_results;
  absl::c_copy_if(profile_results, std::back_inserter(filtered_results),
                  [](const tensorflow::AutotuneResult& r) {
                    return !r.has_failure() ||
                           r.failure().kind() == tensorflow::AutotuneResult::WRONG_RESULT;
                  });
  return filtered_results;
}

Status AllAlgorithmsFailedInternalError(
    absl::optional<absl::string_view> instr_str,
    absl::Span<tensorflow::AutotuneResult const> profile_results) {
  std::ostringstream msg;
  if (instr_str.has_value()) {
    msg << "All algorithms tried for " << instr_str.value()
        << " failed. Falling back to default algorithm.  Per-algorithm "
           "errors:";
  } else {
    msg << "All algorithms failed. Falling back to the default algorithm. "
        << "Per-algorithm errors:";
  }
  for (const auto& result : profile_results) {
    msg << "\n  " << result.failure().msg();
  }
  return Internal("%s", msg.str());
}

Status NoAlgorithmSuppliedInternalError(
    absl::optional<absl::string_view> instr_str) {
  std::ostringstream msg;
  if (instr_str.has_value()) {
    msg << "There are no algorithm candidates for computing: \n  "
        << instr_str.value()
        << "\nThis likely means that the instruction shape is not supported by "
           "the target GPU library.";
  } else {
    msg << "There are no algorithm candidates for computing the instruction.\n"
           "This likely means that the instruction shape is not supported by "
           "the target GPU library.";
  }
  return Internal("%s", msg.str());
}

void SortAutotuningResultsByRunTime(std::vector<tensorflow::AutotuneResult>& results) {
  absl::c_sort(results,
               [](const tensorflow::AutotuneResult& lhs, const tensorflow::AutotuneResult& rhs) {
                 return tensorflow::proto_utils::FromDurationProto(lhs.run_time()) <
                        tensorflow::proto_utils::FromDurationProto(rhs.run_time());
               });
}

absl::Span<tensorflow::AutotuneResult const> TopResultsWithinMeasurementError(
    std::vector<tensorflow::AutotuneResult>& results_sorted_by_runtime) {
  // This value was picked by repeatedly running a few kernels that run for a
  // short time and observing the run-time variance. A more rigorous analysis
  // of the measurement error might yield a better error threshold.
  constexpr absl::Duration kMeasurementError = absl::Microseconds(4);

  absl::Duration min_time = tensorflow::proto_utils::FromDurationProto(
      results_sorted_by_runtime.front().run_time());
  absl::Duration limit_time = min_time + kMeasurementError;

  auto limit_time_it = absl::c_find_if(
      results_sorted_by_runtime, [limit_time](const tensorflow::AutotuneResult& x) {
        return tensorflow::proto_utils::FromDurationProto(x.run_time()) > limit_time;
      });
  return absl::MakeSpan(&*results_sorted_by_runtime.begin(), &*limit_time_it);
}
}  // namespace

StatusOr<tensorflow::AutotuneResult> PickBestResult(
    absl::Span<tensorflow::AutotuneResult const> profile_results,
    absl::optional<absl::string_view> instr_str) {
  if (profile_results.empty()) {
    return NoAlgorithmSuppliedInternalError(instr_str);
  }

  std::vector<tensorflow::AutotuneResult> filtered_results =
      KeepNonFailures(profile_results);

  if (filtered_results.empty()) {
    return AllAlgorithmsFailedInternalError(instr_str, profile_results);
  }

  // Kernel run-time measurements within kMeasurementError are not precise.
  // Consider the lowest measurements within the error margin as equivalent and
  // within them prefer algorithms that use the least amount of scratch memory.
  SortAutotuningResultsByRunTime(filtered_results);
  auto top_within_error = TopResultsWithinMeasurementError(filtered_results);
  return *absl::c_min_element(top_within_error, [](const tensorflow::AutotuneResult& lhs,
                                                   const tensorflow::AutotuneResult& rhs) {
    return lhs.scratch_bytes() < rhs.scratch_bytes();
  });
}

bool IsVoltaOrLater(const se::StreamExecutor& stream_executor) {
  int major, minor;
  CHECK(stream_executor.GetDeviceDescription().cuda_compute_capability(&major,
                                                                       &minor));
  return major >= 7;
}

StatusOr<std::tuple<Layout, Layout, Layout>>
StreamExecutorConvLayoutsToXlaLayouts(const ConvolutionDimensionNumbers& dnums,
                                      DataLayout input, FilterLayout filter,
                                      DataLayout output) {
  std::vector<int64> input_layout;
  switch (input) {
    case DataLayout::kBatchDepthYX:
      input_layout.push_back(dnums.input_batch_dimension());
      input_layout.push_back(dnums.input_feature_dimension());
      input_layout.insert(input_layout.end(),
                          dnums.input_spatial_dimensions().begin(),
                          dnums.input_spatial_dimensions().end());
      break;
    case DataLayout::kBatchYXDepth:
      input_layout.push_back(dnums.input_batch_dimension());
      input_layout.insert(input_layout.end(),
                          dnums.input_spatial_dimensions().begin(),
                          dnums.input_spatial_dimensions().end());
      input_layout.push_back(dnums.input_feature_dimension());
      break;
    default:
      return InternalError("Invalid input layout %s for conv with dnums %s",
                           DataLayoutString(input),
                           ConvolutionDimensionNumbersToString(dnums));
  }

  std::vector<int64> filter_layout;
  switch (filter) {
    case FilterLayout::kOutputInputYX:
      filter_layout.push_back(dnums.kernel_output_feature_dimension());
      filter_layout.push_back(dnums.kernel_input_feature_dimension());
      filter_layout.insert(filter_layout.end(),
                           dnums.kernel_spatial_dimensions().begin(),
                           dnums.kernel_spatial_dimensions().end());
      break;
    case FilterLayout::kOutputYXInput:
      filter_layout.push_back(dnums.kernel_output_feature_dimension());
      filter_layout.insert(filter_layout.end(),
                           dnums.kernel_spatial_dimensions().begin(),
                           dnums.kernel_spatial_dimensions().end());
      filter_layout.push_back(dnums.kernel_input_feature_dimension());
      break;
    default:
      return InternalError("Invalid filter layout %s for conv with dnums %s",
                           FilterLayoutString(filter),
                           ConvolutionDimensionNumbersToString(dnums));
  }

  std::vector<int64> output_layout;
  switch (output) {
    case DataLayout::kBatchDepthYX:
      output_layout.push_back(dnums.output_batch_dimension());
      output_layout.push_back(dnums.output_feature_dimension());
      output_layout.insert(output_layout.end(),
                           dnums.output_spatial_dimensions().begin(),
                           dnums.output_spatial_dimensions().end());
      break;
    case DataLayout::kBatchYXDepth:
      output_layout.push_back(dnums.output_batch_dimension());
      output_layout.insert(output_layout.end(),
                           dnums.output_spatial_dimensions().begin(),
                           dnums.output_spatial_dimensions().end());
      output_layout.push_back(dnums.output_feature_dimension());
      break;
    default:
      return InternalError("Invalid output layout %s for conv with dnums %s",
                           DataLayoutString(output),
                           ConvolutionDimensionNumbersToString(dnums));
  }

  return std::make_tuple(LayoutUtil::MakeLayoutFromMajorToMinor(input_layout),
                         LayoutUtil::MakeLayoutFromMajorToMinor(filter_layout),
                         LayoutUtil::MakeLayoutFromMajorToMinor(output_layout));
}

StatusOr<std::tuple<DataLayout, FilterLayout, DataLayout>>
XlaConvLayoutsToStreamExecutorLayouts(const ConvolutionDimensionNumbers& dnums,
                                      const Layout& input, const Layout& filter,
                                      const Layout& output) {
  Layout nchw_input, nchw_filter, nchw_output;
  std::tie(nchw_input, nchw_filter, nchw_output) =
      StreamExecutorConvLayoutsToXlaLayouts(dnums, DataLayout::kBatchDepthYX,
                                            FilterLayout::kOutputInputYX,
                                            DataLayout::kBatchDepthYX)
          .ConsumeValueOrDie();

  Layout nhwc_input, nhwc_filter, nhwc_output;
  std::tie(nhwc_input, nhwc_filter, nhwc_output) =
      StreamExecutorConvLayoutsToXlaLayouts(dnums, DataLayout::kBatchYXDepth,
                                            FilterLayout::kOutputYXInput,
                                            DataLayout::kBatchYXDepth)
          .ConsumeValueOrDie();

  DataLayout input_layout;
  if (LayoutUtil::Equal(input, nchw_input)) {
    input_layout = DataLayout::kBatchDepthYX;
  } else if (LayoutUtil::Equal(input, nhwc_input)) {
    input_layout = DataLayout::kBatchYXDepth;
  } else {
    return InternalError("Invalid input layout %s for conv with dnums %s",
                         LayoutUtil::HumanString(input),
                         ConvolutionDimensionNumbersToString(dnums));
  }

  FilterLayout filter_layout;
  if (LayoutUtil::Equal(filter, nchw_filter)) {
    filter_layout = FilterLayout::kOutputInputYX;
  } else if (LayoutUtil::Equal(filter, nhwc_filter)) {
    filter_layout = FilterLayout::kOutputYXInput;
  } else {
    return InternalError("Invalid filter layout %s for conv with dnums %s",
                         LayoutUtil::HumanString(filter),
                         ConvolutionDimensionNumbersToString(dnums));
  }

  DataLayout output_layout;
  if (LayoutUtil::Equal(output, nchw_output)) {
    output_layout = DataLayout::kBatchDepthYX;
  } else if (LayoutUtil::Equal(output, nhwc_output)) {
    output_layout = DataLayout::kBatchYXDepth;
  } else {
    return InternalError("Invalid output layout %s for conv with dnums %s",
                         LayoutUtil::HumanString(output),
                         ConvolutionDimensionNumbersToString(dnums));
  }

  return std::make_tuple(input_layout, filter_layout, output_layout);
}

tensorflow::mutex_lock LockGpu(const se::StreamExecutor* stream_exec) {
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  // se::Platform*s are global singletons guaranteed to live forever.
  static auto* mutexes =
      new std::map<std::pair<const se::Platform*, /*device_ordinal*/ int64>,
                   tensorflow::mutex>();

  tensorflow::mutex_lock global_lock(mu);
  auto it = mutexes
                ->emplace(std::piecewise_construct,
                          std::make_tuple(stream_exec->platform(),
                                          stream_exec->device_ordinal()),
                          std::make_tuple())
                .first;
  return tensorflow::mutex_lock{it->second};
}

StatusOr<std::unique_ptr<se::KernelBase>> CreateKernel(
    absl::string_view kernel_name, uint64 num_args, absl::string_view ptx,
    absl::Span<const uint8> cubin_data, se::StreamExecutor* stream_exec) {
  se::MultiKernelLoaderSpec loader_spec(num_args);
  loader_spec.AddCudaPtxInMemory(ptx, kernel_name);

  if (!cubin_data.empty()) {
    loader_spec.AddCudaCubinInMemory(
        reinterpret_cast<const char*>(cubin_data.data()), kernel_name);
  }

  auto kernel_base = absl::make_unique<se::KernelBase>(stream_exec);
  TF_RETURN_IF_ERROR(stream_exec->GetKernel(loader_spec, kernel_base.get()));
  return std::move(kernel_base);
}

Status ExecuteKernelOnStream(const se::KernelBase& kernel,
                             absl::Span<const se::DeviceMemoryBase> args,
                             int64 threads_per_block, int64 block_count,
                             se::Stream* stream) {
  static constexpr int kKernelArgsLimit = 1024;
  auto kernel_args = absl::make_unique<se::KernelArgsArray<kKernelArgsLimit>>();
  for (const se::DeviceMemoryBase& buf : args) {
    kernel_args->add_device_memory_argument(buf);
  }
  return stream->parent()->Launch(stream, se::ThreadDim(threads_per_block),
                                  se::BlockDim(block_count), kernel,
                                  *kernel_args);
}

se::GpuAsmOpts GpuAsmOptsFromConfig(const HloModuleConfig& hlo_module_config) {
  return se::GpuAsmOpts(
      hlo_module_config.debug_options().xla_gpu_disable_ptxas_optimizations(),
      hlo_module_config.debug_options().xla_gpu_cuda_data_dir());
}

// Unimplemented for integers yet.
template <typename T, typename Generator>
typename std::enable_if<std::is_integral<T>::value,
                        T>::type static UniformDistribution(T lhs, T rhs,
                                                            Generator* gen) =
    delete;

template <typename T, typename Generator>
typename std::enable_if<std::is_floating_point<T>::value,
                        T>::type static UniformDistribution(T lhs, T rhs,
                                                            Generator* gen) {
  return std::uniform_real_distribution<T>(lhs, rhs)(*gen);
}

template <typename T>
static void InitializeTypedBuffer(se::Stream* stream,
                                  se::DeviceMemoryBase buffer,
                                  int64* rng_state) {
  // Accesses to static variables are not locked, since the caller is already
  // in a critical section.

  // Use a large prime number to fragment the accesses.
  constexpr int host_buffer_size = 10069;
  static std::vector<T>* host_buffer = [] {
    auto* ret = new std::vector<T>(host_buffer_size);
    // Default-seeded random numbers.
    std::mt19937 gen;
    for (auto& element : *ret) {
      constexpr bool kIsIntegral = std::numeric_limits<T>::is_integer;
      constexpr bool kIsLowRange =
          !kIsIntegral && std::numeric_limits<T>::max_exponent <=
                              std::numeric_limits<Eigen::half>::max_exponent;
      // Only double gets random values in double.  Other data types get random
      // values in float then cast them to the target data types.
      using RandomType = typename std::conditional<std::is_same<T, double>::value,
                                                   double, float>::type;
      // Scale down the values for fp16 to have less overflows.
      auto upper_bound = RandomType(kIsLowRange ? 0.1 : 1.0);
      auto rand_val = UniformDistribution(RandomType(0), upper_bound, &gen);
      // For bf16, float or double, it is between [0,1].
      // For fp16, it ranges between [0, 0.1].
      // For integer types, element is either 0 or 1 for less overflows
      // especially for int8_t.
      element = T(kIsIntegral ? rand_val + 0.5 : rand_val);
    }
    return ret;
  }();
  // The buffer of random numbers is treated as being circular, and the seed in
  // *rng_state is the offset in host_buffer that is copied to the zeroth index
  // on the device. For large buffers then repeatedly copying the data from the
  // host is expensive, so we just copy it once and use a kernel to repeat the
  // data as needed.
  CHECK_EQ(0, buffer.size() % sizeof(T));
  int64 elements_to_fill = buffer.size() / sizeof(T);
  int64 host_index = *rng_state;
  CHECK_LT(host_index, host_buffer_size);
  *rng_state = (*rng_state + elements_to_fill) % host_buffer_size;
  // Copy the last part of `host_buffer` to the start of `buf` on the device
  int64 first_size =
      std::min<int64>(host_buffer_size - host_index, elements_to_fill);
  stream->ThenMemcpy(&buffer, host_buffer->data() + host_index,
                             first_size * sizeof(T));
  elements_to_fill -= first_size;
  if (elements_to_fill == 0) {
    // Nothing more to do
    return;
  }
  // Issue a second host->device copy to transfer the rest of host_buffer
  int64 second_size = std::min<int64>(host_index, elements_to_fill);
  CHECK_LE(first_size + second_size, host_buffer_size);
  // = buffer.GetByteSlice(first_size * sizeof(T), second_size * sizeof(T));
  se::DeviceMemoryBase mem(static_cast< uint8_t *>(buffer.opaque()) 
        + first_size * sizeof(T), second_size * sizeof(T));

  stream->ThenMemcpy(&mem, host_buffer->data(), mem.size());
  elements_to_fill -= second_size;
  if (elements_to_fill == 0) {
    // Nothing more to do
    return;
  }
#ifdef GOOGLE_CUDA
  // Repeat the host_buffer_size elements at the start of `buf` to the end
  CHECK_EQ(elements_to_fill, buffer.size() / sizeof(T) - host_buffer_size);
  se::StreamExecutor* executor = stream->parent();
  auto kernel =
      se::TypedKernelFactory<se::DeviceMemoryBase, int64, int64>::Create(
          executor, "RepeatBufferKernel", repeat_buffer_kernel::kernel());
  if (!kernel.ok()) {
    LOG(FATAL) << "Could not create RepeatBufferKernel: " << kernel.status();
  }
  // Launch the kernel with at least host_buffer_bytes threads. Each thread
  // will read one byte of `host_buffer` from the start of `buffer`, where the
  // Memcpy call(s) above put it, and scatter it through the rest of `buffer`.
  constexpr int64 host_buffer_bytes = host_buffer_size * sizeof(T);
  constexpr int threads_per_block = 256;
  constexpr int blocks_per_grid =
      (host_buffer_bytes + threads_per_block - 1) / threads_per_block;
  TF_CHECK_OK(stream->ThenLaunch(se::ThreadDim(threads_per_block, 1, 1),
                                 se::BlockDim(blocks_per_grid, 1, 1), *kernel,
                                 buffer, host_buffer_bytes,
                                 static_cast<int64>(buffer.size())));
#endif
}

void InitializeBuffer(se::Stream* stream, PrimitiveType buffer_type,
                           int64* rng_state, se::DeviceMemoryBase buffer) {
  switch (buffer_type) {
    case xla::F16:
      return InitializeTypedBuffer<Eigen::half>(stream, buffer, rng_state);
    case xla::F32:
    case xla::C64:
      return InitializeTypedBuffer<float>(stream, buffer, rng_state);
    case xla::F64:
    case xla::C128:
      return InitializeTypedBuffer<double>(stream, buffer, rng_state);
    case xla::S8:
      return InitializeTypedBuffer<int8_t>(stream, buffer, rng_state);
    case xla::U8:
      return InitializeTypedBuffer<uint8_t>(stream, buffer, rng_state);
    default:
      LOG(FATAL) << "Unexpected type";
  }
}

}  // namespace gpu
}  // namespace xla
