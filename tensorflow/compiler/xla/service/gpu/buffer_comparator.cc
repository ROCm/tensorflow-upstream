/* Copyright 2018 The OpenXLA Authors.

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

#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/stream_executor/device_description.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

template <typename ElementT>
using ComparisonKernelT =
    se::TypedKernel<se::DeviceMemory<ElementT>, se::DeviceMemory<ElementT>,
                    float, uint64_t, se::DeviceMemory<uint64_t>>;

// Compares two buffers on the GPU.
//
// Returns `true` if two buffers are equal, `false` otherwise.
template <typename ElementT>
StatusOr<bool> BufferComparator::DeviceCompare(
    se::Stream* stream, se::DeviceMemoryBase current,
    se::DeviceMemoryBase expected, absl::string_view kernel_name,
    void* kernel_symbol) const {
  se::StreamExecutor* executor = stream->parent();

  se::ScopedDeviceMemory<uint64_t> out_param =
      executor->AllocateOwnedScalar<uint64_t>();

  stream->ThenMemZero(out_param.ptr(), sizeof(uint64_t));
  if (current.size() != expected.size()) {
    return Internal("Mismatched buffer size: %d bytes vs. %d bytes",
                    current.size(), expected.size());
  }

  se::DeviceMemory<ElementT> current_typed(current);
  se::DeviceMemory<ElementT> expected_typed(expected);
  uint64_t buffer_size = current_typed.ElementCount();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ComparisonKernelT<ElementT>> comparison_kernel,
      (executor->CreateTypedKernel<se::DeviceMemory<ElementT>,
                                    se::DeviceMemory<ElementT>, float, uint64_t,
                                    se::DeviceMemory<uint64_t>>(
           kernel_name, kernel_symbol)));

  const se::DeviceDescription& gpu_device_info =
      executor->GetDeviceDescription();

  LaunchDimensions dim = CalculateLaunchDimensions(shape_, gpu_device_info);

  stream->ThenLaunch(se::ThreadDim(dim.threads_per_block()),
                     se::BlockDim(dim.block_count()), *comparison_kernel,
                  current_typed, expected_typed, static_cast<float>(tolerance_),
                     buffer_size, out_param.cref());

  uint64_t result = -1;
  CHECK_EQ(out_param->size(), sizeof(result));
  stream->ThenMemcpy(&result, *out_param, sizeof(result));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return result == 0;
}

// Host side comparison code that does the same thing, but reports some of the
// differences as well. It only print logs for debugging.
//
// Returns true if no differences were seen, false otherwise.
template <typename ElementType, typename ComparisonType>
StatusOr<bool> BufferComparator::HostCompare(
    se::Stream* stream, se::DeviceMemoryBase current,
    se::DeviceMemoryBase expected) const {
  int64_t n = current.size() / sizeof(ElementType);
  std::vector<ElementType> host_current(n), host_expected(n);

  stream->ThenMemcpy(host_current.data(), current, current.size());
  stream->ThenMemcpy(host_expected.data(), expected, expected.size());
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  const auto canonicalize = [](ComparisonType a) -> ComparisonType {
    if (std::is_same<ElementType, Eigen::half>::value && a) {
      constexpr ComparisonType kMaxFp16Value = 65505;
      if (std::isnan(a)) {
        return a;
      }
      return std::max(-kMaxFp16Value, std::min(a, kMaxFp16Value));
    }
    return a;
  };
  int differences_seen = 0;
  for (int64_t i = 0; i < n && differences_seen < 10; ++i) {
    auto current_value = static_cast<ComparisonType>(host_current[i]);
    auto expected_value = static_cast<ComparisonType>(host_expected[i]);
    ComparisonType current_value_canonical = canonicalize(current_value);
    ComparisonType expected_value_canonical = canonicalize(expected_value);
    if (std::isnan(current_value_canonical) &&
        std::isnan(expected_value_canonical)) {
      continue;
    }
    if (std::isinf(current_value_canonical) &&
        std::isinf(expected_value_canonical) &&
        current_value_canonical == expected_value_canonical) {
      continue;
    }
    if (std::isfinite(current_value_canonical) !=
            std::isfinite(expected_value_canonical) ||
        !(std::abs(current_value_canonical - expected_value_canonical) /
              (std::max(std::abs(current_value_canonical),
                        std::abs(expected_value_canonical)) +
               1) <
          tolerance_)) {
      ++differences_seen;
      LOG(ERROR) << "Difference at " << i << ": " << current_value
                 << ", expected " << expected_value;
    }
  }
  return differences_seen == 0;
}

template <typename ElementT, typename ComparisonT>
StatusOr<bool> BufferComparator::CompareEqualParameterized(
    se::Stream* stream, se::DeviceMemoryBase current,
    se::DeviceMemoryBase expected, absl::string_view kernel_name,
    void* kernel_symbol) const {
  XLA_SCOPED_LOGGING_TIMER("BufferComparator::CompareEqual");
  TF_ASSIGN_OR_RETURN(bool result,
                      DeviceCompare<ElementT>(stream, current, expected,
                                              kernel_name, kernel_symbol));

  if (result) {
    return true;
  }

  TF_ASSIGN_OR_RETURN(bool host_return, (HostCompare<ElementT, ComparisonT>(
                                            stream, current, expected)));
  CHECK_EQ(host_return, result)
      << "Host comparison succeeded even though GPU comparison failed.";
  return false;
}

StatusOr<bool> BufferComparator::CompareEqual(
    se::Stream* stream, se::DeviceMemoryBase current,
    se::DeviceMemoryBase expected) const {
  switch (shape_.element_type()) {
    case xla::F16:
      return CompareEqualParameterized<Eigen::half, float>(
          stream, current, expected, "fp16_comparison",
          buffer_comparator::fp16_comparison());
    case xla::BF16:
      return CompareEqualParameterized<tensorflow::bfloat16, float>(
          stream, current, expected, "bf16_comparison",
          buffer_comparator::bf16_comparison());
    case xla::F32:
      return CompareEqualParameterized<float, float>(
          stream, current, expected, "fp32_comparison",
          buffer_comparator::fp32_comparison());
    case xla::F64:
      return CompareEqualParameterized<double, double>(
          stream, current, expected, "fp64_comparison",
          buffer_comparator::fp64_comparison());
    case xla::S8:
      return CompareEqualParameterized<int8_t, float>(
          stream, current, expected, "int8_comparison",
          buffer_comparator::int8_comparison());
    case xla::S32:
      return CompareEqualParameterized<int32_t, float>(
          stream, current, expected, "int32_comparison",
          buffer_comparator::int32_comparison());
    default:
      return Unimplemented("Unimplemented element type");
  }
}

BufferComparator::BufferComparator(const Shape& shape,
                                   const HloModuleConfig& config,
                                   double tolerance)
    : shape_(shape), config_(config), tolerance_(tolerance) {
  // Normalize complex shapes: since we treat the passed array as a contiguous
  // storage it does not matter which dimension are we doubling.
  auto double_dim_size = [&]() {
    int64_t prev_zero_dim_size = shape_.dimensions(0);
    shape_.set_dimensions(0, prev_zero_dim_size * 2);
  };

  if (shape_.element_type() == PrimitiveType::C64) {
    // C64 is just two F32s next to each other.
    shape_.set_element_type(PrimitiveType::F32);
    double_dim_size();
  } else if (shape_.element_type() == PrimitiveType::C128) {
    // C128 is just two F64s next to each other.
    shape_.set_element_type(PrimitiveType::F64);
    double_dim_size();
  }
}

}  // namespace gpu
}  // namespace xla
