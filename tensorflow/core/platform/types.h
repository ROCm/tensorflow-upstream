/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_TYPES_H_
#define TENSORFLOW_CORE_PLATFORM_TYPES_H_

#include <string>
#include <atomic>

#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/tstring.h"

// Include appropriate platform-dependent implementations
#if defined(PLATFORM_GOOGLE) || defined(GOOGLE_INTEGRAL_TYPES)
#include "tensorflow/core/platform/google/integral_types.h"
#elif defined(PLATFORM_WINDOWS)
#include "tensorflow/core/platform/windows/integral_types.h"
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) || \
    defined(PLATFORM_GOOGLE_ANDROID)
#include "tensorflow/core/platform/default/integral_types.h"
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

namespace tensorflow {

    
//[PROF-STATS]
struct ProfStats {
  std::atomic<uint64> flops;
  std::atomic<float> blaze_latency_ms;
  uint64 tao_op_calls = 0;
  bool dump_shapes = false;
  std::atomic<float> blaze_wait_ms;
  std::atomic<int> batch_size;
  std::atomic<int> blaze_running_counter;
  std::atomic<int> blaze_waiting_counter;
  std::atomic<int> blaze_nan;
  std::atomic<int> blaze_nan_counter;
  std::atomic<uint64> tensorflow_ops;
  std::atomic<uint64> cpu_flops;
  std::atomic<uint64> cpu_tensor_size;
  std::atomic<uint64> gpu_flops;
  std::atomic<uint64> gpu_tensor_size;
  std::atomic<uint64> gpu_kernels;
  std::atomic<uint64> pcie_h2d_times;
  std::atomic<uint64> pcie_h2d_size;
  std::atomic<uint64> pcie_d2h_times;
  std::atomic<uint64> pcie_d2h_size;
  std::atomic<bool> nan_qps;

  ProfStats() {
    flops = 0;
    blaze_latency_ms = 0;
    tao_op_calls = 0;
    blaze_wait_ms = 0;
    batch_size = 0;
    blaze_running_counter = 0;
    blaze_waiting_counter = 0;
    blaze_nan = 0;
    blaze_nan_counter = 0;
    dump_shapes = false;
    tensorflow_ops = 0;
    cpu_flops = 0;
    cpu_tensor_size = 0;
    gpu_flops = 0;
    gpu_tensor_size = 0;
    gpu_kernels = 0;
    pcie_h2d_times = 0;
    pcie_h2d_size = 0;
    pcie_d2h_times = 0;
    pcie_d2h_size = 0;
    nan_qps = false;
  }
};

// Alias tensorflow::string to std::string.
using std::string;

static const uint8 kuint8max = ((uint8)0xFF);
static const uint16 kuint16max = ((uint16)0xFFFF);
static const uint32 kuint32max = ((uint32)0xFFFFFFFF);
static const uint64 kuint64max = ((uint64)0xFFFFFFFFFFFFFFFFull);
static const int8 kint8min = ((int8)~0x7F);
static const int8 kint8max = ((int8)0x7F);
static const int16 kint16min = ((int16)~0x7FFF);
static const int16 kint16max = ((int16)0x7FFF);
static const int32 kint32min = ((int32)~0x7FFFFFFF);
static const int32 kint32max = ((int32)0x7FFFFFFF);
static const int64 kint64min = ((int64)~0x7FFFFFFFFFFFFFFFll);
static const int64 kint64max = ((int64)0x7FFFFFFFFFFFFFFFll);

// A typedef for a uint64 used as a short fingerprint.
typedef uint64 Fprint;

}  // namespace tensorflow

// Alias namespace ::stream_executor as ::tensorflow::se.
namespace stream_executor {}
namespace tensorflow {
namespace se = ::stream_executor;
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_TYPES_H_
