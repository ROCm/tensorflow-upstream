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

#include "tensorflow/stream_executor/rocm/rocm_platform.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"

namespace stream_executor {
namespace gpu {

namespace {

// Synchronize with spinlocks.
const char kScheduleSpinString[] = "spin";
// Synchronize with spinlocks that also call CPU yield instructions.
const char kScheduleYieldString[] = "yield";
// Synchronize with a "synchronization primitive" (e.g. mutex).
const char kScheduleBlockingSyncString[] = "blocking_sync";

const DeviceOptions GetDeviceOptionsFromEnv() {
  const char* gpu_schedule_string =
      std::getenv("TF_ROCM_PLATFORM_GPU_DEVICE_SCHEDULE");

  if (gpu_schedule_string == nullptr) {
    return DeviceOptions::Default();
  }

  unsigned device_flags = 0;
  if (strcmp(kScheduleSpinString, gpu_schedule_string) == 0) {
    device_flags = DeviceOptions::kScheduleSpin;
  } else if (strcmp(kScheduleYieldString, gpu_schedule_string) == 0) {
    device_flags = DeviceOptions::kScheduleYield;
  } else if (strcmp(kScheduleBlockingSyncString, gpu_schedule_string) == 0) {
    device_flags = DeviceOptions::kScheduleBlockingSync;
  } else {
    LOG(QFATAL) << "Unknown option for environment variable "
                   "TF_ROCM_PLATFORM_GPU_DEVICE_SCHEDULE "
                << gpu_schedule_string << " should be one of {"
                << kScheduleBlockingSyncString << ", " << kScheduleSpinString
                << ", " << kScheduleYieldString << "}";
  }

  return DeviceOptions(device_flags);
}

}  // namespace

ROCmPlatform::ROCmPlatform()
    : name_("ROCM"), min_numa_node_(0), limit_numa_node_(0) {}

ROCmPlatform::~ROCmPlatform() {}

// Due to legacy issues in user code, we can't currently call InpectNumaNodes
// at module initialization time, because non-GPU programs still include this
// plugin via various methods, so instead, it has to be init-on-reference.
void ROCmPlatform::InspectNumaNodes() {
  // To get NUMA node information, we need to create all executors, so we can
  // examine their device descriptions to see their bus assignments.
  std::once_flag once;
  std::call_once(once, [&] {
    StreamExecutorConfig config;
    for (int i = 0; i < VisibleDeviceCount(); i++) {
      config.ordinal = i;
      StreamExecutor* exec = GetExecutor(config).ValueOrDie();
      if (i == 0) {
        // NUMA nodes may not start at 0, so set the minimum node  based on the
        // first executor we see.
        min_numa_node_ = exec->GetDeviceDescription().numa_node();
        limit_numa_node_ = min_numa_node_ + 1;
      } else {
        min_numa_node_ =
            std::min(min_numa_node_, exec->GetDeviceDescription().numa_node());
        limit_numa_node_ = std::max(
            limit_numa_node_, exec->GetDeviceDescription().numa_node() + 1);
      }
    }
  });
}

int ROCmPlatform::BusCount() {
  InspectNumaNodes();
  return limit_numa_node_ - min_numa_node_;
}

int ROCmPlatform::DeviceToBus(int device_ordinal) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  StreamExecutor* exec = GetExecutor(config).ValueOrDie();
  return exec->GetDeviceDescription().numa_node() - min_numa_node_;
}

port::StatusOr<StreamExecutor*> ROCmPlatform::FirstExecutorForBus(
    int bus_ordinal) {
  InspectNumaNodes();
  CHECK_LT(bus_ordinal, BusCount()) << "bus ordinal out of available range";
  for (int i = 0; i < VisibleDeviceCount(); i++) {
    if (DeviceToBus(i) == bus_ordinal) {
      StreamExecutorConfig config;
      config.ordinal = i;
      return GetExecutor(config).ValueOrDie();
    }
  }

  return port::Status{
      port::error::NOT_FOUND,
      absl::StrFormat("Executor for bus %d not found.", bus_ordinal)};
}

Platform::Id ROCmPlatform::id() const { return rocm::kROCmPlatformId; }

int ROCmPlatform::VisibleDeviceCount() const {
  // Throw away the result - it logs internally, and this [containing] function
  // isn't in the path of user control. It's safe to call this > 1x.

  if (!gpu::GpuDriver::Init().ok()) {
    return -1;
  }

  return GpuDriver::GetDeviceCount();
}

int ROCmPlatform::VirtualDeviceCount(int physical_gpu_id) const {
  auto iter = virtual_device_count_.find(physical_gpu_id);
  if (iter == virtual_device_count_.end()) {
    return 1;
  } else {
    return iter->second;
  }
}

port::Status ROCmPlatform::SetVirtualDeviceCount(int physical_gpu_id, int virtual_gpu_count) {
  if (virtual_gpu_count < 1) {
    return port::Status(
        port::error::INVALID_ARGUMENT,
        absl::StrFormat("virutal_gpu_count must >= 1, whereas the input is %d.", virtual_gpu_count));
  }
  auto iter = virtual_device_count_.find(physical_gpu_id);
  if (iter != virtual_device_count_.end()) {
    if (iter->second != virtual_gpu_count) {
      return port::Status(
          port::error::INVALID_ARGUMENT,
          absl::StrFormat("virtual gpu settings for device %d are not consistent: %d vs %d.",
                          physical_gpu_id, iter->second, virtual_gpu_count));
    } else {
      return port::Status::OK();
    }
  }

  virtual_device_count_[physical_gpu_id] = virtual_gpu_count;  return port::Status::OK();
}

const string& ROCmPlatform::Name() const { return name_; }

port::StatusOr<std::unique_ptr<DeviceDescription>>
ROCmPlatform::DescriptionForDevice(int ordinal) const {
  return GpuExecutor::CreateDeviceDescription(ordinal);
}

port::StatusOr<StreamExecutor*> ROCmPlatform::ExecutorForDevice(int ordinal) {
  return ExecutorForDevice(ordinal, 0);
}

port::StatusOr<StreamExecutor*> ROCmPlatform::ExecutorForDevice(
      int ordinal, int virtual_ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.virtual_ordinal = virtual_ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> ROCmPlatform::ExecutorForDevice(
    int ordinal, int virtual_ordinal, int stream_id) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.virtual_ordinal = virtual_ordinal;
  config.stream_id = stream_id;
  config.plugin_config = PluginConfig();
  config.device_options = GetDeviceOptionsFromEnv();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> ROCmPlatform::ExecutorForDeviceWithPluginConfig(
  int device_ordinal, const PluginConfig& plugin_config){
  return ExecutorForDeviceWithPluginConfig(device_ordinal, 0, plugin_config);
}

port::StatusOr<StreamExecutor*> ROCmPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, int virtual_ordinal,
    const PluginConfig& plugin_config) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.virtual_ordinal = virtual_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> ROCmPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, int virtual_ordinal,
    const PluginConfig& plugin_config, int stream_id) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.virtual_ordinal = virtual_ordinal;
  config.stream_id = stream_id;
  config.plugin_config = plugin_config;
  config.device_options = GetDeviceOptionsFromEnv();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> ROCmPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

port::StatusOr<std::unique_ptr<StreamExecutor>>
ROCmPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = absl::make_unique<StreamExecutor>(
      this, absl::make_unique<GpuExecutor>(config.plugin_config),
      config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return port::Status{
        port::error::INTERNAL,
        absl::StrFormat(
            "failed initializing StreamExecutor for ROCM device ordinal %d: %s",
            config.ordinal, init_status.ToString().c_str())};
  }

  return std::move(executor);
}

void ROCmPlatform::RegisterTraceListener(
    std::unique_ptr<TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register ROCM trace listener";
}

void ROCmPlatform::UnregisterTraceListener(TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister ROCM trace listener";
}

}  // namespace gpu

static void InitializeROCmPlatform() {
  // Disabling leak checking, MultiPlatformManager does not destroy its
  // registered platforms.
  auto status = MultiPlatformManager::PlatformWithName("ROCM");
  if (!status.ok()) {
    std::unique_ptr<gpu::ROCmPlatform> platform(new gpu::ROCmPlatform);
    SE_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(rocm_platform,
                            stream_executor::InitializeROCmPlatform());

DECLARE_MODULE_INITIALIZER(multi_platform_manager);
// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(rocm_platform, multi_platform_manager);
