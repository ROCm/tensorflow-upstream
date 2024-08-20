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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/platform/numa.h"

namespace tensorflow {

//------------------------------------------------------------------------------
// A StreamGPUDevice is a virtual device that manages one stream group for the
// given GPU Device.
// -----------------------------------------------------------------------------
class StreamGPUDevice : public BaseGPUDevice {
 public:
  StreamGPUDevice(const SessionOptions& options, const string& name,
            Bytes memory_limit, const DeviceLocality& locality,
            TfGpuId tf_gpu_id, const string& physical_device_desc,
            Allocator* gpu_allocator, Allocator* cpu_allocator,
            const int stream_id)
      : BaseGPUDevice(options, name,
                      memory_limit, locality, tf_gpu_id,
                      physical_device_desc, gpu_allocator, cpu_allocator,
                      false /* sync every op */, stream_id),
        stream_id_(stream_id) {}
//        device_(real_device) {}

 // const Device* GetRealDevice() const override { return device_; }

  const int stream_id() const override { return stream_id_; }

 private:
  int stream_id_ = 0;
//  Device* device_ = nullptr;  // not owned, its real device
};

class GPUDevice : public BaseGPUDevice {
 public:
  GPUDevice(const SessionOptions& options, const string& name,
            Bytes memory_limit, const DeviceLocality& locality,
            TfGpuId tf_gpu_id, const string& physical_device_desc,
            std::vector<Allocator*>& gpu_allocators, Allocator* cpu_allocator)
      : BaseGPUDevice(options, name, memory_limit / gpu_allocators.size(), locality, tf_gpu_id,
                      physical_device_desc, gpu_allocators[0], cpu_allocator,
                      false /* sync every op */, 0 /* stream id */) {
    TF_CHECK_OK(tensorflow::ReadInt64FromEnvVar("TF_GPU_STREAM_GROUP_COUNT", 0, &stream_num_)); 
    for (int i = 0 ; i < stream_num_; ++i) {
      string stream_gpu_name = strings::StrCat("/job:localhost/replica:0/task:0/device:STREAM_GPU_",
                                               i, ":", tf_gpu_id.value());
      stream_devices_.push_back(absl::make_unique<StreamGPUDevice>(
        options, stream_gpu_name, memory_limit / gpu_allocators.size(), locality, tf_gpu_id,
        physical_device_desc, gpu_allocators[i], cpu_allocator, i /* stream id */));
    }
  }

  int GetStreamNum() const override { return stream_num_; }

  Device* GetStreamDevice(const int stream_id) override {
    if (stream_num_ == 0) {
      return this;
    }
    if (stream_id < 0 || stream_id >= stream_num_) {
      LOG(ERROR) << "Invalid value for stream_id: " << stream_id << ", max stream id: "
                 << stream_num_ << " when GetStreamDevice()";
      return nullptr;
    }
    return stream_devices_[stream_id].get();
  }

  Status InitStreamDevice(const SessionOptions& options) override {
    for (auto i = 0; i < stream_num_; ++i) {
      TF_RETURN_IF_ERROR(stream_devices_[i]->Init(options));
    }
    return Status::OK();
  }

 private:
  std::vector<std::unique_ptr<Device>> stream_devices_;
  int64 stream_num_ = 0;
};

class GPUDeviceFactory : public BaseGPUDeviceFactory {
 private:
  std::unique_ptr<BaseGPUDevice> CreateGPUDevice(
      const SessionOptions& options, const string& name, Bytes memory_limit,
      const DeviceLocality& locality, TfGpuId tf_gpu_id,
      const string& physical_device_desc, std::vector<Allocator*>& gpu_allocators,
      Allocator* cpu_allocator) override {
    return absl::make_unique<GPUDevice>(options, name, memory_limit, locality,
                                        tf_gpu_id, physical_device_desc,
                                        gpu_allocators, cpu_allocator);
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory, 210);

//------------------------------------------------------------------------------
// A CPUDevice that optimizes for interaction with GPUs in the
// process.
// -----------------------------------------------------------------------------
class GPUCompatibleCPUDevice : public ThreadPoolDevice {
 public:
  GPUCompatibleCPUDevice(const SessionOptions& options, const string& name,
                         Bytes memory_limit, const DeviceLocality& locality,
                         Allocator* allocator)
      : ThreadPoolDevice(options, name, memory_limit, locality, allocator),
        numa_node_(locality.numa_node()) {
    if (options.config.has_gpu_options()) {
      force_gpu_compatible_ =
          options.config.gpu_options().force_gpu_compatible();
    }
  }
  ~GPUCompatibleCPUDevice() override {}

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    GPUProcessState* ps = GPUProcessState::singleton();
    if (attr.gpu_compatible() || force_gpu_compatible_) {
      return ps->GetGpuHostAllocator(numa_node_);
    } else {
      // Call the parent's implementation.
      return ThreadPoolDevice::GetAllocator(attr);
    }
  }

 private:
  bool force_gpu_compatible_ = false;
  int numa_node_ = port::kNUMANoAffinity;
};

// The associated factory.
class GPUCompatibleCPUDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    devices->push_back("/physical_device:CPU:0");

    return Status::OK();
  }

  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }
    int num_numa_nodes = options.config.experimental().use_numa_affinity()
                             ? port::NUMANumNodes()
                             : 1;
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/device:CPU:", i);
      int numa_node = i % num_numa_nodes;
      DeviceLocality locality;
      locality.set_numa_node(numa_node);
      devices->push_back(absl::make_unique<GPUCompatibleCPUDevice>(
          options, name, Bytes(256 << 20), DeviceLocality(),
          ProcessState::singleton()->GetCPUAllocator(numa_node)));
    }

    return Status::OK();
  }
};
REGISTER_LOCAL_DEVICE_FACTORY("CPU", GPUCompatibleCPUDeviceFactory, 70);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
