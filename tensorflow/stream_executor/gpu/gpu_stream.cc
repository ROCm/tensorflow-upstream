/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/gpu/gpu_stream.h"

#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/core/util/env_var.h"

namespace stream_executor {
namespace gpu {

bool GpuStream::Init(int priority) {
  bool ok = GpuDriver::InitEvent(parent_->gpu_context(), &completed_event_,
                              GpuDriver::EventFlags::kDisableTiming)
      .ok();

  if (!ok)
    return false;
  tensorflow::int64 stream_strategy;
  tensorflow::ReadInt64FromEnvVar("STREAM_STRATEGY", 0, &stream_strategy);

  if (stream_strategy>=0 && priority >= 0 && priority <= 2) {
    StreamPool pool = parent_->GetStreamPool();
    if (pool.size()>0) {
      int id = parent_->virtual_ordinal() % 4;

      if (stream_strategy<3)
        id += 4*((priority+stream_strategy)%3);
      else if(stream_strategy<6)
        id += 4*((3-priority+stream_strategy)%3);

      gpu_stream_ = (GpuStreamHandle) pool[id];
    }
    return true;
  }

  return GpuDriver::CreateStream(parent_->gpu_context(), &gpu_stream_, priority);
}

void GpuStream::Destroy() {
  if (completed_event_ != nullptr) {
    port::Status status =
        GpuDriver::DestroyEvent(parent_->gpu_context(), &completed_event_);
    if (!status.ok()) {
      LOG(ERROR) << status.error_message();
    }
  }

  StreamPool pool = parent_->GetStreamPool();
  for (auto s: pool)
    if (gpu_stream_ == s)
      return;

  GpuDriver::DestroyStream(parent_->gpu_context(), &gpu_stream_);
}

bool GpuStream::IsIdle() const {
  return GpuDriver::IsStreamIdle(parent_->gpu_context(), gpu_stream_);
}

GpuStream* AsGpuStream(Stream* stream) {
  DCHECK(stream != nullptr);
  return static_cast<GpuStream*>(stream->implementation());
}

GpuStreamHandle AsGpuStreamValue(Stream* stream) {
  DCHECK(stream != nullptr);
  return AsGpuStream(stream)->gpu_stream();
}

}  // namespace gpu
}  // namespace stream_executor
