/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cpu/runtime/while_thunk.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/buffer_allocations.h"
#include "xla/service/cpu/runtime/resource_use.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/runtime/thunk_testlib.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS

#include "Eigen/ThreadPool"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

TEST(WhileThunkTest, BufferUses) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice pred_slice(&alloc, 0, sizeof(char));
  BufferAllocation::Slice cond_read_slice(&alloc, 10, 10);
  BufferAllocation::Slice body_read_slice(&alloc, 20, 10);

  ThunkSequence cond_sequence;
  cond_sequence.push_back(
      std::make_unique<BufferUseThunk>(BufferUse::Read(cond_read_slice)));

  ThunkSequence body_sequence;
  body_sequence.push_back(
      std::make_unique<BufferUseThunk>(BufferUse::Read(body_read_slice)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      WhileThunk::Create({"while"}, pred_slice, std::move(cond_sequence),
                         std::move(body_sequence)));

  EXPECT_EQ(thunk->buffer_uses().size(), 3);
  EXPECT_EQ(thunk->buffer_uses()[0], BufferUse::Write(pred_slice));
  EXPECT_EQ(thunk->buffer_uses()[1], BufferUse::Read(cond_read_slice));
  EXPECT_EQ(thunk->buffer_uses()[2], BufferUse::Read(body_read_slice));
}

TEST(WhileThunkTest, ResourceUses) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice pred_slice(&alloc, 0, sizeof(char));

  auto token0 = Resource::Create(Resource::kToken);
  auto token1 = Resource::Create(Resource::kToken);

  ThunkSequence cond_sequence;
  cond_sequence.push_back(
      std::make_unique<ResourceUseThunk>(ResourceUse::Read(token0)));

  ThunkSequence body_sequence;
  body_sequence.push_back(
      std::make_unique<ResourceUseThunk>(ResourceUse::Read(token1)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      WhileThunk::Create({"while"}, pred_slice, std::move(cond_sequence),
                         std::move(body_sequence)));

  EXPECT_EQ(thunk->resource_uses().size(), 2);
  EXPECT_EQ(thunk->resource_uses()[0], ResourceUse::Read(token0));
  EXPECT_EQ(thunk->resource_uses()[1], ResourceUse::Read(token1));
}

// Below are fake thunks that always launch tasks into the intra-op thread pool,
// so that we can test that WhileThunk::Execute correctly handles asynchronous
// cond and body thunk sequences.

class CondThunk : public Thunk {
 public:
  CondThunk(size_t counter, BufferAllocation::Slice pred_slice)
      : Thunk(Kind::kKernel, {"cond"}),
        counter_(counter + 1),
        pred_slice_(pred_slice) {}

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final {
    auto event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();

    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase predicate_mem,
        params.buffer_allocations->GetDeviceAddress(pred_slice_));
    bool* predicate = reinterpret_cast<bool*>(predicate_mem.opaque());

    // Continue while loop until counter reaches 0.
    *predicate = counter_.fetch_sub(1) > 1;

    params.intra_op_threadpool->getPool()->Schedule(
        [event] { event.SetStateConcrete(); });

    return event;
  }

  BufferUses buffer_uses() const final {
    return {BufferUse::Write(pred_slice_)};
  }

 private:
  std::atomic<size_t> counter_;
  BufferAllocation::Slice pred_slice_;
};

class BodyThunk : public Thunk {
 public:
  explicit BodyThunk(BufferAllocation::Slice counter_slice)
      : Thunk(Kind::kKernel, {"body"}), counter_slice_(counter_slice) {}

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final {
    auto event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();

    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase counter_mem,
        params.buffer_allocations->GetDeviceAddress(counter_slice_));

    int32_t* counter = reinterpret_cast<int32_t*>(counter_mem.opaque());
    ++*counter;

    params.intra_op_threadpool->getPool()->Schedule(
        [event] { event.SetStateConcrete(); });

    return event;
  }

  BufferUses buffer_uses() const final { return {}; }

 private:
  BufferAllocation::Slice counter_slice_;
};

TEST(WhileThunkTest, NonBlockingExecute) {
  static constexpr size_t kNumIterations = 100;

  BufferAllocation pred_alloc(0, sizeof(char), 0);
  BufferAllocation cnt_alloc(1, sizeof(int32_t), 0);

  BufferAllocation::Slice pred_slice(&pred_alloc, 0, sizeof(char));
  BufferAllocation::Slice cnt_slice(&cnt_alloc, 0, sizeof(int32_t));

  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<char> predicate = {false};
  std::vector<int32_t> counter = {0};

  buffers.emplace_back(se::DeviceMemoryBase(predicate.data(), sizeof(char)));
  buffers.emplace_back(se::DeviceMemoryBase(counter.data(), sizeof(int32_t)));

  BufferAllocations allocations(buffers);

  ThunkSequence cond_sequence;
  cond_sequence.push_back(
      std::make_unique<CondThunk>(kNumIterations, pred_slice));

  ThunkSequence body_sequence;
  body_sequence.push_back(std::make_unique<BodyThunk>(cnt_slice));

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      WhileThunk::Create({"while"}, pred_slice, std::move(cond_sequence),
                         std::move(body_sequence)));

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "while-test", 8);
  Eigen::ThreadPoolDevice device(thread_pool.AsEigenThreadPool(),
                                 thread_pool.NumThreads());

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = &device;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(counter[0], kNumIterations);
}

}  // namespace
}  // namespace xla::cpu
