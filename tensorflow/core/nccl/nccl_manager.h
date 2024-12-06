/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_NCCL_NCCL_MANAGER_H_
#define TENSORFLOW_CORE_NCCL_NCCL_MANAGER_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <vector>

// TODO(rmlarsen): Get rid of this workaround. "gpu_assert" is defined when
// setting EIGEN_USE_THREADS. But when defining EIGEN_USE_THREADS here,
// incAtomic and other CUDA specific symbols are no longer recognized.
#ifndef gpu_assert
#define gpu_assert(x)
#endif

#include "absl/container/flat_hash_map.h"
#if GOOGLE_CUDA
#include "third_party/nccl/nccl.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#endif
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "xla/pjrt/event_pool.h"

#define NCCL_COMM_ID_LEN 13
#define ALLTOALL_BACKWARD_PREFIX_LEN 4
#define SOCKET_NAME_MAXLEN (NI_MAXHOST + NI_MAXSERV)
#define NCCL_COMM_ID_MAXLEN (NCCL_COMM_ID_LEN + SOCKET_NAME_MAXLEN)

namespace tensorflow {

// NCCL manager is used to make the asynchronous communicator calls and to
// manage the per-device streams used for communication.
//
// See nccl_ops.cc for example usage, including description of memory
// management and stream synchronization.
class NcclManager {
 public:
  typedef std::function<void(Status)> DoneCallback;
  NcclManager();
  ~NcclManager();

  static NcclManager* instance();

#if USE_ROCM
  static int instance_count;
#endif

  // Calls `ncclGetUniqueId` and returns the id as a string.  The returned value
  // may be shared with other participants on different nodes and passed in to
  // multi-node collective invocations.
  string GenerateCommunicatorKey(char* nccl_comm_id, int rank, bool init_step);

  int LocalRanks();
  int WorkerIndex();
  int WorkerCount();
  int GlobalRanks();

  // A participant in a Collective.
  struct Participant {
    Participant(tensorflow::se::StreamExecutor* executor,
                tensorflow::se::Stream* tensor_stream,
#ifdef USE_TF215
                const tensorflow::DeviceBase::AcceleratorDeviceInfo* info,
#else
                const tensorflow::DeviceBase::GpuDeviceInfo* info,
#endif
                std::vector<const tensorflow::Tensor*> input_tensors,
                std::unique_ptr<tensorflow::se::Event> input_evt,
                std::vector<tensorflow::Tensor*> outputs,
                std::vector<int32_t> send_offsets,
                std::vector<int32_t> recv_offsets,
                std::vector<int32_t> send_counts,
                std::vector<int32_t> recv_counts, int global_rank,
                DoneCallback done_callback)
        : executor(executor),
          tensor_stream(tensor_stream),
          event_mgr(info->event_mgr),
          gpu_device_id(info->gpu_id),
#if USE_ROCM
          context(static_cast<GPUDeviceContext*>(info->default_context)),
#endif
          inputs(std::move(input_tensors)),
          input_event(std::move(input_evt)),
          outputs(std::move(outputs)),
          send_offsets(std::move(send_offsets)),
          recv_offsets(std::move(recv_offsets)),
          send_counts(std::move(send_counts)),
          recv_counts(std::move(recv_counts)),
          global_rank(global_rank),
          done_callback(std::move(done_callback)),
          event_pool(false),
          root(false) {
      DCHECK(executor != nullptr);
      DCHECK(event_mgr != nullptr);
      DCHECK(tensor_stream != nullptr);
      if (inputs.size() > 0 && inputs[0] != nullptr && !input_event) {
        absl::StatusOr<xla::EventPool::Handle> handel_event =
            event_pool.ThenAllocateAndRecordEvent(tensor_stream);
        input_event = std::unique_ptr<tensorflow::se::Event>(
            handel_event.value().event());
      }
    }

    // StreamExecutor for the device. Expected to be live for process lifetime.
    se::StreamExecutor* const executor = nullptr;

    // `tensor_stream` is the stream that should be waited on to ensure
    // `input`'s data is available on the GPU for the communication stream to
    // access. It is also the stream that will use the produced data;
    // `done_callback` is not called until the next kernel launched on `stream`
    // would see the data. Owned by the caller, who must keep it live until
    // `done_callback` is called.
    se::Stream* const tensor_stream;

    // EventMgr which polls on executor.
    // Owned by the caller, who must keep it live until `done_callback` is
    // called.
    EventMgr* const event_mgr;

    const int gpu_device_id;

#if USE_ROCM
    GPUDeviceContext* const context;
#endif

    // Owned by the caller, who must keep it live until `done_callback` is
    // called. Is NULL for participants that only receive data.
    std::vector<const tensorflow::Tensor*> inputs;

    // Wait on this event rather than synchronizing on the entire stream.
    // This allows greater concurrency between compute and nccl streams.
    std::unique_ptr<tensorflow::se::Event> input_event;

    // Owned by the caller, who must keep it live until `done_callback` is
    // called. Is NULL for participants that only send data.
    std::vector<tensorflow::Tensor*> outputs;

    // Split vector for alltoall, only for all2all
    std::vector<int32_t> send_offsets;
    std::vector<int32_t> recv_offsets;
    std::vector<int32_t> send_counts;
    std::vector<int32_t> recv_counts;

    // Rank across all devices and all nodes.
    // `global_rank` is not required for single-node collectives.
    const int global_rank;

    // The callback which is called at the completion of the NCCL operation.
    // When called, `output` has been set to the result of the operation. (note:
    // the stream may not yet have been synced)
    DoneCallback done_callback;

    // True if this is the root of the collective, e.g. source of broadcast.
    bool root;

    xla::EventPool event_pool;
  };

  // Data that provides context for the collective operation, including the
  // operation key, number of participants, and communicator key.
  struct Context {
    Context(const string& collective_key, int num_local_devices,
            int num_global_devices, const string& communicator_key,
            int source_rank)
        : collective_key(collective_key),
          num_local_devices(num_local_devices),
          num_global_devices(num_global_devices),
          communicator_key(communicator_key),
          source_rank(source_rank) {}

    // Unique key for this collective instance
    const string& collective_key;

    // Devices local to this node
    int num_local_devices;

    // Devices across all nodes
    int num_global_devices;

    // In order to use NCCL across nodes, the callee first has to generate a
    // `communicator_key` via `GenerateCommunicatorKey()` function and share
    // this with all the other nodes.  Each node should pass in this
    // `communicator_key` to the `NcclManager` functions.
    // `communicator_key` is not required for single-node collectives and can be
    // empty.
    const string& communicator_key;

    // Rank of broadcast source.
    int source_rank;
  };

  // Adds one participant to an all-reduce.
  void AddToAllReduce(std::unique_ptr<Participant> participant,
                      const Context& context, ncclRedOp_t reduction_op);

  // Adds one participant to an all-gather.
  void AddToAllGather(std::unique_ptr<Participant> participant,
                      const Context& context);

  // Adds one participant to a reduce-scatter.
  void AddToReduceScatter(std::unique_ptr<Participant> participant,
                          const Context& context, ncclRedOp_t reduction_op);

  // AddBroadcastSend and AddBroadcastRecv combine to send data from one sender
  // to all receivers.
  void AddBroadcast(std::unique_ptr<Participant> participant,
                    const Context& context);

  // AddReduceSend and AddReduceRecv combine to send data from all senders
  // to one receiver.
  void AddReduceSend(std::unique_ptr<Participant> participant,
                     const Context& context, ncclRedOp_t reduction_op);
  void AddReduceRecv(std::unique_ptr<Participant> participant,
                     const Context& context, ncclRedOp_t reduction_op);

  // Adds one participant to an all-to-all.
  void AddToAllToAll(std::unique_ptr<Participant> participant,
                     const Context& context);

  // Signals that the `Collective` corresponding to `key` is ready to launch
  // across all nodes participating in this multi-node collective operation.
  //
  // This should only be called for multi-node collectives; single-node
  // collectives are implicitly ready when all participants have called Add*
  // function.
  void SignalMultiNodeReady(const string& collective_key);

  // Aborts all collectives. After abortion, no further collectives can be
  // launched with this NcclManager.
  void StartAbort(const Status& s);

  // Resets a previously aborted NcclManager, making it available for future
  // collectives.
  void Reset();

 private:
  enum CollectiveType {
    kAllReduce = 1,
    kBroadcast = 2,
    kReduce = 3,
    kAllGather = 4,
    kReduceScatter = 5,
    kAllToAll = 6,
  };
  struct Collective;
  struct Communicator;
  struct CommunicatorMember;
  struct NcclStream;
  struct NcclProfiler;

  // Gets the `Communicator` object that will be used to enqueue NCCL kernels
  // for `collective`, and returns it via `communicator`.
  //
  // This may involve creating CUDA streams and NCCL initialization.  If a NCCL
  // or CUDA error occurs in the process, this returns an INTERNAL error with
  // the corresponding NCCL/CUDA error string.
  Status GetCommunicator(Collective* collective, Communicator** communicator);

  // Adds a participant device to the local `Collective` instance corresponding
  // to `collective_key`.  Launches the `Collective` if it is ready, which it
  // checks by calling `CheckReady()`.  Also performs consistency and sanity
  // checks before launching.
  void AddParticipant(std::unique_ptr<Participant> participant,
                      const Context& context, CollectiveType collective_type,
                      ncclRedOp_t reduction_op);

  // If `collective` is ready to run, removes it from the `collectives_` map and
  // returns true.  Otherwise returns false.
  // Assumes `collective_key` corresponds to `collective`.
  //
  // A collective is ready to run when all local participants have called Add*
  // function, and the collective is signalled globally ready via
  // `SetMultiNodeReady`.
  bool CheckReady(const string& collective_key, Collective* collective)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Run <collective>.  This calls takes ownership of <collective>.
  void RunCollective(Collective* collective);
  void LoopKernelLaunches(NcclStream* stream);

  mutex mu_;

  // Maps key to collectives currently being assembled or run.
  absl::flat_hash_map<string, Collective*> collectives_ TF_GUARDED_BY(mu_);

  // Maps a device to the communication streams that make up its collective.
  // This is used to share the stream across different communicators that
  // include the same device.
  absl::flat_hash_map<se::StreamExecutor*, std::vector<NcclStream*>>
      device_to_comm_streams_ TF_GUARDED_BY(mu_);

#ifdef USE_TF215
  std::vector<std::unique_ptr<Communicator>> communicators_ TF_GUARDED_BY(mu_);
#else
  std::vector<std::unique_ptr<Communicator>> communicators_;
#endif
  std::unique_ptr<Communicator> mgr_comm_;
  Status status_ TF_GUARDED_BY(mu_);

  int local_ranks_ = -1;
  int global_ranks_ = -1;
  int worker_index_ = -1;
  int worker_count_ = -1;

  NcclManager(const NcclManager&) = delete;
  void operator=(const NcclManager&) = delete;
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_NCCL_NCCL_MANAGER_H_
