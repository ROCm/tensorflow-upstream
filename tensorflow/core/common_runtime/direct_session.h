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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DIRECT_SESSION_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DIRECT_SESSION_H_

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/debugger_state_interface.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

class CostModel;
class DebugGateway;
class Device;
class DirectSessionFactory;
class CallbackFrame;

// StreamGroupMgr manages the allocation and recycling of stream groups. It
// maintains a min-heap, so it can give the stream group with the lowest load at
// each request.
class StreamGroupMgr {
 public:
  StreamGroupMgr(const size_t total_num);
  virtual ~StreamGroupMgr(){};

  // Apply for a stream group.
  int Require();

  // Release the stream group when finish using it.
  void Release(const int stream_id);

 private:
  // One stream group is represented by a node in the min-heap. The node
  // contains a workload counter to record how many workloads are running in the
  // stream group, and an accumulator to record how many times has the node been
  // used for. New task should be allocated to the node of the lowest load.
  struct StreamGroupNode {
    int id_;
    int workload_;
    StreamGroupNode(const int id, const int workload = 0)
        : id_(id), workload_(workload) {}
  };

  // Swap two stream group nodes.
  void swap(const size_t, const size_t);

  size_t total_num_;
  mutable mutex mu_;
  int swap_left_ GUARDED_BY(mu_);
  std::vector<std::unique_ptr<StreamGroupNode>> stream_group_heap_
      GUARDED_BY(mu_);
  std::unordered_map<int, size_t> id2heap_map_ GUARDED_BY(mu_);
};

class DirectSession : public Session {
 public:
  typedef std::function<void(Session*)> CloseCallback;

  // Takes ownership of 'device_mgr'.
  // 'factory' is used to unregister the DirectSession with 'factory' when its
  // closed. This ensures that Reset requests from the 'factory' don't get sent
  // to sessions that are already closed.
  DirectSession(const SessionOptions& options, const DeviceMgr* device_mgr,
                DirectSessionFactory* factory);
  ~DirectSession() override;

  typedef std::vector<std::pair<string, Tensor>> NamedTensorList;
  typedef std::unordered_map<StringPiece, Node*, StringPieceHasher> NameNodeMap;

  ::tensorflow::Status Create(const GraphDef& graph) override;
  ::tensorflow::Status Create(GraphDef&& graph) override;
  ::tensorflow::Status Extend(const GraphDef& graph) override;
  ::tensorflow::Status Extend(GraphDef&& graph) override;
  ::tensorflow::Status Run(const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           const std::vector<string>& target_nodes,
                           std::vector<Tensor>* outputs) override;

  // NOTE: Experimental and subject to change.
  ::tensorflow::Status Run(const ::tensorflow::RunOptions& run_options,
                           const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           const std::vector<string>& target_nodes,
                           std::vector<Tensor>* outputs,
                           RunMetadata* run_metadata) override;

  // NOTE: PRunSetup and PRun are added to support partial execution. This
  // feature is experimental and subject to change.
  ::tensorflow::Status PRunSetup(const std::vector<string>& input_names,
                                 const std::vector<string>& output_names,
                                 const std::vector<string>& target_nodes,
                                 string* handle) override;
  ::tensorflow::Status PRun(const string& handle, const NamedTensorList& inputs,
                            const std::vector<string>& output_names,
                            std::vector<Tensor>* outputs) override;

  // Reset clears 'containers' from the device_mgr of the DirectSession.
  // If 'containers' is empty, then Reset clears the default container.
  ::tensorflow::Status Reset(const std::vector<string>& containers);

  ::tensorflow::Status ListDevices(
      std::vector<DeviceAttributes>* response) override;
  ::tensorflow::Status Close() override;
  ::tensorflow::Status LocalDeviceManager(const DeviceMgr** output) override {
    *output = device_mgr_.get();
    return ::tensorflow::Status::OK();
  }

  void ExportCostModels(CostModelManager::CostModelMap* cost_models) {
    cost_model_manager_.ExportCostModels(cost_models);
  }

  ::tensorflow::Status MakeCallable(const CallableOptions& callable_options,
                                    CallableHandle* out_handle) override;

  ::tensorflow::Status RunCallable(CallableHandle handle,
                                   const std::vector<Tensor>& feed_tensors,
                                   std::vector<Tensor>* fetch_tensors,
                                   RunMetadata* run_metadata,
                                   int blaze_stream_id = -1,
                                   uint64_t before_padding = 0,
                                   uint64_t after_padding = 0) override;

  ::tensorflow::Status RunCallable(
      CallableHandle handle, const std::vector<Tensor>& feed_tensors,
      std::vector<Tensor>* fetch_tensors, RunMetadata* run_metadata,
      const thread::ThreadPoolOptions& threadpool_options,
      int blaze_stream_id = -1,
      uint64_t before_padding = 0,
      uint64_t after_padding = 0) override;

  ::tensorflow::Status ReleaseCallable(CallableHandle handle) override;

  const SessionOptions& options() const { return options_; }
  
  ::tensorflow::Status PreCreateExecutors(const std::vector<std::string>& inputs,
                                          const std::vector<std::string>& outputs,
                                          const std::vector<std::string>& target_nodes,
                                          const ::tensorflow::RunOptions& run_options);

  int RequireStreamGroup() override;

  void ReleaseStreamGroup(const int stream_id) override;

#ifdef GOOGLE_CUDA
  ::tensorflow::Status CreateForCapture(const GraphDef& graph) override;
  ::tensorflow::Status CreateForCapture(GraphDef&& graph) override;
  ::tensorflow::Status RunForCapture(const std::vector<std::pair<string, Tensor> >& inputs,
                     const std::vector<string>& output_tensor_names,
                     const std::vector<string>& target_node_names,
                     CudaGraphMeta* cuda_graph_meta) override;
  ::tensorflow::Status RunForCapture(const RunOptions& run_options,
                     const std::vector<std::pair<string, Tensor> >& inputs,
                     const std::vector<string>& output_tensor_names,
                     const std::vector<string>& target_node_names,
                     RunMetadata* run_metadata,
                     CudaGraphMeta* cuda_graph_meta) override;
  bool SupportsCudaGraph() override { return true; };
  cudaStream_t EnableGraphCapture() override;
  void DisableGraphCapture() override;
  std::unordered_map<std::string, GraphDef>* GetCudaGraphRewriteDefs() override {
    return &cudagraph_defs_;
  };
#endif

  void RunAsync(const RunOptions& run_options,
      const NamedTensorList& inputs,
      const std::vector<string>& output_names,
      const std::vector<string>& target_nodes,
      std::vector<Tensor>* outputs,
      RunMetadata* run_metadata,
      StatusCallback done,
      std::atomic<int64_t>* flops = nullptr) override;

  void RunAsync(const RunOptions& run_options,
      const NamedTensorList& inputs,
      const std::vector<string>& output_names,
      const std::vector<string>& target_nodes,
      std::vector<Tensor>* outputs,
      RunMetadata* run_metadata,
      CallbackFrame* frame,
      StatusCallback done,
      std::atomic<int64_t>* flops = nullptr);


  void SetStepInitId(int step_id) {
    step_id_counter_ = step_id;
  }

  // TODO: const cast check for aios auto scale
  SessionOptions* get_options() const override {return const_cast<SessionOptions*>(&options_);}
 private:
  // For access to collective_graph_key_.
  friend class DirectSessionCollectiveTest;

  // We create one executor and its dependent library runtime for
  // every partition.
  struct PerPartitionExecutorsAndLib {
    Graph* graph = nullptr;                  // not owned.
    Device* device = nullptr;                // not owned.
    FunctionLibraryRuntime* flib = nullptr;  // not owned.
    std::unique_ptr<Executor> executor;
  };

  // An ExecutorsAndKeys is created for a given set of feeds/fetches.
  // 'step_count' is the number of times this graph is executed.
  // 'graph' is the entire graph being executed. 'name_to_node'
  // maps node name to node. We keep 'graph' and 'name_to_node' only in
  // the case of partial runs. Each item in 'items' is the executor for
  // a partition of the graph bundled with its dependent library runtime.
  // 'input_keys' are the rendezvous keys for the feeds and 'output_keys'
  // are rendezvous keys for the fetches.
  struct ExecutorsAndKeys {
    ExecutorsAndKeys() : step_count(0) {}

    std::atomic_int_fast64_t step_count;
    std::unique_ptr<Graph> graph;
    NameNodeMap name_to_node;
    std::vector<PerPartitionExecutorsAndLib> items;
    std::vector<std::vector<PerPartitionExecutorsAndLib>> stream_items;
    std::unordered_map<string, size_t> input_name_to_index;
    std::unordered_map<string, string> input_name_to_rendezvous_key;
    std::unordered_map<string, size_t> output_name_to_index;
    std::unordered_map<string, string> output_name_to_rendezvous_key;

    DataTypeVector input_types;
    DataTypeVector output_types;

    CallableOptions callable_options;

    int64 collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;
  };

  // A FunctionInfo object is created for every unique set of feeds/fetches.
  // This info could be folded into the ExecutorsAndKeys object but we would
  // like to maintain a deletion order in which the OpKernels (owned by the
  // executor) should be destroyed first, followed by the resources in the
  // device and then followed by the function stuff.
  // TODO(rohanj): Consolidate function library definitions so that we can
  // instantiate only one ProcFLR and lib_def and make this just a member
  // variable and not a vector.
  // 'flib_def' is the function library used.
  // 'proc_flr' is the collection of FunctionLibraryRuntime objects, one per
  // device.
  struct FunctionInfo {
    std::unique_ptr<FunctionLibraryDefinition> flib_def;
    std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr;
  };

  // For each live partial execution, the session maintains a RunState.
  // 'status' is the current status of this partial execution. 'executor_done'
  // is "notified" when all executors are done. 'pending_inputs' are the set
  // of pending feeds and 'pending_outputs' are the set of pending fetches.
  struct RunState {
    mutex mu_;
    Status status GUARDED_BY(mu_);
    IntraProcessRendezvous* rendez = nullptr;
    std::unique_ptr<CollectiveExecutor::Handle> collective_executor;
    std::unique_ptr<StepStatsCollector> collector;
    Notification executors_done;
    std::unordered_map<string, bool> pending_inputs;   // true if fed
    std::unordered_map<string, bool> pending_outputs;  // true if fetched
    TensorStore tensor_store;
    ScopedStepContainer step_container;

    RunState(int64 step_id, const std::vector<Device*>* devices);

    RunState(const std::vector<string>& pending_input_names,
             const std::vector<string>& pending_output_names, int64 step_id,
             const std::vector<Device*>* devices);

    // Returns true if all pending inputs and outputs have been completed.
    bool PendingDone() const;

    ~RunState();
  };

  struct RunStateArgs {
    RunStateArgs(const DebugOptions& options) : debug_options(options) {}

    bool is_partial_run = false;
    string handle;
    std::unique_ptr<Graph> graph;
    const DebugOptions& debug_options;
    int64 collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;
  };

  // Retrieves an already existing set of executors to run 'inputs' and
  // 'outputs', or creates and caches them for future use.
  ::tensorflow::Status GetOrCreateExecutors(
      gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
      gtl::ArraySlice<string> target_nodes,
      ExecutorsAndKeys** executors_and_keys, RunStateArgs* run_state_args);

  // Creates a set of executors to run the subgraph defined by
  // `callable_options`.
  ::tensorflow::Status CreateExecutors(
      const CallableOptions& callable_options,
      std::unique_ptr<ExecutorsAndKeys>* out_executors_and_keys,
      std::unique_ptr<FunctionInfo>* out_func_info,
      RunStateArgs* run_state_args);

  // Creates several graphs given the existing graph_def_ and the
  // input feeds and fetches, given 'devices'. The graphs share a common
  // function library 'flib_def'.
  ::tensorflow::Status CreateGraphs(
      const BuildGraphOptions& options,
      std::unordered_map<string, std::unique_ptr<Graph>>* outputs,
      std::unique_ptr<FunctionLibraryDefinition>* flib_def,
      RunStateArgs* run_state_args, DataTypeVector* input_types,
      DataTypeVector* output_types, int64* collective_graph_key);

  ::tensorflow::Status RunInternal(
      int64 step_id, const RunOptions& run_options,
      CallFrameInterface* call_frame, ExecutorsAndKeys* executors_and_keys,
      RunMetadata* run_metadata,
      const thread::ThreadPoolOptions& threadpool_options, int blaze_stream_id = -1,
      CudaGraphMeta* cuda_graph_meta = nullptr);

  void RunInternalAsync(
      int64 step_id, const RunOptions& run_options,
      CallFrameInterface* call_frame, ExecutorsAndKeys* executors_and_keys,
      RunMetadata* run_metadata,
      const thread::ThreadPoolOptions& threadpool_options,
      const NamedTensorList& inputs,
      const std::vector<string>& output_names,
      const std::vector<string>& target_nodes,
      std::vector<Tensor>* outputs,
      CallbackFrame* frame,
      StatusCallback done,
      std::atomic<int64_t>* flops = nullptr);

  ::tensorflow::Status AfterRunAsync(const ::tensorflow::RunOptions& run_options,
       const std::vector<string>& output_names,
       const std::vector<string>& target_nodes,
       std::vector<Tensor> *outputs,
       CallbackFrame* frame,
       RunMetadata* run_metadata,
	   uint64 start_time_usecs);
  // Returns whether inter-op execution uses a global pool or the input
  // `run_options` requests being run on inter_op_thread_pool = 0 in case
  // multiple pools are configured.
  bool ShouldUseRunHandlerPool(const RunOptions& run_options) const;

  ::tensorflow::Status ExtendLocked(GraphDef graph)
      EXCLUSIVE_LOCKS_REQUIRED(graph_state_lock_);

  ::tensorflow::Status ResourceHandleToInputTensor(
      const Tensor& resource_tensor, Tensor* retrieved_tensor);

  // Feeds more inputs to the executors, triggering further execution.
  ::tensorflow::Status SendPRunInputs(
      const std::vector<std::pair<string, Tensor>>& inputs,
      const ExecutorsAndKeys* executors_and_keys,
      IntraProcessRendezvous* rendez);

  // Fetches more outputs from the executors. It waits until the output
  // tensors are computed.
  ::tensorflow::Status RecvPRunOutputs(
      const std::vector<string>& output_names,
      const ExecutorsAndKeys* executors_and_keys, RunState* run_state,
      std::vector<Tensor>* outputs);

  // Check if the specified fetches can be computed from the feeds
  // that we have already provided.
  ::tensorflow::Status CheckFetch(
      const std::vector<std::pair<string, Tensor>>& feeds,
      const std::vector<string>& fetches,
      const ExecutorsAndKeys* executors_and_keys, const RunState* run_state);

  // Use the appropriate WaitForNotification function based on whether
  // operation_timeout_in_ms is greater than 0.
  //
  // If the timeout expires, the `cm->StartCancel()` will be called.
  ::tensorflow::Status WaitForNotification(Notification* n,
                                           int64 timeout_in_ms);
  void WaitForNotification(RunState* run_state, CancellationManager* cm,
                           int64 timeout_in_ms);

  ::tensorflow::Status CheckNotClosed() {
    mutex_lock l(closed_lock_);
    if (closed_) return errors::Cancelled("Session has been closed.");
    return ::tensorflow::Status::OK();
  }

  ::tensorflow::Status CheckGraphCreated(const char* method) {
    mutex_lock l(graph_state_lock_);
    if (!graph_created_) {
      return errors::InvalidArgument(
          "Session was not created with a graph before ", method, "!");
    }
    return ::tensorflow::Status::OK();
  }

  ::tensorflow::Status CreateDebuggerState(
      const CallableOptions& options, int64 global_step,
      int64 session_run_index, int64 executor_step_index,
      std::unique_ptr<DebuggerStateInterface>* debugger_state);

  ::tensorflow::Status DecorateAndPublishGraphForDebug(
      const DebugOptions& debug_options, Graph* graph, Device* device);

  const SessionOptions options_;

  // Device structures.
  const std::unique_ptr<const DeviceMgr> device_mgr_;
  std::vector<Device*> devices_;  // not owned
  DeviceSet device_set_;

  // Unique session identifier.
  string session_handle_;
  mutex graph_state_lock_;
  bool graph_created_ GUARDED_BY(graph_state_lock_) = false;

  // The thread-pools to use for running ops, with a bool indicating if the pool
  // is owned.
  std::vector<std::pair<thread::ThreadPool*, bool>> thread_pools_;

  Status init_error_;  // Set to an error if construction failed.

  //[DYNAMIC-SHAPE]
  bool gemm_dynamic_batchsize_ = false;
  //[PROF-STATS]
  const int kProfStatsSampleRatio = 97;
  bool enable_prof_stats_ = true;
  int64 sampling_prof_stats_steps_ = kProfStatsSampleRatio;

  int64 gpu_stream_group_count_ = 0;

  // If true, blocks until device has finished all queued operations in a step.
  bool sync_on_finish_ = true;

  std::vector<std::unique_ptr<FunctionInfo>> functions_
      GUARDED_BY(executor_lock_);

  mutex executor_lock_;  // protects executors_
  // Holds mappings from signature to the executors that process
  // it. The reason for a level of indirection around mapped_type is
  // to guarantee address stability.
  // The map value is a shared_ptr since multiple map keys can point to the
  // same ExecutorsAndKey object.
  std::unordered_map<string, std::shared_ptr<ExecutorsAndKeys>> executors_
      GUARDED_BY(executor_lock_);

  class RunCallableCallFrame;
  struct Callable {
    std::shared_ptr<ExecutorsAndKeys> executors_and_keys;
    std::shared_ptr<FunctionInfo> function_info;
    ~Callable();
  };
  mutex callables_lock_;
  int64 next_callable_handle_ GUARDED_BY(callables_lock_) = 0;
  std::unordered_map<int64, Callable> callables_ GUARDED_BY(callables_lock_);

  // Holds mappings from handle to partial run state.
  std::unordered_map<string, std::unique_ptr<RunState>> partial_runs_
      GUARDED_BY(executor_lock_);

  // This holds all the tensors that are currently alive in the session.
  SessionState session_state_;

  DirectSessionFactory* const factory_;  // not owned
  CancellationManager* cancellation_manager_;
  std::unique_ptr<CollectiveExecutorMgrInterface> collective_executor_mgr_;

  // Map of placed stateful nodes, i.e. nodes for which is_stateful()
  // is true, such as "params" and "queue" nodes.  Once placed these
  // nodes can not be moved to a different device.  Maps node names to
  // device names.
  std::unordered_map<string, string> stateful_placements_
      GUARDED_BY(graph_state_lock_);

  // Execution_state; used when placing the entire graph.
  std::unique_ptr<GraphExecutionState> execution_state_
      GUARDED_BY(graph_state_lock_);

  // The function library, before any rewrites or optimizations have been
  // performed. In particular, CreateGraphs() may need to modify the function
  // library; it copies and modifies the function library.
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;

  // true if the Session has been Closed.
  mutex closed_lock_;
  bool closed_ GUARDED_BY(closed_lock_) = false;

  // For generating unique names for this session instance.
  std::atomic<int64> edge_name_counter_ = {0};
  std::atomic<int64> handle_name_counter_ = {0};

  // For generating step ids that are unique among all sessions.
  static std::atomic_int_fast64_t step_id_counter_;

  // Global timeout for all blocking operations in this session.
  const int64 operation_timeout_in_ms_ = 0;

  // Manages all the cost models for the graphs executed in this session.
  CostModelManager cost_model_manager_;

  // For testing collective graph key generation.
  mutex collective_graph_key_lock_;
  int64 collective_graph_key_ GUARDED_BY(collective_graph_key_lock_) = -1;

  // Run in caller's thread if RunOptions.inter_op_thread_pool is negative or
  // all of following conditions are met:
  // 1. This session doesn't own any thread pool.
  // 2. RunOptions.inter_op_thread_pool is unspecified or 0.
  // 3. This session has a single executor.
  // 4. config.inter_op_parallelism_threads is specified to negative explicitly
  //    or through environment variable TF_NUM_INTEROP_THREADS.
  // 5. RunOptions.experimental.use_run_handler_pool is unspecified or false.
  // Otherwise run in global thread pool, session owned thread pool or handler
  // pool according to other specifications of RunOptions and ConfigProto.
  bool run_in_caller_thread_ = false;
  bool force_run_in_caller_thread_ = false;
  bool pai_enable_online_tuning_;

  std::unique_ptr<StreamGroupMgr> stream_group_mgr_;

  TF_DISALLOW_COPY_AND_ASSIGN(DirectSession);

  // EXPERIMENTAL: debugger (tfdbg) related
  friend class DebugGateway;
};

class BlazeConfSingleton {
 public:
  static BlazeConfSingleton* GetInstance() {
    mutex_lock l(mu_);
    static BlazeConfSingleton* sig = new BlazeConfSingleton;
    return sig;
  }

  void Set(const ConfigProto& tf_options) {
    mutex_lock l(mu_);
    if (!setted_) {
      run_config_ = tf_options;
      setted_ = true;
      VLOG(0) << "Blaze will use globla_opts : " << run_config_.DebugString();
    }
  }

  const ConfigProto& GetConfig() {
    return run_config_;
  }

  bool Setted() {
    return setted_;
  }

 private:
  BlazeConfSingleton() {};

 private:
  static mutex mu_;
  bool setted_ = false;;
  ConfigProto run_config_;
};
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DIRECT_SESSION_H_
