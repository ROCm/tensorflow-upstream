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
#include "nccl_manager.h"

#include <iostream>
#include <string>
#include <utility>

#if defined(USE_ROCM)
#include "absl/base/call_once.h"
#include "hip/hip_runtime.h"
#include "tensorflow/core/platform/rocm.h"
#else
#include "cuda_runtime.h"
#include "tensorflow/core/platform/cuda.h"
#endif

#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/env.h"

namespace jaguar {

namespace {
const auto STATUS_OK = absl::Status();
}

#if defined(USE_ROCM)
using stream_executor::gpu::ScopedActivateContext;
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaSuccess hipSuccess
int NcclManager::instance_count = 0;
#else
using tensorflow::se::cuda::ScopedActivateExecutorContext;
#endif

#define NCCL_RETURN_IF_ERROR(...)                                           \
  do {                                                                      \
    ncclResult_t nccl_status = (__VA_ARGS__);                               \
    if (nccl_status != ncclSuccess) {                                       \
      return tensorflow::errors::Internal(ncclGetErrorString(nccl_status)); \
    }                                                                       \
  } while (0)

#define CUDA_RETURN_IF_ERROR(...)                                           \
  do {                                                                      \
    cudaError_t cuda_status = (__VA_ARGS__);                                \
    if (cuda_status != cudaSuccess) {                                       \
      return tensorflow::errors::Internal(cudaGetErrorString(cuda_status)); \
    }                                                                       \
  } while (0)

#define CUDA_VLOG_IF_ERROR(...)                   \
  do {                                            \
    cudaError_t cuda_status = (__VA_ARGS__);      \
    if (cuda_status != cudaSuccess) {             \
      VLOG(0) << cudaGetErrorString(cuda_status); \
    }                                             \
  } while (0)

// Contains data for a single stream used for nccl communication; this includes
// a background thread that calls NcclManager::LoopKernelLaunches.
struct NcclManager::NcclStream : public tensorflow::core::RefCounted {
 public:
  NcclStream() = default;
  ~NcclStream() = default;

  tensorflow::se::StreamExecutor* executor = nullptr;

  // The stream on which to run the nccl collective.
  // This is a different stream than the tensorflow compute stream.
#if defined(USE_ROCM)
  // On ROCm, we borrow the nccl stream from the device context.
  tensorflow::se::Stream* stream = nullptr;
#else
  std::unique_ptr<tensorflow::se::Stream> stream;
#endif

  // `mu` protects access to `pending_launches_`, which is the list of
  // collectives ready but whose kernels are yet to be launched.  When the
  // NcclManager object that owns this NcclStream object is destroyed, it
  // signals `cv` to unblock the thread waiting on more collectives.
  tensorflow::mutex mu;
  tensorflow::condition_variable cv;
  // Has (collective, participant_idx) pairs.
#ifdef USE_TF215
  std::deque<std::pair<Collective*, int>> pending_launches_ TF_GUARDED_BY(mu);
  bool shutdown_requested TF_GUARDED_BY(mu) = false;
#else
  std::deque<std::pair<Collective*, int>> pending_launches_ TF_GUARDED_BY(mu);
  bool shutdown_requested TF_GUARDED_BY(mu) = false;
#endif
};

struct NcclManager::CommunicatorMember {
 public:
  CommunicatorMember() {}
  ~CommunicatorMember() {
    if (nccl_comm != nullptr) ncclCommDestroy(nccl_comm);
  }

  ncclComm_t nccl_comm = nullptr;
  // Owned by NcclManager::device_to_comm_streams_ and LoopKernelLaunches.
  NcclStream* nccl_stream = nullptr;
};

struct NcclManager::Communicator {
 public:
  explicit Communicator(std::vector<CommunicatorMember> members,
                        const tensorflow::string& key)
      : num_devices(members.size()), members(std::move(members)), key(key) {}

  const int num_devices;
  const std::vector<CommunicatorMember> members;
  const tensorflow::string key;
};

namespace {

ncclDataType_t ToNcclType(tensorflow::DataType t) {
  switch (t) {
    case tensorflow::DT_HALF:
      return ncclHalf;
#ifndef DISABLE_BFLOAT16
    case tensorflow::DT_BFLOAT16:
      return ncclBfloat16;
#endif
    case tensorflow::DT_FLOAT:
      return ncclFloat;
    case tensorflow::DT_DOUBLE:
      return ncclDouble;
    case tensorflow::DT_INT32:
      return ncclInt;
    case tensorflow::DT_INT64:
      return ncclInt64;
    default:
      return ncclFloat;
  }
}

std::size_t NcclTypeSize(ncclDataType_t t) {
  switch (t) {
    case ncclHalf:
      return 2;
#ifndef DISABLE_BFLOAT16
    case ncclBfloat16:
      return 2;
#endif

    case ncclFloat:
      return sizeof(float);
    case ncclDouble:
      return sizeof(double);
    case ncclInt:
      return sizeof(int32_t);
    case ncclInt64:
      return sizeof(int64_t);
    default:
      return sizeof(float);
  }
}

void StringToNcclUniqueId(const tensorflow::string& str_id,
                          ncclUniqueId* nccl_id) {
  if (str_id.size() == NCCL_UNIQUE_ID_BYTES) {
    memcpy(nccl_id->internal, str_id.data(), NCCL_UNIQUE_ID_BYTES);
  }
}

struct netIf {
  char prefix[64];
  int port;
};

union socketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

struct ncclBootstrapHandle {
  uint64_t magic;
  union socketAddress addr;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  if (!string) return 0;

  // Ignore 'NCCL_COMM_ID='
  const char* ptr = string + NCCL_COMM_ID_LEN;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr + 1);
        ifNum++;
        ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++;
        ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

ncclResult_t GetNcclUniqueIdFromString(ncclUniqueId* id, const char* comm_id) {
  memset(id, 0, sizeof(ncclUniqueId));
  auto handler = (struct ncclBootstrapHandle*)(id);
  std::string str(comm_id);
  std::cout << "comm_id :" << comm_id << std::endl;
  size_t end = str.find('+');
  handler->magic = std::stoull(str.substr(end + 1, str.size()));
  auto str_ip_port = str.substr(0, end);
  const char* ip_port_pair = str_ip_port.c_str();
  union socketAddress* ua = &(handler->addr);

  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    std::cout << "Net : string is null" << std::endl;
    return ncclInvalidArgument;
  }

  std::cout << "Net address :" << ip_port_pair << std::endl;
  bool ipv6 = ip_port_pair[NCCL_COMM_ID_LEN] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      std::cout << "Net : No valid <IPv4_or_hostname>:<port> pair found"
                << std::endl;

      return ncclInvalidArgument;
    }
    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ((rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      std::cout << "Net : error encountered when getting address info : "
                << gai_strerror(rv) << std::endl;
      return ncclInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in& sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;  // IPv4
      // inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port);  // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6& sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;      // IPv6
      sin6.sin6_port = htons(ni.port);  // port
      sin6.sin6_flowinfo = 0;           // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;           // should be global scope, set to 0
    } else {
      VLOG(0) << "Net : unsupported IP family";
      return ncclInvalidArgument;
    }

    freeaddrinfo(p);  // all done with this structure

  } else {
    // Ignore 'NCCL_COMM_ID='
    const char* ptr = ip_port_pair + NCCL_COMM_ID_LEN;
    int i, j = -1, len = strlen(ptr);
    for (i = 1; i < len; i++) {
      if (ptr[i] == '%') j = i;
      if (ptr[i] == ']') break;
    }

    if (i == len) {
      std::cout << "Net : No valid [IPv6]:port pair found" << std::endl;
      return ncclInvalidArgument;
    }
    bool global_scope =
        (j == -1
             ? true
             : false);  // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ptr + 1, global_scope ? i - 1 : j - 1);
    strncpy(port_str, ptr + i + 2, len - i - 1);
    int port = atoi(port_str);
    if (!global_scope)
      strncpy(if_name, ptr + j + 1,
              i - j - 1);  // If not global scope, we need the intf name

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                     // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));  // IP address
    sin6.sin6_port = htons(port);                    // port
    sin6.sin6_flowinfo = 0;  // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id =
        global_scope
            ? 0
            : if_nametoindex(
                  if_name);  // 0 if global scope; intf index if link scope
  }
  return ncclSuccess;
}

const char* socketToString(struct sockaddr* saddr, char* buf) {
  if (buf == NULL || saddr == NULL) return NULL;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) {
    buf[0] = '\0';
    return buf;
  }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  (void)getnameinfo(saddr, sizeof(union socketAddress), host, NI_MAXHOST,
                    service, NI_MAXSERV, NI_NUMERICHOST | NI_NUMERICSERV);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

// nccl op sequential
std::atomic_int NCCL_KERNEL_LAUNCH_SEQ(0);
std::array<tensorflow::se::Stream*, 5> NCCL_STREAM;
class IncrSeqHook {
 public:
  IncrSeqHook(tensorflow::se::Stream* s, int32_t seq_launch_len,
              int32_t seq_launch_idx)
      : seq_launch_idx(seq_launch_idx) {
    stream = s;
    seq_launch_next =
        (seq_launch_idx == seq_launch_len - 1 ? 0 : seq_launch_idx + 1);
  }
  ~IncrSeqHook() {
    NCCL_STREAM[seq_launch_idx] = stream;
    int32_t expected = seq_launch_idx;
    if (!NCCL_KERNEL_LAUNCH_SEQ.compare_exchange_strong(expected,
                                                        seq_launch_next)) {
      LOG(FATAL) << "invalid op_seq " << NCCL_KERNEL_LAUNCH_SEQ
                 << " vs expected " << seq_launch_idx;
    }
    // LOG(INFO) << "reset seq to " << seq_launch_next;
  }

 private:
  tensorflow::se::Stream* stream;
  int32_t seq_launch_idx;
  int32_t seq_launch_next;
};

#define WAIT_FOR_KERNEL_LAUNCH_SEQ(stream, seq_launch_len, seq_launch_idx) \
  std::unique_ptr<IncrSeqHook> __TMP_INCR_SEQ_HOOK__ = nullptr;            \
  do {                                                                     \
    if (seq_launch_len) {                                                  \
      while (seq_launch_idx != NCCL_KERNEL_LAUNCH_SEQ) {                   \
      }                                                                    \
      if (seq_launch_idx) {                                                \
        stream->WaitFor(NCCL_STREAM[seq_launch_idx - 1]).IgnoreError();    \
      }                                                                    \
      __TMP_INCR_SEQ_HOOK__ = std::make_unique<IncrSeqHook>(               \
          stream, seq_launch_len, seq_launch_idx);                         \
    }                                                                      \
  } while (0)

void ThreadSetNameOnce(const std::string& name) {
  thread_local bool name_set = false;
  if (name_set) {
    return;
  }
  auto ret = pthread_setname_np(pthread_self(), name.c_str());
  if (ret != 0) {
    LOG(WARNING) << "ThreadSetName failed for " << name;
    return;
  }
  name_set = true;
}

}  // namespace

// A `Collective` encapsulates state for a collective instance at one node.
// Typically, an instance in TensorFlow context would be defined by a collective
// group and the (step, frame iteration) for that execution.
//
// For each collective instance there will be one `Collective` object per node.
// For example,  a NCCL collective that runs on a single node with 4 GPUs would
// have a single `Collective` per step.  However, a collective that executes on
// 3 nodes with 4 GPUs each would have a `Collective` per node, each of which is
// tracking the 4 GPUs local to that node.
struct NcclManager::Collective : public tensorflow::core::RefCounted {
  Collective(const tensorflow::string& collective_key_in,
             tensorflow::DataType data_type_in, CollectiveType type_in,
             ncclRedOp_t reduction_op_in, int num_local_devices_in,
             int num_global_devices_in,
             const tensorflow::string& communicator_key_in,
             int seq_launch_len = 0, int seq_launch_idx = 0)

      : collective_key(collective_key_in),
        data_type(data_type_in),
        type(type_in),
        reduction_op(reduction_op_in),
        num_local_devices(num_local_devices_in),
        num_global_devices(num_global_devices_in),
        single_node(num_local_devices_in == num_global_devices_in),
        communicator_key(communicator_key_in),
        seq_launch_len(seq_launch_len),
        seq_launch_idx(seq_launch_idx) {
    participants.reserve(num_local_devices_in);
#if USE_ROCM
    // On ROCm platform, this allows caller to either use the singleton instance
    // or to manage one non-singleton NcclManager instance.
    // For example, the nccl_manager_test will use both paradigms in the same
    // executable, but not running concurrently (which would hang otherwise).
    if (NcclManager::instance_count > 1) {
      status = tensorflow::errors::Internal(
          "ROCm cannot use multi-node NCCL collectives on a single node");
    }
#endif
  }

  const tensorflow::string collective_key;  // A unique key for debugging.
  const tensorflow::DataType data_type;
  const CollectiveType type;
  const ncclRedOp_t reduction_op;  // applies when <type> is a reduction.
  const int num_local_devices;     // devices local to this node
  const int num_global_devices;    // devices across all nodes
  const bool single_node;          // true if all devices are at one node
  const tensorflow::string communicator_key;

  Communicator* communicator = nullptr;

  // All collective participants.
  //
  // Adding values in this vector is guarded by the mutex of the containing
  // NcclManager.
  std::vector<std::unique_ptr<Participant>> participants;

  // For collective types that have a root (e.g. the root of broadcast is the
  // sender), this is the rank of the root.
  int root_rank = -1;
  // How many participants have been registered so far. The Collective is
  // eligible for running with <available_participants> == num_local_devices.
  //
  // If this is a multi-node collective, we additionally have to synchronize
  // across nodes.  The caller would need to signal multi node readiness by
  // calling NcclManager::SignalMultiNodeReady, which sets `multi_node_ready` to
  // true.
  //
  // Guarded by the mutex of the containing Communicator.
  int available_participants = 0;
  bool multi_node_ready = false;

  tensorflow::Status status;

  // if set to greater than zero, collective from all streams will be launched
  // one by one according to launch idx
  int32_t seq_launch_len = 0;
  int32_t seq_launch_idx = 0;
};

NcclManager::NcclManager() {
  VLOG(2) << "New NcclManager " << this;
#if USE_ROCM
  ++instance_count;
#endif
  char* env = getenv("JAGUAR_LOCAL_RANKS");
  if (env) {
    local_ranks_ = atoi(env);
  }
  env = getenv("JAGUAR_WORKER_INDEX");
  if (env) {
    worker_index_ = atoi(env);
  }
  env = getenv("JAGUAR_WORKER_COUNT");
  if (env) {
    worker_count_ = atoi(env);
  }
  global_ranks_ = local_ranks_ * worker_count_;
}

NcclManager::~NcclManager() {
  VLOG(2) << "~NcclManager " << this;
#if USE_ROCM
  --instance_count;
#endif
  for (auto& it : device_to_comm_streams_) {
    for (NcclStream* nccl_stream : it.second) {
      {
        tensorflow::mutex_lock l(nccl_stream->mu);
        nccl_stream->shutdown_requested = true;
        nccl_stream->cv.notify_all();
      }
      nccl_stream->Unref();
    }
  }
}

NcclManager* NcclManager::instance() {
  static NcclManager* instance = new NcclManager();
#if USE_ROCM
  // singleton does not count against total instances
  // see comment above in Collective constructor concerning ROCm platform
  static absl::once_flag once;
  absl::call_once(once, [] { --NcclManager::instance_count; });
#endif
  return instance;
}

tensorflow::string NcclManager::GenerateCommunicatorKey(char* nccl_comm_id,
                                                        int rank,
                                                        bool init_step) {
  tensorflow::mutex_lock l(mu_);
  ncclUniqueId nccl_id;
  if (init_step && rank == -1) {
    VLOG(0) << "ERROR: NcclManager::GenerateCommunicatorKey ncclGetUniqueId "
               "should never got called";
    putenv(nccl_comm_id);
    ncclGetUniqueId(&nccl_id);
  } else {
    GetNcclUniqueIdFromString(&nccl_id, nccl_comm_id);
  }
  return tensorflow::string(nccl_id.internal, NCCL_UNIQUE_ID_BYTES);
}

int NcclManager::LocalRanks() { return local_ranks_; }

int NcclManager::WorkerIndex() { return worker_index_; }

int NcclManager::WorkerCount() { return worker_count_; }

int NcclManager::GlobalRanks() { return global_ranks_; }

tensorflow::Status NcclManager::CreateCommunicator(
    tensorflow::se::StreamExecutor* executor,
#ifdef USE_TF215
    const tensorflow::DeviceBase::AcceleratorDeviceInfo* info,
#else
    const tensorflow::DeviceBase::GpuDeviceInfo* info,
#endif
    const tensorflow::string& communicator_key, int global_rank) {
  if (LocalRanks() != 1) {
    return tensorflow::errors::Internal(
        "NcclManager::CreateCommunicator only support LocalRanks=1");
  }
  tensorflow::mutex_lock l(mu_);
  if (communicator_key.size() != NCCL_UNIQUE_ID_BYTES) {
    return tensorflow::errors::Internal(
        "Expected communicator_key of size ", NCCL_UNIQUE_ID_BYTES,
        " but found size ", communicator_key.size());
  }

  // This is an instance of multi-node collective.  We have previously
  // created a NCCL unique id and shared with all workers.  Now we find the
  // `Communicator` corresponding to this id.
  for (auto& comm : communicators_) {
    if (comm->key == communicator_key) {
      return STATUS_OK;
    }
  }

  auto* env = tensorflow::Env::Default();
  // Create and initialize a new communicator.
  // Note that this is done under the lock; performance is not expected to
  // matter as this happens a very small number of times.
  std::vector<CommunicatorMember> members(LocalRanks());
  int device_id = info->gpu_id;

  // Find a communication stream to use for the device.
  auto& streams = device_to_comm_streams_[executor];
  NcclStream* nccl_stream;
  nccl_stream = new NcclStream();
  nccl_stream->executor = executor;
  VLOG(2) << "Create new stream";
#if USE_ROCM
  auto stream_or_status = executor->CreateStream();
  nccl_stream->stream = stream_or_status->release();
#else
  nccl_stream->stream.reset(new tensorflow::se::Stream(executor));
  nccl_stream->stream->Init();
#endif

  streams.emplace_back(nccl_stream);
  // used_streams.insert(nccl_stream);

  nccl_stream->Ref();
  env->SchedClosure([this, nccl_stream]() {
    LoopKernelLaunches(nccl_stream);
    nccl_stream->Unref();
  });

  members[0].nccl_stream = nccl_stream;

  ncclComm_t nccl_comm;
  VLOG(2) << "Create new communicator";

  // For NCCL 2, we always initialize using ncclCommInitRank guarded by NCCL
  // group primitives.
  ncclUniqueId nccl_id;
  StringToNcclUniqueId(communicator_key, &nccl_id);
  char line_a[SOCKET_NAME_MAXLEN + 1];
  union socketAddress& addr = ((struct ncclBootstrapHandle*)(&nccl_id))->addr;
  VLOG(2) << "Try to init rank " << global_rank << " "
          << socketToString(&(addr.sa), line_a);
  VLOG(2) << "NCCL_COMM_ID " << getenv("NCCL_COMM_ID");
  int saved_device = 0;
  CUDA_RETURN_IF_ERROR(cudaGetDevice(&saved_device));
  NCCL_RETURN_IF_ERROR(ncclGroupStart());
  CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
  NCCL_RETURN_IF_ERROR(
      ncclCommInitRank(&nccl_comm, GlobalRanks(), nccl_id, global_rank));
  NCCL_RETURN_IF_ERROR(ncclGroupEnd());

  CUDA_RETURN_IF_ERROR(cudaSetDevice(saved_device));

  members[0].nccl_comm = nccl_comm;

  communicators_.emplace_back(
      new Communicator(std::move(members), communicator_key));
  return STATUS_OK;
}

tensorflow::Status NcclManager::GetCommunicator(
    NcclManager::Collective* collective,
    NcclManager::Communicator** communicator) {
  // Sort by global rank to make ordering of participants deterministic.
  std::sort(collective->participants.begin(), collective->participants.end(),
            [](const std::unique_ptr<Participant>& a,
               const std::unique_ptr<Participant>& b) {
              if (a->gpu_device_id != b->gpu_device_id) {
                return a->gpu_device_id < b->gpu_device_id;
              }
              if (a->executor != b->executor) {
                return a->executor < b->executor;
              }
              return a->global_rank < b->global_rank;
            });

  tensorflow::mutex_lock l(mu_);

  if (collective->communicator_key.empty()) {
    // For single-node collectives, when the caller does not specify a
    // `communicator_key`, we identify a communicator uniquely by the set of
    // devices participating in the collective.  For example, if a collective is
    // for GPUs 0, 1, and 2 then this will scan to find the communicator for
    // GPUs 0, 1, and 2.
    //
    // Note that each executor identifies a context on one device, so this is
    // the same as getting the communicator connecting the devices in the
    // collective. A device can be in different communicators as well - for
    // example, a communicator for GPUs 0 and 1 is separate from one for GPUs 0,
    // 1, and 2.
    //
    // Since it's expected that a small number of distinct communicators will
    // be needed, communicators_ is not garbage collected currently.
    //
    // Launching of kernels must be serialized so that, given collectives A and
    // B, and an order of them (e.g., A before B), then for each comm_stream
    // involved, the kernel for A is launched before the kernel for B. This is
    // guaranteed currently be a global mutex controlling additions of the
    // kernels to per-stream launch queues.  The launch queues are processed by
    // LoopKernelLaunches.
    for (auto& comm : communicators_) {
      if (comm->num_devices == collective->num_global_devices) {
        int i;
        for (i = 0; i < collective->num_local_devices; ++i) {
          if (comm->members[i].nccl_stream->executor !=
              collective->participants[i]->executor) {
            break;
          }
        }
        if (i == collective->num_local_devices) {
          *communicator = comm.get();
          return STATUS_OK;
        }
      }
    }
  } else {
#if NCCL_MAJOR < 2
    return tensorflow::errors::Internal(
        "Cannot use multi-node NCCL collectives with NCCL 1.x");
#endif
    if (collective->communicator_key.size() != NCCL_UNIQUE_ID_BYTES) {
      return tensorflow::errors::Internal(
          "Expected communicator_key of size ", NCCL_UNIQUE_ID_BYTES,
          " but found size ", collective->communicator_key.size());
    }
    // This is an instance of multi-node collective.  We have previously
    // created a NCCL unique id and shared with all workers.  Now we find the
    // `Communicator` corresponding to this id.
    for (auto& comm : communicators_) {
      if (comm->key == collective->communicator_key) {
        *communicator = comm.get();
        return STATUS_OK;
      }
    }
  }
  VLOG(0) << "ERROR: NcclManager lazy ncclInitRank should never got called";
  auto* env = tensorflow::Env::Default();
  std::set<NcclStream*> used_streams;
  // Create and initialize a new communicator.
  // Note that this is done under the lock; performance is not expected to
  // matter as this happens a very small number of times.
  std::vector<CommunicatorMember> members(collective->num_local_devices);
  std::vector<int> devices(collective->num_local_devices);
  for (int i = 0; i < collective->num_local_devices; ++i) {
    auto* executor = collective->participants[i]->executor;

    // Find a communication stream to use for the device.
    auto& streams = device_to_comm_streams_[executor];
    NcclStream* nccl_stream = nullptr;
    // for (const auto& s : streams) {
    //  if (used_streams.insert(s).second) {
    //    nccl_stream = s;
    //    break;
    //  }
    //}
    if (nccl_stream == nullptr) {
      nccl_stream = new NcclStream();
      nccl_stream->executor = executor;
      VLOG(2) << "Create new stream";
#if USE_ROCM
      nccl_stream->stream = collective->participants[i]->context->nccl_stream();
#else
      nccl_stream->stream.reset(new tensorflow::se::Stream(executor));
      nccl_stream->stream->Init();
#endif

      streams.emplace_back(nccl_stream);
      // used_streams.insert(nccl_stream);

      nccl_stream->Ref();

      env->SchedClosure([this, nccl_stream]() {
        LoopKernelLaunches(nccl_stream);
        nccl_stream->Unref();
      });
    }

    members[i].nccl_stream = nccl_stream;
    devices[i] = collective->participants[i]->gpu_device_id;
  }

  std::vector<ncclComm_t> nccl_comms(collective->num_local_devices);
  VLOG(2) << "Create new communicator";
  // For NCCL 2, we always initialize using ncclCommInitRank guarded by NCCL
  // group primitives.
  ncclUniqueId nccl_id;
  StringToNcclUniqueId(collective->communicator_key, &nccl_id);
  char line_a[SOCKET_NAME_MAXLEN + 1];
  union socketAddress& addr = ((struct ncclBootstrapHandle*)(&nccl_id))->addr;
  VLOG(2) << "Try to init rank " << socketToString(&(addr.sa), line_a);
  int saved_device = 0;
  CUDA_RETURN_IF_ERROR(cudaGetDevice(&saved_device));
  NCCL_RETURN_IF_ERROR(ncclGroupStart());
  for (int i = 0; i < collective->num_local_devices; ++i) {
    // Set rank to `participant->global_rank` if provided, else `i`.
    const int rank = collective->participants[i]->global_rank >= 0
                         ? collective->participants[i]->global_rank
                         : i;
    CUDA_RETURN_IF_ERROR(cudaSetDevice(devices[i]));
    NCCL_RETURN_IF_ERROR(ncclCommInitRank(
        nccl_comms.data() + i, collective->num_global_devices, nccl_id, rank));
  }
  NCCL_RETURN_IF_ERROR(ncclGroupEnd());
  CUDA_RETURN_IF_ERROR(cudaSetDevice(saved_device));

  for (int i = 0; i < collective->num_local_devices; ++i) {
    members[i].nccl_comm = nccl_comms[i];
  }
  communicators_.emplace_back(
      new Communicator(std::move(members), collective->communicator_key));
  *communicator = communicators_.back().get();
  return STATUS_OK;
}

void NcclManager::AddToAllReduce(std::unique_ptr<Participant> participant,
                                 const Context& context,
                                 ncclRedOp_t reduction_op) {
  AddParticipant(std::move(participant), context, kAllReduce, reduction_op);
}

void NcclManager::AddToAllGather(std::unique_ptr<Participant> participant,
                                 const Context& context) {
  AddParticipant(std::move(participant), context, kAllGather,
                 ncclSum /* unused */);
}

void NcclManager::AddToAllToAll(std::unique_ptr<Participant> participant,
                                const Context& context) {
  AddParticipant(std::move(participant), context, kAllToAll,
                 ncclSum /* unused */);
}

void NcclManager::AddToBroadcast(std::unique_ptr<Participant> participant,
                                 const Context& context) {
  AddParticipant(std::move(participant), context, kBroadcast,
                 ncclSum /* unused */);
}

void NcclManager::AddParticipant(std::unique_ptr<Participant> participant,
                                 const Context& context,
                                 CollectiveType collective_type,
                                 ncclRedOp_t reduction_op) {
  Collective* to_run = nullptr;
  tensorflow::DataType data_type;
  if (participant->inputs.size() > 0) {
    data_type = participant->inputs[0]->dtype();
  } else if (participant->outputs.size() > 0) {
    data_type = participant->outputs[0]->dtype();
  }
  {
    tensorflow::mutex_lock l(mu_);
    auto collective_it = collectives_.find(context.collective_key);
    Collective* collective = nullptr;
    if (collective_it == collectives_.end()) {
      collective =
          new Collective(context.collective_key, data_type, collective_type,
                         reduction_op, context.num_local_devices,
                         context.num_global_devices, context.communicator_key,
                         context.seq_launch_len, context.seq_launch_idx);
      collectives_.emplace(context.collective_key, collective);
    } else {
      collective = collective_it->second;
    }

    // Check `collective` is correct and consistent.
    if (collective->status.ok() && !collective->single_node &&
        collective->communicator_key.empty()) {
      collective->status = tensorflow::errors::Internal(
          "Collective ", reduction_op, " is multi node with num_local_devices=",
          collective->num_local_devices,
          " and num_global_devices=", collective->num_global_devices,
          " but has an empty communicator_key");
    }
    if (collective->status.ok() && collective->communicator_key.size() !=
                                       context.communicator_key.size()) {
      collective->status = tensorflow::errors::Internal(
          "Collective ", reduction_op,
          " mismatch in member communicator_key with size ",
          collective->communicator_key.size(),
          " and arg communicator_key with size ",
          context.communicator_key.size());
    }
    if (collective->status.ok() && collective->type != collective_type) {
      collective->status = tensorflow::errors::Internal(
          "Collective ", reduction_op, " previously initialized with type ",
          collective->type, " but now got type ", collective_type);
    }
    if (collective->status.ok() &&
        collective->num_global_devices != context.num_global_devices) {
      collective->status = tensorflow::errors::Internal(
          "Collective ", reduction_op,
          " previously initialized with num_global_devices ",
          collective->num_global_devices, " but now got ",
          context.num_global_devices);
    }
    if (collective->status.ok() &&
        collective->num_local_devices != context.num_local_devices) {
      collective->status = tensorflow::errors::Internal(
          "Collective ", reduction_op,
          "previously initialized with num_local_devices ",
          collective->num_local_devices, " but now got ",
          context.num_local_devices);
    }
    if (collective->status.ok() &&
        collective->participants.size() >= collective->num_local_devices) {
      collective->status = tensorflow::errors::Internal(
          "Collective ", reduction_op, " expected ",
          collective->num_local_devices, " participants but now has ",

          collective->participants.size(),

          " with one more participant being added");
    }
    if (collective->status.ok() && collective->root_rank >= 0 &&
        context.source_rank >= 0 &&
        collective->root_rank != context.source_rank) {
      collective->status = tensorflow::errors::Internal(
          "Collective ", collective->collective_key, " already has root_rank ",
          collective->root_rank, " but new participant has root_rank ",
          context.source_rank);
    }

    if (context.source_rank >= 0) {
      collective->root_rank = context.source_rank;
    }
    collective->participants.emplace_back(std::move(participant));
    ++collective->available_participants;

    if (CheckReady(context.collective_key, collective)) {
      to_run = collective;
    }
  }
  if (to_run != nullptr) RunCollective(to_run);
}

bool NcclManager::CheckReady(const tensorflow::string& collective_key,
                             Collective* collective) {
  if (collective->available_participants == collective->num_local_devices) {
    if (collective->num_global_devices == collective->num_local_devices ||
        collective->multi_node_ready) {
      // Ownership transferred to callee.
      collectives_.erase(collective_key);
      return true;
    }
  }
  return false;
}

void NcclManager::RunCollective(Collective* collective) {
  static tensorflow::mutex collective_mu(tensorflow::LINKER_INITIALIZED);

  tensorflow::Status status = collective->status;
  if (status.ok()) {
    status = GetCommunicator(collective, &collective->communicator);
  }

  for (int i = 0; status.ok() && i < collective->num_local_devices; ++i) {
    Participant* p = collective->participants[i].get();
    NcclStream* nccl_stream = collective->communicator->members[i].nccl_stream;
    CHECK(nccl_stream != nullptr);
    const int rank = p->global_rank >= 0 ? p->global_rank : i;

    if (p->inputs.size() > 0 && p->inputs[0] != nullptr) {
      // Wait to ensure that the kernel that produces the data in the input
      // tensor has finished running before the nccl kernel runs on the
      // communication stream.
      nccl_stream->stream->WaitFor(p->tensor_stream).IgnoreError();
    }
    if (p->root) {
      if (collective->root_rank == -1) {
        collective->root_rank = rank;
      } else if (collective->root_rank != rank) {
        status = tensorflow::errors::Internal(
            "Inconsistent root rank ", collective->root_rank, " and GPU id ",
            p->gpu_device_id, " rank ", rank, " also marked as root.");
      }
    }
    VLOG(2) << "RunCollective rank " << rank << " global_rank "
            << p->global_rank << " root_rank " << collective->root_rank;
  }

  if (status.ok() && collective->type == kBroadcast &&
      collective->root_rank < 0) {
    status = tensorflow::errors::Internal(
        "Root rank not indicated for collective ", collective->collective_key);
  }

  if (!status.ok()) {
    for (int i = 0; i < collective->num_local_devices; ++i) {
      collective->participants[i]->done_callback(status);
    }
    collective->Unref();
    return;
  }

  {
    // Allow only one collective at a time to queue kernels for launching. This
    // is to prevent collectives from deadlocking each other.
    // Note that it would be possible to run multiple collectives at once, if
    // they have non-intersecting sets of devices.
    tensorflow::mutex_lock l(collective_mu);
    for (int i = 0; i < collective->num_local_devices; ++i) {
      NcclStream* nccl_stream =
          collective->communicator->members[i].nccl_stream;
      tensorflow::mutex_lock l(nccl_stream->mu);

      nccl_stream->pending_launches_.push_front(std::make_pair(collective, i));
      // Ownership is shared between LoopKernelLaunches for each stream in this
      // collective.
      collective->Ref();
      nccl_stream->cv.notify_all();
    }
  }
  collective->Unref();
}

void NcclManager::LoopKernelLaunches(NcclStream* nccl_stream) {
#if USE_ROCM
  tensorflow::se::Stream* comm_stream = nccl_stream->stream;
  ScopedActivateContext scoped_context(nccl_stream->executor);
#else
  tensorflow::se::Stream* comm_stream = nccl_stream->stream.get();
  ScopedActivateExecutorContext scoped_context(nccl_stream->executor);
#endif
  const cudaStream_t* cu_stream = reinterpret_cast<const cudaStream_t*>(
      comm_stream->platform_specific_handle().stream);
  ThreadSetNameOnce("nccl_loop");

  while (true) {
    // Find collective to run.
    std::pair<Collective*, int> next_launch;
    {
      VLOG(3) << "Locking mutex nccl_stream " << nccl_stream;
      tensorflow::mutex_lock l(nccl_stream->mu);
      while (nccl_stream->pending_launches_.empty()) {
        if (nccl_stream->shutdown_requested) {
          // No work and shutdown requested, exit.
          return;
        }
        nccl_stream->cv.wait(l);
      }
      next_launch = nccl_stream->pending_launches_.back();
      nccl_stream->pending_launches_.pop_back();
    }

    // Launch the nccl kernel.
    Collective* collective = next_launch.first;
    ncclDataType_t data_type = ToNcclType(collective->data_type);
    int p_idx = next_launch.second;
    Participant* p = collective->participants[p_idx].get();
    auto nccl_comm = collective->communicator->members[p_idx].nccl_comm;
    ncclResult_t nccl_result = ncclSuccess;

    // sequential launch
    WAIT_FOR_KERNEL_LAUNCH_SEQ(comm_stream, collective->seq_launch_len,
                               collective->seq_launch_idx);
    p->event_mgr->ThenExecute(comm_stream, [] {});
    std::string group_key = collective->collective_key;
    std::string metric_key;
    for (int i = 0; i < group_key.size(); i++) {
      if (group_key[i] == ';') {
        metric_key = group_key.substr(0, i);
        break;
      }
    }
#ifdef COMPILING_JAGUAR
    GlobalCudaTimerManager()->StartCudaTimer(metric_key, *cu_stream,
                                             jaguar::COMM);
#endif
    switch (collective->type) {
      case kAllReduce: {
        const void* sendbuff = p->inputs[0]->tensor_data().data();
        void* recvbuff = const_cast<char*>(p->outputs[0]->tensor_data().data());
        VLOG(2) << "call NcclAllReduce collective_key "
                << collective->collective_key << " participant " << p_idx
                << " sendbuff " << sendbuff << " recvbuff " << recvbuff
                << " nccl_comm " << nccl_comm << " comm_stream " << comm_stream
                << " cuda_stream " << cu_stream;
        nccl_result = ncclAllReduce(
            sendbuff, recvbuff, p->inputs[0]->NumElements(), data_type,
            collective->reduction_op, nccl_comm, *cu_stream);
        break;
      }
      case kBroadcast: {
        const void* sendbuff = nullptr;
        void* recvbuff = nullptr;
        int num_elements = -1;
        if (p->inputs.size() > 0) {
          sendbuff = p->inputs[0]->tensor_data().data();
          num_elements = p->inputs[0]->NumElements();
        }
        if (p->outputs.size() > 0) {
          recvbuff = const_cast<char*>(p->outputs[0]->tensor_data().data());
          num_elements = p->outputs[0]->NumElements();
        }
        if (num_elements < 0) {
          p->done_callback(tensorflow::errors::Internal(
              "Both input and output are null in ncclBroadcast"));
          collective->Unref();
          continue;
        }
        LOG(INFO) << "call NcclBroadcast collective_key "
                  << collective->collective_key << " participant " << p_idx
                  << " sendbuff " << sendbuff << " recvbuff " << recvbuff
                  << " num_elements " << num_elements
                  << " collective root_rank " << collective->root_rank
                  << " nccl_comm " << nccl_comm << " comm_stream "
                  << comm_stream << " cuda_stream " << cu_stream;
        nccl_result =
            ncclBroadcast(sendbuff, recvbuff, num_elements, data_type,
                          collective->root_rank, nccl_comm, *cu_stream);
        break;
      }
      case kReduce: {
        const void* sendbuff = p->inputs[0]->tensor_data().data();

        void* recvbuff =
            p->outputs.size() > 0
                ? const_cast<char*>(p->outputs[0]->tensor_data().data())
                : nullptr;
        nccl_result =
            ncclReduce(sendbuff, recvbuff, p->inputs[0]->NumElements(),
                       data_type, collective->reduction_op,
                       collective->root_rank, nccl_comm, *cu_stream);
        break;
      }
      case kAllGather: {
        const void* sendbuff = nullptr;
        void* recvbuff = nullptr;
        int send_num_elements = -1;
        int recv_num_elements = -1;
        if (p->inputs.size() > 0) {
          sendbuff = p->inputs[0]->tensor_data().data();
          send_num_elements = p->inputs[0]->NumElements();
        }
        if (p->outputs.size() > 0) {
          recvbuff = const_cast<char*>(p->outputs[0]->tensor_data().data());
          recv_num_elements = p->outputs[0]->NumElements();
        }

        VLOG(2) << "call NcclAllGather collective_key "
                << collective->collective_key << " participant " << p_idx
                << " sendbuff " << sendbuff << " sendcount "
                << send_num_elements << " recvbuff " << recvbuff
                << " recvcount " << recv_num_elements << " nccl_comm "
                << nccl_comm << " comm_stream " << comm_stream
                << " cuda_stream " << cu_stream;
        nccl_result = ncclAllGather(sendbuff, recvbuff, send_num_elements,
                                    data_type, nccl_comm, *cu_stream);
        break;
      }
      case kAllToAll: {
        VLOG(2) << "call NcclAlltoAll collective_key "
                << collective->collective_key << " participant " << p_idx
                << " nccl_comm " << nccl_comm << " comm_stream " << comm_stream
                << " cuda_stream " << cu_stream;

        int32_t total_sendcount = 0;
        int32_t total_recvcount = 0;
        ncclResult_t tmp_nccl_result = ncclSuccess;
        char* sendptr = const_cast<char*>(p->inputs[0]->tensor_data().data());
        char* recvptr = const_cast<char*>(p->outputs[0]->tensor_data().data());
        ncclGroupStart();
        for (int r = 0; r < collective->num_global_devices; ++r) {
          const void* sendbuff =
              sendptr + p->send_offsets[r] * NcclTypeSize(data_type);
          int32_t sendcount = p->send_counts[r];
          total_sendcount += sendcount;
          tmp_nccl_result = ncclSend(sendbuff, sendcount, data_type, r,
                                     nccl_comm, *cu_stream);
          if (tmp_nccl_result != ncclSuccess) nccl_result = tmp_nccl_result;
          void* recvbuff =
              recvptr + p->recv_offsets[r] * NcclTypeSize(data_type);
          int32_t recvcount = p->recv_counts[r];
          total_recvcount += recvcount;
          tmp_nccl_result = ncclRecv(recvbuff, recvcount, data_type, r,
                                     nccl_comm, *cu_stream);
          if (tmp_nccl_result != ncclSuccess) nccl_result = tmp_nccl_result;
        }
        nccl_result = tmp_nccl_result;
        ncclGroupEnd();
        if (collective->collective_key.substr(
                0, ALLTOALL_BACKWARD_PREFIX_LEN) == "Grad") {
        }
        break;
      }
    }
#ifdef COMPILING_JAGUAR
    GlobalCudaTimerManager()->StopCudaTimer(metric_key, *cu_stream);
    GlobalSessionStatus()->worker_status()->nccl_status.push(metric_key);
#endif
    // Run the done_callback when the nccl kernel finishes running.
    auto done_callback = [collective, p_idx, nccl_result, metric_key]() {
      VLOG(2) << "done Nccl kernel collective_key "
              << collective->collective_key << " participant " << p_idx
              << " ncclResult " << nccl_result;
      if (nccl_result == ncclSuccess) {
        collective->participants[p_idx]->done_callback(STATUS_OK);
      } else {
        // Propagate the error, but note that if other members of the collective
        // did launch their kernels, then they are hanging.
        collective->participants[p_idx]->done_callback(
            tensorflow::errors::Unknown("Error invoking NCCL: ",
                                        ncclGetErrorString(nccl_result)));
      }
      collective->Unref();
    };
    p->event_mgr->ThenExecute(comm_stream, done_callback);
  }
}

}  // namespace jaguar
