#include "tensorflow/core/graph/dist_graph_partition.h"

#include <stack>
#include <utility>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>

#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/framework/rendezvous.h"

using namespace std;

namespace tensorflow {

// Add an input to dst that comes from the "src_slot" output of the
// node named by "src_name".
void AddInput(NodeDef* dst, StringPiece src_name, int src_slot) {
  if (src_slot == Graph::kControlSlot) {
    dst->add_input(strings::StrCat("^", src_name));
  } else if (src_slot == 0) {
    dst->add_input(src_name.data(), src_name.size());
  } else {
    dst->add_input(strings::StrCat(src_name, ":", src_slot));
  }
}

std::string node_to_loc(const Node *node) {
  DeviceNameUtils::ParsedName p;
  if (!DeviceNameUtils::ParseFullName(node->def().device(), &p)) {
    return node->def().device();
  }
  if (p.has_job) {
    return p.job;
  }
  return "";
}

#define ADD_VAR_IF_NOT_EXIST(var_node) {          \
  int id = var_node->id();                        \
  if (var_id_set.find(id) == var_id_set.end()) {  \
    var_id_set.insert(id);                        \
    AddNode(var_node);                            \
  }                                               \
}

const Node* GetVarOfIdentity(const Node* identity_node) {
  for (const Edge* in : identity_node->in_edges()) {
    const Node* src = in->src();
    if (src->IsVariable()) {
      return src;
    }
  }

  return NULL;
}

void SubGraph::CompleteVariables(const PartitionOptions& opts) {
  set<int> var_id_set;
  int size = nodes_.size();
  for (int i = 0; i < size; i++) {
    auto node = nodes_[i];
    if (!node->IsOp()) continue;
    std::string loc = opts.node_to_loc(node);
    for (const Edge* edge : node->in_edges()) {
      Node* src = edge->src();
      if (!src->IsOp()) continue;
      if (src->IsVariable() && opts.node_to_loc(src) == loc) {
        ADD_VAR_IF_NOT_EXIST(src);
      }

      if (src->IsIdentity() && opts.node_to_loc(src) == loc) {
        const Node *var = GetVarOfIdentity(src);
        if (var == NULL) {
          continue;
        }

        ADD_VAR_IF_NOT_EXIST(src);
        ADD_VAR_IF_NOT_EXIST(var);
      }
    }
  }
}

GraphPartitionerBase::GraphPartitionerBase(
  const PartitionOptions& popts, Graph* g,
  bool zero_copy, bool use_fuse_recv,
  const std::function<bool (const std::string &)> &is_main_loc_func,
  SendRecvPolicy* user_define_policy)
  : opts_(popts), graph_(g),
    is_main_loc_func_(is_main_loc_func),
    zero_copy_(zero_copy),
    use_fuse_recv_(use_fuse_recv) {
  if (user_define_policy) {
    use_default_policy_ = false;
    send_recv_policy_ = user_define_policy;
  } else {
    use_default_policy_ = true;
    send_recv_policy_ = new DetectPsToPsCyclePolicy(graph_);
  }
  send_recv_policy_->AddUserDefinedPolicy();
}

bool GraphPartitionerBase::IsSplitedIgnored(
    const std::string &cur_loc,
    const Node* node)
{
  if (!node->IsOp()) {
    return true;
  }

  if (is_main_loc_func_(cur_loc)) {
    return false;
  }

  if (node->IsVariable()) {
    return true;
  }

  if (node->IsIdentity() && GetVarOfIdentity(node) != NULL) {
    return true;
  }

  return false;
}

void GraphPartitionerBase::TryAddEdgeNode(
    const std::string &cur_loc, const Node *node, deque<const Node*> &nodes,
    std::vector<bool> &node_visited, SubGraph &cur_sub_graph)
{
  if (IsSplitedIgnored(cur_loc, node)) {
    return;
  }
  std::string loc = opts_.node_to_loc(node);
  if (loc == cur_loc) {
    if (node_visited[node->id()]) {
      return;
    } else {
      cur_sub_graph.AddNode(node);
      node_visited[node->id()] = true;
      nodes.push_back(node);
    }
  }
}

int FindInputIdxInDef(const NodeDef *node_def, const Edge* in_edge) {
  int src_slot = in_edge->src_output();
  std::string src_name = in_edge->src()->name();
  if (in_edge->IsControlEdge()) {
    src_name = "^" + src_name;
  }
  for (int i = 0; i < node_def->input_size(); i++) {
    std::string input_name = node_def->input(i);
    size_t pos = input_name.rfind(":");
    if (pos == std::string::npos) {
      if (src_name == input_name
          && (src_slot == 0
          || src_slot == Graph::kControlSlot))
      {
        return i;
      }
    } else {
      std::string name = input_name.substr(0, pos);
      int idx;
      bool ret = strings::safe_strto32(input_name.substr(pos + 1), &idx);
      if (!ret) {
        LOG(ERROR) << "Failed to strto32 from:" << input_name
                   << ", pos:" << pos;

        return -1;
      }
      if (src_name == name && src_slot == idx) {
        return i;
      }
    }
  }

  LOG(ERROR) << "Can not find input idx in node:" << node_def->DebugString();
  return -1;
}

Status GraphPartitionerBase::ConstructNodeDef(const Node* node, NodeDef *node_def) {
  *node_def = node->def();
  node_def->set_device(node->assigned_device_name());
  node_def->clear_input();

  int num_inputs = 0;
  std::vector<const Edge*> inputs;
  inputs.resize(node->num_inputs(), nullptr);
  for (const Edge* in_edge : node->in_edges()) {
    if (in_edge->IsControlEdge()) {
      inputs.push_back(in_edge);
    } else {
      num_inputs++;
      inputs[in_edge->dst_input()] = in_edge;
    }
  }

  if (num_inputs != node->num_inputs()) {
    return errors::InvalidArgument("Incomplete graph, missing ",
            (node->num_inputs() - num_inputs),
            " inputs for ", node->def().DebugString());
  }

  for (const Edge* in_edge : inputs) {
    if (in_edge->src()->IsSource()) {
      continue;
    }
    AddInput(node_def, in_edge->src()->name(), in_edge->src_output());
  }

  return Status::OK();
}

Status GraphPartitionerBase::ConstructRecvNodeDef(const PartitionOptions &opts,
    const std::string &node_name,
    const std::string &send_device_name,
    const std::string &recv_device_name,
    const std::string &tensor_name,
    const DataType &tensor_type,
    bool client_terminated,
    NodeDef *node_def)
{
  return NodeDefBuilder(node_name, "_Recv").Device(recv_device_name)
      .Attr("tensor_type", tensor_type)
      .Attr("tensor_name", tensor_name)
      .Attr("send_device", send_device_name)
      .Attr("recv_device", recv_device_name)
      .Attr("send_device_incarnation",
            static_cast<int64>(opts.get_incarnation(send_device_name)))
      .Attr("client_terminated", client_terminated)
      .Finalize(node_def);
}

Status GraphPartitionerBase::ConstructSendNodeDef(
    const std::string &node_name,
    const std::string &send_device_name,
    const std::string &recv_device_name,
    const std::string &input_node_name,
    const int &input_idx,
    const std::string &tensor_name,
    const DataType &tensor_type,
    bool client_terminated,
    NodeDef *node_def)
{
  return NodeDefBuilder(node_name, "_Send").Device(send_device_name)
      .Input(input_node_name, input_idx, tensor_type)
      .Attr("tensor_name", tensor_name)
      .Attr("send_device", send_device_name)
      .Attr("recv_device", recv_device_name)
      .Attr("send_device_incarnation",
            static_cast<int64>(opts_.get_incarnation(send_device_name)))
      .Attr("client_terminated", client_terminated)
      .Finalize(node_def);
}

#define RETURN_IF_NOT_OK(expr)      \
  do {                              \
    auto __s = expr;                \
    if (!__s.ok()) {return __s;}    \
  } while(0)
    
Status CreateFeedAndFetchKey(const NodeDef &ndef, std::string *key) {
  bool client_terminated = false;
  RETURN_IF_NOT_OK(GetNodeAttr(ndef, "client_terminated",
                               &client_terminated));
  std::string name;
  RETURN_IF_NOT_OK(GetNodeAttr(ndef, "tensor_name", &name));
  std::string send_device;
  RETURN_IF_NOT_OK(GetNodeAttr(ndef, "send_device", &send_device));
  std::string recv_device;
  RETURN_IF_NOT_OK(GetNodeAttr(ndef, "recv_device", &recv_device));
  uint64 send_device_incarnation;
  RETURN_IF_NOT_OK(GetNodeAttr(ndef, "send_device_incarnation",
                               reinterpret_cast<int64*>(&send_device_incarnation)));
  *key = Rendezvous::CreateKey(send_device,
                               send_device_incarnation,
                               recv_device, name,
                               FrameAndIter(0, 0));
  // send_device == recv_device: RunGraph node
  // else: native Send/Recv node
  if ((send_device == recv_device && !client_terminated) ||
      (send_device != recv_device && client_terminated)) {
    return errors::Internal("node has a wrong client_terminated value.");
  }
    
  return Status::OK();
}


InputSrcKey MakeInputSrcKey(const Edge* in_edge) {
  return make_pair(in_edge->src()->name(), in_edge->src_output());
}

std::string MakeInputName(const std::string &src_name, int src_idx,
                          const std::string &delimeter=":") {
  if (src_idx == Graph::kControlSlot) {
    return strings::StrCat("^", src_name);
  } else {
    return strings::StrCat(src_name, delimeter, src_idx);
  }
}

std::string MakeInputName(const Edge* in_edge,
                          const std::string &delimeter=":") {
  return MakeInputName(in_edge->src()->name(), in_edge->src_output(), delimeter);
}

Status TrainGraphPartitioner::MakeInputSrcNode(
    const Edge *in_edge,
    NodeDef *src_node_def,
    std::string *feed_key)
{
  const std::string &src_node_name = in_edge->src()->name();
  const std::string &device_name = in_edge->dst()->assigned_device_name();

  std::string recv_node_name = strings::StrCat("_recv_", src_node_name, "__1");
  std::string src_tensor_name = strings::StrCat(src_node_name);

  auto data_type = DT_FLOAT;
  if (!in_edge->IsControlEdge()) {
    data_type = BaseType(in_edge->src()->output_type(in_edge->src_output()));
    recv_node_name = strings::StrCat("_recv_", src_node_name, "_", in_edge->src_output());
    src_tensor_name = strings::StrCat(src_node_name,
                                      ":", in_edge->src_output());
  }

  std::string send_device = device_name;
  std::string recv_device = device_name;
  bool client_terminated = true;
  if (ShouldUseSendRecvMode(in_edge->dst(), in_edge->src())) {
    send_device = in_edge->src()->assigned_device_name();
    client_terminated = false;
  }

  Status s = ConstructRecvNodeDef(opts_, recv_node_name,
                                  send_device,
                                  recv_device,
                                  src_tensor_name,
                                  data_type,
                                  client_terminated,
                                  src_node_def);
  RETURN_IF_NOT_OK(s);
  RETURN_IF_NOT_OK(CreateFeedAndFetchKey(*src_node_def, feed_key));
  return Status::OK();
}

void TrainGraphPartitioner::dealWithNodeOfNullptr(string *fetch_key,
        const Edge* out_edge)
{
  LOG(FATAL) << "Send node is nullptr.";
}

Status InferGraphPartitioner::MakeInputSrcNode(
    const Edge *in_edge,
    NodeDef *src_node_def,
    std::string *feed_key)
{
  auto src_tensor_name = MakeInputName(in_edge, "_");
  const std::string &device_name = in_edge->dst()->assigned_device_name();
  auto node_name = strings::StrCat("place_holder_for_", src_tensor_name);
  auto dtype = in_edge->dst()->input_type(in_edge->dst_input());

  auto s = NodeDefBuilder(node_name, "Placeholder")
           .Device(device_name)
           .Attr("dtype", dtype)
           .Finalize(src_node_def);
  *feed_key = src_node_def->name();
  return s;
}

void InferGraphPartitioner::dealWithNodeOfNullptr(string *fetch_key,
        const Edge* out_edge)
{
  *fetch_key = MakeInputName(out_edge);
}

void ResetInputs(const std::vector<std::string> &input_names,
                 NodeDef *node_def) {
  node_def->clear_input();
  std::vector<size_t> control_input_idxs;
  for (size_t i = 0; i < input_names.size(); i++) {
    const std::string &input_name = input_names[i];
    assert(!input_name.empty());
    if (input_name[0] == '^') {
      control_input_idxs.push_back(i);
      continue;
    }
    *node_def->add_input() = input_name;
  }

  for (size_t i = 0; i < control_input_idxs.size(); i++) {
    *node_def->add_input() = input_names[control_input_idxs[i]];
  }
}

void ResetInputs(const std::vector<std::pair<int, std::string>> &inputs, const Node* node,
                 NodeDef *node_def)
{
  node_def->clear_input();

  std::vector<std::string> input_names;
  input_names.resize(node->num_inputs());
  for(const auto& input : inputs) {
    if (input.first != Graph::kControlSlot) {
      input_names[input.first] = input.second;
    } else {
      input_names.push_back(input.second);
    }
  }
  for (const auto& name : input_names) {
    *node_def->add_input() = name;
  }
}

Status GraphPartitionerBase::ProcessSubNodeInputs(
    const Node* node,
    const std::string &loc,
    NodeDef *node_def,
    GraphDef *graph_def,
    std::map<InputSrcKey, NodeDef*> *input_src_nodes_map,
    std::vector<std::pair<std::string, const Edge*>> *boundary_input_edges,
    bool& use_send_recv,
    NodeDef* fuse_recv_node,
    std::unordered_map<std::string, int>& key_to_idx)
{
  std::vector<std::pair<int, std::string>> input_names;
  for (const Edge* in_edge : node->in_edges()) {
    const Node* src = in_edge->src();
    if (!src->IsOp()) {
      continue;
    }

    if (loc == opts_.node_to_loc(src)) {
      input_names.push_back({in_edge->dst_input(), MakeInputName(in_edge)});
      continue;
    }

    if (in_edge->IsControlEdge() &&
        !ShouldUseSendRecvMode(in_edge->dst(), in_edge->src())) {
        boundary_input_edges->push_back(make_pair("", in_edge));
      continue;
    }

    // Do not use fuse recv node
    if (ShouldUseSendRecvMode(in_edge->dst(), in_edge->src())) {
      std::string feed_key;
      auto src_node_def = DealWithSubNodeInputEdge(in_edge,
        input_src_nodes_map, &feed_key, graph_def);

      if (src_node_def != nullptr) {
        std::string input_name = MakeInputName(src_node_def->name(),
           in_edge->IsControlEdge() ? Graph::kControlSlot : 0);
        input_names.push_back({in_edge->dst_input(), input_name});

        if (!ShouldUseSendRecvMode(in_edge->dst(), in_edge->src())) {
          boundary_input_edges->push_back(make_pair(feed_key, in_edge));
        } else {
          use_send_recv = true;
        }
      }
    } else {
      // Fuse recv node
      const std::string &src_node_name = in_edge->src()->name();
      std::string src_tensor_name = strings::StrCat(src_node_name,":",
                                                    in_edge->src_output());
      const std::string &device_name = in_edge->dst()->assigned_device_name();
      const std::string& feed_key =
        Rendezvous::CreateKey(device_name,
                              static_cast<uint64>(opts_.get_incarnation(device_name)),
                              device_name,
                              src_tensor_name,
                              FrameAndIter(0, 0));
      if (key_to_idx.find(feed_key) == key_to_idx.end()) {
        LOG(FATAL) << "Feed key not found! feed_key=" << feed_key;
      }
      input_names.push_back({in_edge->dst_input(),
                             MakeInputName(fuse_recv_node->name(),
                                           key_to_idx[feed_key])});      
      boundary_input_edges->push_back(make_pair(feed_key, in_edge));
    }
  }

  ResetInputs(input_names, node, node_def);

  return Status::OK();
}

Status GraphPartitionerBase::ProcessSubNodeInputs(
    const Node* node,
    const std::string &loc,
    NodeDef *node_def,
    GraphDef *graph_def,
    std::map<InputSrcKey, NodeDef*> *input_src_nodes_map,
    std::vector<std::pair<std::string, const Edge*>> *boundary_input_edges,
    bool& use_send_recv)
{
  std::vector<std::pair<int, std::string>> input_names;
  for (const Edge* in_edge : node->in_edges()) {
    const Node* src = in_edge->src();
    if (!src->IsOp()) {
      continue;
    }

    if (loc == opts_.node_to_loc(src)) {
      input_names.push_back({in_edge->dst_input(), MakeInputName(in_edge)});
      continue;
    }

    // NOTE(jiankeng.pt) If use Send/Recv instead of RunGraph,
    // for control edge, also should add a related 'Recv' node,
    if (in_edge->IsControlEdge() &&
        !ShouldUseSendRecvMode(in_edge->dst(), in_edge->src())) {
        boundary_input_edges->push_back(make_pair("", in_edge));
      continue;
    }

    std::string feed_key;
    auto src_node_def = DealWithSubNodeInputEdge(in_edge,
        input_src_nodes_map, &feed_key, graph_def);
        
    if (src_node_def != nullptr) {
      std::string input_name = MakeInputName(src_node_def->name(),
          in_edge->IsControlEdge() ? Graph::kControlSlot : 0);
      input_names.push_back({in_edge->dst_input(), input_name});

      if (!ShouldUseSendRecvMode(in_edge->dst(), in_edge->src())) {
        boundary_input_edges->push_back(make_pair(feed_key, in_edge));
      } else {
        use_send_recv = true;
      }
    }
  }

  ResetInputs(input_names, node, node_def);

  return Status::OK();
}

Status GraphPartitionerBase::ProcessSubNodeOutputs(
    const Node* node,
    const std::string &loc,
    NodeDef *node_def,
    GraphDef *graph_def,
    std::vector<std::pair<std::string, const Edge*>> *boundary_output_edges,
    bool& use_send_recv)
{
  std::unordered_map<std::string, NodeDef*> send_nodes_map;
  std::unordered_map<std::string, int> send_nodes_index;
 
  for (const Edge* out_edge : node->out_edges()) {
    const Node* dst = out_edge->dst();
    if (!dst->IsOp()) {
      continue;
    }

    if (loc == opts_.node_to_loc(dst)) {
      continue;
    }

    // NOTE(jiankeng.pt) If use Send/Recv instead of RunGraph,
    // for control edge, also should add a related 'Send' node,
    // and a dummy 'Const' node.
    if (out_edge->IsControlEdge() &&
        !ShouldUseSendRecvMode(out_edge->src(), out_edge->dst())) {
        boundary_output_edges->push_back(make_pair("", out_edge));
      continue;
    }

    NodeDef *send_node_def = DealWithSubNodeOutputEdge(out_edge,
        send_nodes_map, send_nodes_index, graph_def);
    std::string fetch_key;
    if (send_node_def != nullptr) {
      Status s = CreateFeedAndFetchKey(*send_node_def, &fetch_key);
      RETURN_IF_NOT_OK(s);
    } else {
      // NOTE(jiankeng.pt): Send node is nullptr, error.
      dealWithNodeOfNullptr(&fetch_key, out_edge);
    }

    if (!ShouldUseSendRecvMode(out_edge->src(), out_edge->dst())) {
      boundary_output_edges->push_back(make_pair(fetch_key, out_edge));
    } else {
      use_send_recv = true;
    }
  }

  return Status::OK();
}

bool GraphPartitionerBase::ShouldUseSendRecvMode(Node* src, Node* dst) {
  return send_recv_policy_->UseSendRecvMode(src, dst);
}

bool GraphPartitionerBase::UseFuseRecv(SubGraph &sub_graph, int graph_idx,
                                       NodeDef** new_fuse_recv_node,
                                       std::unordered_map<std::string, int>& src_to_slot) {
  // Use fuse recv or not
  if (use_fuse_recv_) {
    static bool should_logging = true;
    if (should_logging) {
      should_logging = false;
      LOG(INFO) << "Use fuse recv for run graph op.";
    }
    std::unordered_map<std::string, const Edge*> name_to_nodes;
    std::vector<const Edge*> fuse_recv_edges;
    const std::string &loc = sub_graph.GetLoc();

    for (const Node* node: sub_graph.GetNodes()) {
      if (node->IsVariable() && !sub_graph.IsOnlyVariable()) {
        continue;
      }

      if (node->IsIdentity() &&
          GetVarOfIdentity(node) != NULL &&
          !sub_graph.IsOnlyVariable()) {
        continue;
      }

      for (const Edge* in_edge : node->in_edges()) {
        if (in_edge->dst()->assigned_device_name().find(loc) == std::string::npos) {
          LOG(FATAL) << "Location not mathed! " << loc << ", "
                     << in_edge->dst()->assigned_device_name();
        }
        const Node* src = in_edge->src();
        if (!src->IsOp() ||
            loc == opts_.node_to_loc(src) ||
            in_edge->IsControlEdge()) {
          continue;
        }
        // TODO: send/recv mode can not be fused
        if (!ShouldUseSendRecvMode(in_edge->dst(), in_edge->src())) {
          const std::string& name = strings::StrCat(in_edge->src()->name(), ":", in_edge->src_output());
          if (name_to_nodes.find(name) != name_to_nodes.end()) {
            if (name_to_nodes[name]->src() != in_edge->src() ||
                name_to_nodes[name]->dst()->assigned_device_name() !=
                in_edge->dst()->assigned_device_name()) {
              LOG(FATAL) << "Error edge info, pointer is not matched! node name=" << name
                         << ", name_to_nodes[name]->src()=" << name_to_nodes[name]->src()->DebugString()
                         << ", in_edge->src()=" << in_edge->src()->DebugString()
                         << ", name_to_nodes[name]->dst()=" << name_to_nodes[name]->dst()->DebugString()
                         << ", in_edge->dst()=" << in_edge->dst()->DebugString();
            }
          } else {
            name_to_nodes[name] = in_edge;
            fuse_recv_edges.push_back(in_edge);
          }
        }
      }
    }

    int fuse_count = fuse_recv_edges.size();
    if (fuse_count > 0) {
      // TODO: (host_memory) ? "_HostFuseRecv" : "_FuseRecv";
      const std::string fuse_recv_op = "_FuseRecv";
      std::string fuse_recv_node_name = strings::StrCat("fuse_recv_", graph_idx);
      NodeDefBuilder fuse_recv_builder(opts_.new_name(fuse_recv_node_name),
                                       fuse_recv_op);

      std::vector<std::string> tensor_names(fuse_count);
      std::vector<std::string> send_devices(fuse_count);
      std::vector<int64> send_device_incarnations(fuse_count);
      std::vector<DataType> cast_dtypes(fuse_count);
      for (int i = 0; i < fuse_count; ++i) {
        auto in_edge = fuse_recv_edges[i];
        tensor_names[i] = strings::StrCat(in_edge->src()->name(),
                                          ":", in_edge->src_output());
        send_devices[i] = in_edge->dst()->assigned_device_name();
        send_device_incarnations[i] = 
          static_cast<int64>(opts_.get_incarnation(send_devices[i]));
        cast_dtypes[i] =  BaseType(in_edge->src()->output_type(in_edge->src_output()));
        const std::string& feed_key =
          Rendezvous::CreateKey(send_devices[i],
                                static_cast<uint64>(send_device_incarnations[i]),
                                in_edge->dst()->assigned_device_name(),
                                tensor_names[i], FrameAndIter(0, 0));
        // Record the feed_key's slot in fuse_recv node.
        // We should promise the correct order.
        src_to_slot[feed_key] = i;
      }
      fuse_recv_builder.Attr("tensor_names", tensor_names);
      fuse_recv_builder.Attr("send_devices", send_devices);
      fuse_recv_builder.Attr("send_device_incarnations", send_device_incarnations);
      auto in_edge = fuse_recv_edges[0];
      fuse_recv_builder.Attr("recv_device", in_edge->dst()->assigned_device_name());
      fuse_recv_builder.Attr("client_terminated", true);

      fuse_recv_builder.Device(in_edge->dst()->assigned_device_name())
        .Attr("tensor_types", cast_dtypes);
      GraphDef &graph_def = sub_graph.GetGraphDef();
      *new_fuse_recv_node = graph_def.add_node();
      Status status = fuse_recv_builder.Finalize(*new_fuse_recv_node);
      if (!status.ok()) {
        LOG(FATAL) << "Finalize fuse recv node failed, " << status.error_message();
      }
    }
  }

  return *new_fuse_recv_node != nullptr;
}

Status GraphPartitionerBase::CompleteSubGraph(SubGraph &sub_graph, int graph_idx) {
  const std::string &loc = sub_graph.GetLoc();
  GraphDef &graph_def = sub_graph.GetGraphDef();

  NodeDef* new_fuse_recv_node = nullptr;
  std::unordered_map<std::string, int> src_to_slot;
  bool use_fuse_recv = UseFuseRecv(sub_graph, graph_idx,
                                   &new_fuse_recv_node,
                                   src_to_slot);

  bool use_send_recv = false;
  std::map<InputSrcKey, NodeDef*> input_src_nodes_map;
  for (const Node* node: sub_graph.GetNodes()) {
    NodeDef *node_def = graph_def.add_node();
    Status s = ConstructNodeDef(node, node_def);
    RETURN_IF_NOT_OK(s);

    if (node->IsVariable() && !sub_graph.IsOnlyVariable()) {
      continue;
    }

    if (node->IsIdentity() &&
        GetVarOfIdentity(node) != NULL &&
        !sub_graph.IsOnlyVariable()) {
      continue;
    }

    std::vector<std::pair<std::string, const Edge*>> boundary_input_edges;
    if (use_fuse_recv) {
      s = ProcessSubNodeInputs(node, loc, node_def,
                               &graph_def, &input_src_nodes_map,
                               &boundary_input_edges,
                               use_send_recv,
                               new_fuse_recv_node,
                               src_to_slot);
    } else {
      s = ProcessSubNodeInputs(node, loc, node_def,
                               &graph_def, &input_src_nodes_map,
                               &boundary_input_edges,
                               use_send_recv);
    }
    RETURN_IF_NOT_OK(s);
    sub_graph.AddInputs(boundary_input_edges);

    std::vector<std::pair<std::string, const Edge*>> boundary_output_edges;
    s = ProcessSubNodeOutputs(node, loc, node_def, &graph_def,
                              &boundary_output_edges,
                              use_send_recv);
    RETURN_IF_NOT_OK(s);
    sub_graph.SetSendRecvFlag(use_send_recv);
    sub_graph.AddOutputs(boundary_output_edges);
  }

  return Status::OK();
}

Status GraphPartitionerBase::CompleteSubGraphs(std::vector<SubGraph> *sub_graphs) {
  int idx = 0;
  for (SubGraph &sub_graph : *sub_graphs) {
    Status status = CompleteSubGraph(sub_graph, idx);
    if (!status.ok()) {
      return status;
    }
    ++idx;
  }

  return Status::OK();
}

NodeDef* AddBridgeNode(const PartitionOptions &opts, const Edge *edge,
                       const std::string &device_name, GraphDef *worker_graph_def)
{
  std::string name = strings::StrCat("_bridge_", edge->src()->name(), "_", edge->src_output());
  NodeDef *node_def = worker_graph_def->add_node();
  if (edge->IsControlEdge()) {
    NodeDefBuilder(name, "NoOp")
        .Device(device_name)
        .Finalize(node_def);
  } else {
    node_def->set_name(name);
    node_def->set_device(device_name);
    node_def->set_op("Identity");
    (*node_def->mutable_attr())["T"].set_type(BaseType(edge->src()->output_type(edge->src_output())));
  }

  return node_def;
}

NodeDef* FindOrCreateBridgeNode(const PartitionOptions &opts, const Edge *edge,
                                const std::string &device_name, GraphDef *worker_graph_def,
                                std::map<InputSrcKey, NodeDef*> *bridge_nodes_map)
{
  NodeDef* bridge_node_def = nullptr;
  InputSrcKey bridge_node_key = MakeInputSrcKey(edge);
  auto it = bridge_nodes_map->find(bridge_node_key);
  if (it == bridge_nodes_map->end()) {
    bridge_node_def = AddBridgeNode(opts, edge, device_name, worker_graph_def);
    bridge_nodes_map->insert(make_pair(bridge_node_key, bridge_node_def));
  } else {
    bridge_node_def = it->second;
  }
  return bridge_node_def;
}

Status GraphPartitionerBase::ProcessRunGraphInputs(
    const SubGraph &ps_graph,
    const std::string &worker_device,
    GraphDef *worker_graph_def,
    NodeDef *run_graph_node_def,
    std::map<InputSrcKey, NodeDef*> *bridge_nodes_map)
{
  std::set<InputSrcKey> src_set;
  for (const pair<std::string, const Edge*> &input : ps_graph.GetInputEdges()) {
    const std::string &feed_key = input.first;
    const Edge *in_edge = input.second;

    std::string input_src_name;
    int input_src_idx = in_edge->src_output();
    if (!in_edge->src()->IsOp()) continue;
    if (!is_main_loc_func_(opts_.node_to_loc(in_edge->src()))) {
      NodeDef *bridge_node_def = FindOrCreateBridgeNode(opts_, in_edge,
          worker_device, worker_graph_def, bridge_nodes_map);
      input_src_name = bridge_node_def->name();
      input_src_idx = in_edge->IsControlEdge() ? Graph::kControlSlot : 0;
    } else {
      input_src_name = in_edge->src()->name();
    }

    if(src_set.find({input_src_name, input_src_idx}) != src_set.end()) {
      continue;
    }
    src_set.insert({input_src_name, input_src_idx});
    AddInput(run_graph_node_def, input_src_name, input_src_idx);

    if (!in_edge->IsControlEdge()) {
      *((*run_graph_node_def->mutable_attr())["feed_names"].mutable_list()->add_s()) = feed_key;
      (*run_graph_node_def->mutable_attr())["T1"].mutable_list()->add_type(
          BaseType(in_edge->src()->output_type(in_edge->src_output())));
    }
  }

  std::vector<std::string> input_names;
  for (int i = 0; i < run_graph_node_def->input_size(); i++) {
    input_names.push_back(run_graph_node_def->input(i));
  }

  ResetInputs(input_names, run_graph_node_def);

  return Status::OK();
}

Status GraphPartitionerBase::ProcessRunGraphOutputs(
    const SubGraph &ps_graph,
    const std::string &worker_device,
    GraphDef *worker_graph_def,
    NodeDef *run_graph_node_def,
    std::map<InputSrcKey, NodeDef*> *bridge_nodes_map,
    std::map<int, NodeDef*> *added_nodes)
{
  int idx = 0;
  std::map<InputSrcKey, int> output_idx_map;
  for (const pair<std::string, const Edge*> &output : ps_graph.GetOutputEdges()) {
    const std::string &fetch_key = output.first;
    const Edge *out_edge = output.second;
    const Node *dst = out_edge->dst();

    std::string output_name;
    if (out_edge->IsControlEdge()) {
      output_name = "^" + run_graph_node_def->name();
    } else {
      int output_idx = 0;
      auto output_key = MakeInputSrcKey(out_edge);
      if (output_idx_map.find(output_key) == output_idx_map.end()) {
        output_idx = idx++;
        output_idx_map[output_key] = output_idx;
        *((*run_graph_node_def->mutable_attr())["fetch_names"].mutable_list()->add_s()) = fetch_key;
        (*run_graph_node_def->mutable_attr())["T2"].mutable_list()->add_type(
            BaseType(out_edge->dst()->input_type(out_edge->dst_input())));
      } else {
        output_idx = output_idx_map[output_key];
      }
      output_name = strings::StrCat(run_graph_node_def->name(), ":", output_idx);
    }

    if (!dst->IsOp()) continue;
    if (!is_main_loc_func_(opts_.node_to_loc(dst))) {
      NodeDef *bridge_node_def = FindOrCreateBridgeNode(opts_, out_edge,
          worker_device, worker_graph_def, bridge_nodes_map);

      if (bridge_node_def->input_size() == 0) {
        *bridge_node_def->add_input() = output_name;
      } else {
        if (bridge_node_def->input(0) != output_name) {
          return errors::Internal("Many input for birdge node, it's expected to be ONE.",
                                  " node:", bridge_node_def->name());
        }
      }
    } else {
      NodeDef *dst_node_def = nullptr;
      if (added_nodes->find(dst->id()) == added_nodes->end()) {
        dst_node_def = worker_graph_def->add_node();
        Status s = ConstructNodeDef(dst, dst_node_def);
        RETURN_IF_NOT_OK(s);
        added_nodes->insert(make_pair(dst->id(), dst_node_def));
      } else {
        dst_node_def = (*added_nodes)[dst->id()];
      }
      int dst_input_idx = FindInputIdxInDef(dst_node_def, out_edge);
      if (dst_input_idx < 0) {
        return errors::Internal("Invalid dst input idx:", dst_input_idx,
                                " for node:", dst_node_def->name(),
                                ", out_edge:", out_edge->src()->name(),
                                ", dst node def:", dst_node_def->DebugString());
      }
      *dst_node_def->mutable_input(dst_input_idx) = output_name;
    }
  }
  return Status::OK();
}

Status GraphPartitionerBase::CompleteMainGraph(
    const std::vector<SubGraph> &sub_graphs,
    SubGraph *worker_graph)

{
  std::unordered_map<std::string, int> ps_graph_count;
  for (auto sub_graph : sub_graphs) {
    if (ps_graph_count.find(sub_graph.GetLoc()) == ps_graph_count.end()) {
      ps_graph_count[sub_graph.GetLoc()] = 1;
    } else {
      ++ps_graph_count[sub_graph.GetLoc()];
    }
  }

  GraphDef &graph_def = worker_graph->GetGraphDef();
  std::string worker_device = worker_graph->GetDeviceName();
  if (worker_device.empty()) {
    // if worker's node count = 0
    if ((worker_graph->GetNodes()).size() == 0) {
      LOG(WARNING) << "Worker's graph is empty, add a RunStarGraphOp node.";
      int64 task_index = -1;
      Status s = ReadInt64FromEnvVar("TASK_INDEX", -1, &task_index);
      if (!s.ok() || task_index == -1) {
        LOG(FATAL) << "Read Env 'TASK_INDEX' failed. task_index=" << task_index;
      }

      // NOTE(rangeng.llb): Add a specific run (star) graph node for each
      // ps subgraph.
      for (auto sub_graph : sub_graphs) {
        NodeDef *run_graph_node_def = graph_def.add_node();
        MakeRunGraphNodeDef(sub_graph,
                            strings::StrCat("/job:worker/replica:0/task:",
                            task_index, "/device:CPU:0"),
                            run_graph_node_def,
                            zero_copy_,
                            ps_graph_count[sub_graph.GetLoc()]);
      }

      return Status::OK();
    }

    return errors::Internal("empty worker device name!");
  }

  std::map<int, NodeDef*> added_nodes;
  std::map<InputSrcKey, NodeDef*> bridge_nodes_map;
  for (const SubGraph &sub_graph : sub_graphs) {
    if (sub_graph.GetInputEdges().empty() &&
        sub_graph.GetOutputEdges().empty() &&
        !sub_graph.GetSendRecvFlag()) {
      // NOTE(jiankeng.pt): We should add a run graph node for every sub graph.
      // One sub graph should has some out/in edges, or has
      // direct edge which connect to other ps.
      // So if code run to here, it may be some wrong there.
      // LOG(FATAL) << "No RunGraph node to trigger the ps graph to run. \
      //              There must be some wrong with your graph partition.";
        std::cerr << "return le a \n";
      continue;
    } else {
        std::cerr << sub_graph.GetInputEdges().size() << std::endl;
        std::cerr << sub_graph.GetOutputEdges().size() << std::endl;
    }

    NodeDef *run_graph_node_def = graph_def.add_node();
    std::cerr << "caixukun1\n";
    MakeRunGraphNodeDef(sub_graph, worker_device,
                        run_graph_node_def, zero_copy_,
                        ps_graph_count[sub_graph.GetLoc()]);
    std::cerr << "caixukun2\n";

    Status s = ProcessRunGraphInputs(sub_graph, worker_device,
                                     &graph_def, run_graph_node_def, &bridge_nodes_map);
    std::cerr << "caixukun3\n";
    RETURN_IF_NOT_OK(s);

    s = ProcessRunGraphOutputs(sub_graph, worker_device,
                               &graph_def, run_graph_node_def, &bridge_nodes_map,
                               &added_nodes);
    RETURN_IF_NOT_OK(s);
    std::cerr << "caixukun4\n";
  }

  for (const Node* node : worker_graph->GetNodes()) {
    if (added_nodes.find(node->id()) == added_nodes.end()) {
      NodeDef *node_def = graph_def.add_node();
      Status s = ConstructNodeDef(node, node_def);
      if (!s.ok()) {
        return s;
      }
    }
  }

    std::cerr << "caixukun5\n";
  return Status::OK();
}

bool IsReadyPsGraph(const PartitionOptions &opts,
                    const SubGraph &ps_graph,
                    const set<const Node*> &ready_nodes)
{
  const std::string &loc = ps_graph.GetLoc();
  std::vector<const Edge*> in_edges;
  for (const Node* node : ps_graph.GetNodes()) {
    for (const Edge* in_edge : node->in_edges()) {
      const Node* src = in_edge->src();
      if (!src->IsOp() || opts.node_to_loc(src) == loc) {
        continue;
      }
      in_edges.push_back(in_edge);
    }
  }

  for (const Edge* in_edge : in_edges) {
    const Node *src = in_edge->src();
    if (ready_nodes.find(src) == ready_nodes.end()) {
      return false;
    }
  }

  return true;
}

void GraphPartitionerBase::MergeReadyPsSubGraphs(
    set<const SubGraph*> *ps_sub_graph_set,
    set<const Node*> *ready_nodes,
    std::vector<SubGraph> *merged_ps_sub_graphs)
{
  std::vector<const SubGraph*> to_removed;
  std::unordered_map<std::string, SubGraph*> merged_ps_graphs;
  std::unordered_map<std::string, SubGraph*> merged_ps_variable_graphs;
  for (const SubGraph* ps_graph : *ps_sub_graph_set) {
    if (!IsReadyPsGraph(opts_, *ps_graph, *ready_nodes)) {
      continue;
    }

    if (ps_graph->IsOnlyVariable()) {
      // merged_ps_sub_graphs->push_back(*ps_graph);

      // NOTE(jiankeng.pt): Before fuse only variable graphs
      // which are in the same ps, the only variable graph
      // has just one or two nodes. But now, we fuse these graphs,
      // a only variable graph will contain many nodes.
      SubGraph* merged_ps_graph = nullptr;
      string loc = ps_graph->GetLoc();
      auto it = merged_ps_variable_graphs.find(loc);
      if (it == merged_ps_variable_graphs.end()) {
        merged_ps_graph = new SubGraph(loc);
        merged_ps_graph->SetOnlyVariable();
        merged_ps_variable_graphs[loc] = merged_ps_graph;
      } else {
        merged_ps_graph = it->second;
      }
      merged_ps_graph->Extend(*ps_graph);
    } else {
      SubGraph* merged_ps_graph = nullptr;
      const std::string &loc = ps_graph->GetLoc();
      auto it = merged_ps_graphs.find(loc);
      if (it == merged_ps_graphs.end()) {
        merged_ps_graph = new SubGraph(loc);
        merged_ps_graphs[loc] = merged_ps_graph;
      } else {
        merged_ps_graph = it->second;
      }

      merged_ps_graph->Extend(*ps_graph);
    }

    for (const Node* node : ps_graph->GetNodes()) {
      ready_nodes->insert(node);
    }

    to_removed.push_back(ps_graph);
  }

  for (size_t i = 0; i < to_removed.size(); i++) {
    ps_sub_graph_set->erase(to_removed[i]);
  }

  for (auto it = merged_ps_variable_graphs.begin(); it != merged_ps_variable_graphs.end(); it++) {
    merged_ps_sub_graphs->push_back(*it->second);
    delete it->second;
  }

  for (auto it = merged_ps_graphs.begin(); it != merged_ps_graphs.end(); it++) {
    merged_ps_sub_graphs->push_back(*it->second);
    delete it->second;
  }
}

void GraphPartitionerBase::RemoveReadyWorkerNodes(
        set<const Node*> *ready_nodes,
        set<const Node*> *not_ready_worker_nodes)
{
  while (true) {
    set<const Node*> tmp_ready_nodes;
    for (const Node* node : *not_ready_worker_nodes) {
      size_t ready_input_count = 0;
      for (const Edge* in_edge : node->in_edges()) {
        const Node *src = in_edge->src();
        if (!src->IsOp()) {
          ready_input_count++;
          continue;
        }

        if (ready_nodes->find(src) != ready_nodes->end()) {
          ready_input_count++;
        }
      }
      if (ready_input_count == node->in_edges().size()) {
        ready_nodes->insert(node);
        tmp_ready_nodes.insert(node);
      }
    }

    for (const Node *node : tmp_ready_nodes) {
      not_ready_worker_nodes->erase(node);
    }

    if (tmp_ready_nodes.empty()) {
      break;
    }
  }
}

std::string GetPsNodePrefix(const SubGraph &ps_sub_graph, size_t i) {
  const std::string &loc = ps_sub_graph.GetLoc();
  string prefix = loc;
  std::replace(prefix.begin(), prefix.end(), '/', '_');
  std::replace(prefix.begin(), prefix.end(), ':', '_');
  stringstream ss;
  ss << prefix << "-" << i;
  return ss.str();
}

void AddNode(const Node* node, const std::map<std::string, std::string> &prefix_map,
             GraphDef *gdef)
{
  NodeDef *node_def = gdef->add_node();
  *node_def = node->def();
  std::string node_name = node_def->name();
  auto it = prefix_map.find(node_name);
  if (it != prefix_map.end()) {
    node_name = it->second + "/" + node_name;
    *(node_def->mutable_name()) = node_name;
  }

  for (int i = 0; i < node_def->input_size(); i++) {
    std::string input_name = node_def->input(i);
    std::string input_node_name;
    std::string input_node_idx;
    size_t pos = input_name.find(":");
    if (pos != string::npos) {
      input_node_name = input_name.substr(0, pos);
      input_node_idx = input_name.substr(pos);
    } else {
      input_node_name = input_name;
    }

    std::string control_prefix;
    if (input_node_name[0] == '^') {
      input_node_name = input_node_name.substr(1);
      control_prefix = "^";
    }

    std::string prefix;
    auto it = prefix_map.find(input_node_name);
    if (it != prefix_map.end()) {
      prefix = it->second;
    }

    std::string new_input_name = control_prefix + prefix +
                                 "/" + input_node_name + input_node_idx;
    *node_def->mutable_input(i) = new_input_name;
  }
}

void PrintGraphForDebug(const SubGraph &worker_sub_graph,
                        const std::vector<SubGraph> &ps_sub_graphs)
{
  std::map<std::string, std::string> node_prefix_map;
  for (size_t i = 0; i < ps_sub_graphs.size(); i++) {
    std::string prefix = GetPsNodePrefix(ps_sub_graphs[i], i);
    for (const Node* node : ps_sub_graphs[i].GetNodes()) {
      node_prefix_map[node->name()] = prefix;
    }
  }

  GraphDef gdef;
  for (size_t i = 0; i < ps_sub_graphs.size(); i++) {
    for (const Node* node: ps_sub_graphs[i].GetNodes()) {
      AddNode(node, node_prefix_map, &gdef);
    }
  }

  for (const Node* node: worker_sub_graph.GetNodes()) {
    AddNode(node, node_prefix_map, &gdef);
  }

  static int i = 0;
  stringstream ss;
  ss << "splited_graph_" << i++;
  ofstream fout(ss.str());
  std::string s;
  gdef.SerializeToString(&s);
  fout.write(s.c_str(), s.length());
  fout.flush();
  fout.close();
}

Status GraphPartitionerBase::MergePsGraphs(
    const SubGraph &worker_sub_graph,
    const std::vector<SubGraph> &ps_sub_graphs,
    std::vector<SubGraph> *merged_ps_graphs)
{
  set<const Node*> not_ready_worker_nodes;
  for (const Node* node : worker_sub_graph.GetNodes()) {
    not_ready_worker_nodes.insert(node);
  }
  set<const SubGraph*> ps_sub_graph_set;
  for (const SubGraph &sub_graph : ps_sub_graphs) {
    ps_sub_graph_set.insert(&sub_graph);
  }

  set<const Node*> ready_nodes;

  // Nodes which have no input edges or inptut edges are inner edge, 
  // should become ready nodes now.
  RemoveReadyWorkerNodes(&ready_nodes, &not_ready_worker_nodes);

  while (!ps_sub_graph_set.empty()) {
    size_t last_ps_count = ps_sub_graph_set.size();
    size_t last_worker_count = not_ready_worker_nodes.size();
    MergeReadyPsSubGraphs(&ps_sub_graph_set,
                          &ready_nodes,
                          merged_ps_graphs);
    RemoveReadyWorkerNodes(&ready_nodes, &not_ready_worker_nodes);

    if ((last_ps_count > 0 && last_ps_count == ps_sub_graph_set.size()) &&
        (last_worker_count == not_ready_worker_nodes.size()))
    {
      LOG(ERROR) << "can not merge the ps graphs. left "
                 << ps_sub_graph_set.size() << " ps sub-graph were not be merged.";
      return errors::Internal("can not merge the ps graphs.");
    }
  }
  return Status::OK();
}

std::string GraphPartitionerBase::GetWorkerDevice() {
  for (Node *node : graph_->nodes()) {
    if (!node->IsOp()) continue;
    if (is_main_loc_func_(opts_.node_to_loc(node))) {
      return node->assigned_device_name();
    }
  }
  return "";
}

// RefSwitch and RefMerge can't place to worker by force,
// the ref type data is hard to handle.
Status GraphPartitionerBase::ResetSwitchOpDevice() {
  bool has_switch = false;
  for (Node *node : graph_->nodes()) {
    if (!node->IsOp()) continue;
    if (node->def().op() == "Switch" || node->def().op() == "Merge") {
      has_switch = true;
      break;
    }
  }

  std::string worker_device = GetWorkerDevice();
  if (has_switch && worker_device.empty()) {
    return errors::Internal("Failed to get worker device.");
  }

  set<Node*> nodes;
  for (Node *node : graph_->nodes()) {
    if (!node->IsOp()) continue;

    std::string loc = opts_.node_to_loc(node);
    if (node->def().op() == "Switch") {
      for (Node *out_node : node->out_nodes()) {
        std::string out_node_loc = opts_.node_to_loc(out_node);
        if (loc == out_node_loc) {
          continue;
        }
        nodes.insert(node);
      }
    } else if (node->def().op() == "Merge") {
      for (Node *in_node : node->in_nodes()) {
        std::string in_node_loc = opts_.node_to_loc(in_node);
        if (loc == in_node_loc) {
          continue;
        }
        nodes.insert(node);
      }
    }
  }

  for (Node *node : nodes) {
    node->set_assigned_device_name(worker_device);
  }

  return Status::OK();
}

Status GraphPartitionerBase::SplitGraphInternal(
    std::vector<SubGraph> *sub_graphs,
    SubGraph *main_graph,
    bool needResetSwitchOp)
{
  Status status;

  GraphInfo g_info;
  if (!opts_.control_flow_added) {
    // Add the "code" for distributed execution of control flow. Code is
    // added only for the frames that are placed on multiple devices. The
    // new graph is an equivalent transformation of the original graph and
    // has the property that it can be subsequently partitioned arbitrarily
    // (down to the level of individual device) for distributed execution.
    status = AddControlFlow(opts_, graph_, &g_info);
    if (!status.ok()) return status;
  }

  // At this point, all the graph mutations have been done. Build memory
  // and device type info for every node and edge in the graph.
  RETURN_IF_NOT_OK(BuildMemoryDeviceInfo(*graph_, &g_info));
  if (needResetSwitchOp) {
    RETURN_IF_NOT_OK(ResetSwitchOpDevice());
  }

  std::vector<bool> node_visited(graph_->num_node_ids(), false); 
  for (const Node* cur : graph_->nodes()) {
    if (!cur->IsOp()) {
      continue;  // Skip Sink/Source nodes.
    }

    if (node_visited[cur->id()]) {
      continue;
    }

    std::string cur_loc = opts_.node_to_loc(cur);
    SubGraph cur_sub_graph(cur_loc);
    cur_sub_graph.SetLoc(cur_loc);
    node_visited[cur->id()] = true;

    if (cur->IsVariable() && !is_main_loc_func_(cur_loc)) {
      for (const auto& out_edge : cur->out_edges()) {
        if (!out_edge->dst()->IsOp()) continue;
        if (opts_.node_to_loc(out_edge->dst()) != cur_loc) {
          cur_sub_graph.AddNode(cur);
          cur_sub_graph.SetOnlyVariable();
          sub_graphs->push_back(cur_sub_graph);
          break;
        }
      }
      continue;
    }

    if (cur->IsIdentity() && !is_main_loc_func_(cur_loc)) {
      const Node* var = GetVarOfIdentity(cur);
      if (var != NULL) {
        bool is_boundary = false;
        for (const auto& out_edge : cur->out_edges()) {
          if (!out_edge->dst()->IsOp()) continue;
          if (opts_.node_to_loc(out_edge->dst()) != cur_loc) {
            is_boundary = true;
            break;
          }
        }
        if (!is_boundary) {
          continue;
        }

        cur_sub_graph.AddNode(var);
        cur_sub_graph.AddNode(cur);
        cur_sub_graph.SetOnlyVariable();
        sub_graphs->push_back(cur_sub_graph);
        continue;
      }
    }

    std::deque<const Node*> nodes;
    nodes.push_back(cur);
    cur_sub_graph.AddNode(cur);
    while (!nodes.empty()) {
      const Node* node = nodes.front();
      nodes.pop_front();
      for (const Edge* edge : node->in_edges()) {
        Node *src = edge->src();
        if (!src->IsOp()) continue;

        TryAddEdgeNode(cur_loc, src, nodes, node_visited, cur_sub_graph);
      }

      for (const Edge* edge : node->out_edges()) {
        Node *dst = edge->dst();
        if (!dst->IsOp()) continue;

        TryAddEdgeNode(cur_loc, dst, nodes, node_visited, cur_sub_graph);
      }
    }

    if (is_main_loc_func_(cur_loc)) {
      main_graph->Extend(cur_sub_graph);
      main_graph->SetLoc(cur_loc);
    } else {
      sub_graphs->push_back(cur_sub_graph);
    }
  }
  return Status::OK();
}

void TrainGraphPartitioner::MakeRunGraphNodeDef(const SubGraph &ps_graph,
        const std::string &worker_device,
        NodeDef *run_graph_node_def,
        bool zero_copy,
        int ps_graph_count)
{
  static int idx = 0;
  std::string ps_loc = ps_graph.GetLoc();
  std::replace(ps_loc.begin(), ps_loc.end(), '/', '_');
  std::replace(ps_loc.begin(), ps_loc.end(), ':', '_');

  run_graph_node_def->set_name(strings::StrCat("run_graph", ps_loc, "_", idx++));
  if (zero_copy) {
    run_graph_node_def->set_op("StarRunGraph");
  } else {
    run_graph_node_def->set_op("RunGraph");
  }
  run_graph_node_def->set_device(worker_device);
  *((*run_graph_node_def->mutable_attr())["graph_handle"].mutable_s()) = ps_graph.GetGraphHandle();
  *((*run_graph_node_def->mutable_attr())["loc"].mutable_s()) = ps_graph.GetLoc();
  *((*run_graph_node_def->mutable_attr())["T2"].mutable_list()) = {};
  *((*run_graph_node_def->mutable_attr())["T1"].mutable_list()) = {};
  *((*run_graph_node_def->mutable_attr())["feed_names"].mutable_list()) = {};
  *((*run_graph_node_def->mutable_attr())["fetch_names"].mutable_list()) = {};
  (*run_graph_node_def->mutable_attr())["ps_graph_count"].set_i(ps_graph_count);
}

Status TrainGraphPartitioner::SplitGraph(
    SubGraph *worker_sub_graph,
    std::vector<SubGraph> *ps_sub_graphs,
    bool merge_ps_graph)
{

  RETURN_IF_NOT_OK(SplitGraphInternal(ps_sub_graphs, worker_sub_graph, true));
    
  for (auto &sub_graph : *ps_sub_graphs) {
    sub_graph.CompleteVariables(opts_);
  }

  if (merge_ps_graph) {
    std::vector<SubGraph> merged_ps_sub_graphs;
    auto status = MergePsGraphs(*worker_sub_graph, *ps_sub_graphs, &merged_ps_sub_graphs);
    if (status.ok()) {
      ps_sub_graphs->swap(merged_ps_sub_graphs);
    } else {
      LOG(ERROR) << "Merge ps graph error, " << status.error_message();
    }
  }

  return Status::OK();
}

Status InferGraphPartitioner::SplitGraph(
    SubGraph *main_sub_graph,
    std::vector<SubGraph> *sub_graphs,
    bool merge_ps_graph)
{
  RETURN_IF_NOT_OK(SplitGraphInternal(sub_graphs, main_sub_graph, false));

  if (merge_ps_graph) {
    std::vector<SubGraph> merged_sub_graphs;
    auto status = MergePsGraphs(*main_sub_graph, *sub_graphs, &merged_sub_graphs);
    if (status.ok()) {
      sub_graphs->swap(merged_sub_graphs);
    }
  }

  return Status::OK();
}

void InferGraphPartitioner::MakeRunGraphNodeDef(const SubGraph &ps_graph,
        const std::string &worker_device,
        NodeDef *run_graph_node_def,
        bool zero_copy,
        int ps_graph_count)
{
  static int idx = 0;
  std::string ps_loc = ps_graph.GetLoc();
  std::replace(ps_loc.begin(), ps_loc.end(), '/', '_');
  std::replace(ps_loc.begin(), ps_loc.end(), ':', '_');

  run_graph_node_def->set_name(strings::StrCat("run_graph", ps_loc, "_", idx++));
  run_graph_node_def->set_op("RunGraphOp");
  run_graph_node_def->set_device(worker_device);
  *((*run_graph_node_def->mutable_attr())["graph_handle"].mutable_s()) = ps_graph.GetGraphHandle();
  *((*run_graph_node_def->mutable_attr())["loc"].mutable_s()) = ps_graph.GetLoc();
  *((*run_graph_node_def->mutable_attr())["T2"].mutable_list()) = {};
  *((*run_graph_node_def->mutable_attr())["T1"].mutable_list()) = {};
  *((*run_graph_node_def->mutable_attr())["feed_names"].mutable_list()) = {};
  *((*run_graph_node_def->mutable_attr())["fetch_names"].mutable_list()) = {};
}


NodeDef* TrainGraphPartitioner::DealWithSubNodeOutputEdge(
    const Edge* out_edge,
    std::unordered_map<std::string, NodeDef*>& send_nodes_map,
    std::unordered_map<std::string, int>& send_nodes_index,
    GraphDef *graph_def)
{
  NodeDef *send_node_def = nullptr;
  const std::string& output_src_key = strings::StrCat(out_edge->src()->name(),
                                                        ":", out_edge->src_output());
  auto data_type = DT_FLOAT;
  std::string node_name, tensor_name, input_name;
  int input_slot;

  if (send_nodes_index.find(output_src_key) == send_nodes_index.end()) {
    send_nodes_index[output_src_key] = 0;
  } else {
    ++send_nodes_index[output_src_key];
  }

  // For the control edge, we should add an extra 'Const' dummy node.
  // so the condition contain the state: ShouldUseSendRecvMode = true
  if (out_edge->IsControlEdge()) {
    node_name = strings::StrCat("_send_", out_edge->src()->name(),
                                "_", send_nodes_index[output_src_key], "__1");
    tensor_name = strings::StrCat(out_edge->src()->name());
    Tensor tensor(DT_FLOAT, TensorShape({0}));
    NodeDef* dummy = graph_def->add_node();
    Status s = NodeDefBuilder(strings::StrCat("_dummy_", out_edge->src()->name(),
                                              "_", send_nodes_index[output_src_key]), "Const")
                   .Device(out_edge->src()->assigned_device_name())
                   .Attr("dtype", DT_FLOAT)
                   .Attr("value", tensor)
                   .Finalize(dummy);
    if (!s.ok()) {
      LOG(ERROR) << "construct dummy node for send failed:" << s.error_message();
      return nullptr;
    }
    AddInput(dummy, out_edge->src()->name(), Graph::kControlSlot);
    input_name = dummy->name();
    input_slot = 0;

  } else {
    data_type = BaseType(out_edge->src()->output_type(out_edge->src_output()));
    node_name = strings::StrCat("_send_", out_edge->src()->name(),
                                "_", send_nodes_index[output_src_key],
                                "_", out_edge->src_output());
    tensor_name = strings::StrCat(out_edge->src()->name(),
                                  ":", out_edge->src_output());
    input_name = out_edge->src()->name();
    input_slot = out_edge->src_output();
  }

  std::string send_device = out_edge->src()->assigned_device_name();
  std::string recv_device = out_edge->src()->assigned_device_name();
  const std::string& dst_device = out_edge->dst()->assigned_device_name();
  bool shouldUseSendRecvMode = ShouldUseSendRecvMode(out_edge->src(),
                                                     out_edge->dst());

  // NOTE(jiankeng.pt): Only one 'send' node is needed when output
  // to different or same device, except 
  // shouldUseSendRecvMode=true.
  // We always should return the boundary_edge
  // to the run graph op node, the boundary_edge
  // can help to add connection between worker node
  // and run graph op node.
  if (send_nodes_map.find(output_src_key) == send_nodes_map.end() ||
      shouldUseSendRecvMode) {
    bool client_terminated = true;
    if (shouldUseSendRecvMode) {
      recv_device = dst_device;
      client_terminated = false;
    }
        
    send_node_def = graph_def->add_node();
    Status s = ConstructSendNodeDef(node_name,
                                    send_device,
                                    recv_device,
                                    input_name,                
                                    input_slot,
                                    tensor_name,
                                    data_type,
                                    client_terminated,
                                    send_node_def);
    if (!s.ok()) {
      LOG(ERROR) << "construct send node failed:" << s.error_message();
      return nullptr;
    }

    // We don't need to record send node when shouldUseSendRecvMode=true.
    // In case of return the wrong send node that will create the wrong rendezvous key.
    if (!shouldUseSendRecvMode) {
      send_nodes_map[output_src_key] = send_node_def;
    }

  } else {
    send_node_def = send_nodes_map[output_src_key];
  }

  return send_node_def;
}

NodeDef* TrainGraphPartitioner::DealWithSubNodeOutputEdge(
    const Edge* out_edge,
    std::map<InputSrcKey, NodeDef*> &send_nodes_map,
    GraphDef *graph_def)
{
  NodeDef *send_node_def = nullptr;
  InputSrcKey output_src_key = MakeInputSrcKey(out_edge);
  auto it = send_nodes_map.find(output_src_key);
  if (it == send_nodes_map.end()) {
    send_node_def = graph_def->add_node();
    Status s = ConstructSendNodeDef(
        strings::StrCat("_send_", out_edge->src()->name(), "_", out_edge->src_output()),
        out_edge->src()->assigned_device_name(),
        out_edge->src()->assigned_device_name(),
        out_edge->src()->name(),
        out_edge->src_output(),      
        strings::StrCat(out_edge->src()->name(), ":", out_edge->src_output()),
        out_edge->src()->output_type(out_edge->src_output()),
        true,
        send_node_def);
    if (!s.ok()) {
      LOG(ERROR) << "construct send node failed:" << s.error_message();
      return nullptr;
    }
    send_nodes_map. insert(make_pair(output_src_key, send_node_def));
  }

  return send_node_def;
}

NodeDef* GraphPartitionerBase::DealWithSubNodeInputEdge(
    const Edge* in_edge,
    std::map<InputSrcKey, NodeDef*> *input_src_nodes_map,
    string *feed_key,
    GraphDef *graph_def)
{
  InputSrcKey input_src_key = MakeInputSrcKey(in_edge);
  NodeDef *src_node_def = nullptr;
  auto it = input_src_nodes_map->find(input_src_key);
  if (it != input_src_nodes_map->end()) {
    src_node_def = it->second;
  } else {
    src_node_def = graph_def->add_node();
    Status s = MakeInputSrcNode(in_edge, src_node_def, feed_key);
    if (!s.ok()) {
      LOG(ERROR) << "create input src node failed:" << s.error_message();
      return nullptr;
    }
    input_src_nodes_map->insert(make_pair(input_src_key, src_node_def));
  }
  return src_node_def;
}

NodeDef* InferGraphPartitioner::DealWithSubNodeOutputEdge(
    const Edge* out_edge,
    std::map<InputSrcKey, NodeDef*> &send_nodes_map,
    GraphDef *graph_def)
{
  // TODO(jiankeng.pt) Infer not implement now.
  return nullptr;
}

NodeDef* InferGraphPartitioner::DealWithSubNodeOutputEdge(
    const Edge* out_edge,
    std::unordered_map<std::string, NodeDef*>& send_nodes_map,
    std::unordered_map<std::string, int>& send_nodes_index,
    GraphDef *graph_def)
{
  // TODO(jiankeng.pt) Infer not implement now.
  return nullptr;
}

SendRecvPolicy::SendRecvPolicy(Graph* graph) : graph_(graph) {
}

SendRecvPolicy::~SendRecvPolicy() {
}

void SendRecvPolicy::UseDefaultPolicy() {
  std::unordered_map<std::string, std::unordered_set<std::string> > ps_depend_in_device;
  for (Node* n : graph_->nodes()) {
    if (!n->IsOp()) continue;
    std::string cur_loc(n->assigned_device_name());
    if (cur_loc.find("ps") == std::string::npos) continue;
    for (const Edge* in_edge : n->in_edges()) {
      if (!in_edge->src()->IsOp() || cur_loc == in_edge->src()->assigned_device_name() ||
          in_edge->src()->assigned_device_name().find("ps") == std::string::npos) continue;
      ps_depend_in_device[n->assigned_device_name()].insert(in_edge->src()->assigned_device_name());
    }
  }

  for (auto info : ps_depend_in_device) {
    for (auto loc : ps_depend_in_device[info.first]) {
      if (ps_depend_in_device[loc].find(info.first) != ps_depend_in_device[loc].end()) {
        _use_send_recv_loc[info.first].insert(loc);
        _use_send_recv_loc[loc].insert(info.first);
      }
    }
  }

  // TODO(jiankeng.pt) Check device again, delete the below code
  for (auto locs : _use_send_recv_loc) {
    auto curr_loc = locs.first;
    if (curr_loc.find("ps") == std::string::npos) {
      LOG(FATAL) << "Direct graph must be ps device, curr is " << curr_loc;
    }
    for (auto loc : locs.second) {
      if (_use_send_recv_loc[loc].find(curr_loc) == _use_send_recv_loc[loc].end()) {
        LOG(FATAL) << "Error to set direct graph location.";
      }
    }
  }
}

void SendRecvPolicy::AddUserDefinedPolicy() {
  UseDefaultPolicy();
}

bool SendRecvPolicy::UseSendRecvMode(Node* src, Node* dst) {
  if (src->assigned_device_name() == "" ||
      dst->assigned_device_name() == "") {
    LOG(FATAL) << "Node's assigned device name is null, src node is "
               << src->DebugString() << ", dst node is " << dst->DebugString();
  }

  return (_use_send_recv_loc[src->assigned_device_name()].find(dst->assigned_device_name())
          != _use_send_recv_loc[src->assigned_device_name()].end());
}

DetectPsToPsCyclePolicy::DetectPsToPsCyclePolicy(Graph* graph)
    : SendRecvPolicy(graph) {
}

DetectPsToPsCyclePolicy::~DetectPsToPsCyclePolicy() {
}

void DetectPsToPsCyclePolicy::AddUserDefinedPolicy() {
  std::unordered_map<std::string, std::unordered_set<std::string> > ps_use_send_recv;
  std::unordered_map<Node*, int> edge_idx;
  std::stack<Node*> node_stack;
  std::unordered_set<Node*> has_visited;
  for (Node* n : graph_->nodes()) {
    if (!n->IsOp()) continue;
    std::string cur_loc(n->assigned_device_name());
    if (cur_loc.find("ps") == std::string::npos ||
      has_visited.find(const_cast<Node*>(n)) != has_visited.end()) {
      continue;
    }
    node_stack.push(n);
    std::unordered_set<Node*> nodes_in_path;
    while (!node_stack.empty()) {
      Node* node = node_stack.top();
      if (edge_idx.find(node) == edge_idx.end()) {
        edge_idx[node] = 0;
      }
      Node* dst = nullptr;
      int i = 0;
      for (const Edge* out_edge : node->out_edges()) {
        if (i++ < edge_idx[node]) continue;
        dst = const_cast<Node*>(out_edge->dst());
        ++edge_idx[node];
        break;
      }

      if (dst == nullptr) {
        has_visited.insert(node);
        node_stack.pop();
        continue;
      }

      // NOTE(jiankeng.pt): "Switch" and "Merge" node will be place to worker in ResetSwitchOpDevice.
      if (!dst->IsOp() ||
          node->def().op() == "Switch" ||
          node->def().op() == "Merge" ||
          node->assigned_device_name().find("ps") == std::string::npos) {
        continue;
      }

      // find a cycle
      if (nodes_in_path.find(dst) != nodes_in_path.end()) {
        ps_use_send_recv[node->assigned_device_name()].insert(dst->assigned_device_name());
        ps_use_send_recv[dst->assigned_device_name()].insert(node->assigned_device_name());
      }

      node_stack.push(dst);
    }
  }
 
}

}  // namespace tensorflow
