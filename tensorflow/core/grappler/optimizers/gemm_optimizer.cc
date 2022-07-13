/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/gemm_optimizer.h"

#include <fstream>

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace grappler {

namespace {

std::unordered_set<string> GetUnaryOps() {
  static std::unordered_set<string> ops = {
      "Softmax",
      "Sigmoid",
      "Tanh",
      "Relu"};
  return ops;
}

std::unordered_set<string> GetBinaryOps() {
  std::unordered_set<string> ops = {"IndicatorMatMul"};
  return ops;
}

Status UpdateAllEdge(Graph* graph, Node* new_src_node, Node* old_dst_node) {
  std::vector<Node*> dst_nodes;
  std::vector<int> dst_inputs;
  for (const Edge* e : old_dst_node->out_edges()) {
    dst_nodes.push_back(e->dst());
    dst_inputs.push_back(e->dst_input());
  }
  for (unsigned int i = 0; i < dst_nodes.size(); i++) {
    TF_RETURN_IF_ERROR(
        graph->UpdateEdge(new_src_node, 0, dst_nodes[i], dst_inputs[i]));
  }
  return Status::OK();
}

// Change ->Unpack->Transpose*n-> to ->Transpose->Unpack->
bool FuseTransposesAfterUnpack(Graph* graph) {
  static int count = 0;
  VLOG(2) << "FuseTransposesAfterUnpack";
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack") continue;
    Node* unpack = node;
    std::vector<Node*> transposes;
    std::vector<Node*> transpose_in_1;
    bool can_reorder = true;
    for (Node* out : unpack->out_nodes()) {
      if (out->type_string() != "Transpose") {
        can_reorder = false;
        break;
      }
      transposes.push_back(out);
      Node* n = nullptr;
      out->input_node(1, &n);
      transpose_in_1.push_back(n);
    }
    if (!can_reorder || transposes.size() < 2) continue;
    for (unsigned int i = 1; i < transposes.size(); i++) {
      if (transpose_in_1[i] != transpose_in_1[0]) continue;
    }
    VLOG(2) << "FuseTransposesAfterUnpack: found pattern";

    string prefix = "GemmOptimizer/FuseTransposesAfterUnpack/" +
                    std::to_string(count++);
    string one_name = prefix + "Transpose/Add/one";
    NodeDefBuilder one_builder(one_name, "Const");
    NodeDef one_node;
    Tensor t_one(DT_INT32, TensorShape({1}));
    auto one_data = t_one.tensor<int, 1>();
    one_data(0) = 1;
    Status status =
        one_builder
            .Attr("dtype", t_one.dtype())
            .Attr("value", t_one)
            .Finalize(&one_node);
    if (!status.ok()) {
      LOG(ERROR) << "Const node construction failed with" << status;
      return false;
    }
    one_node.set_device(transpose_in_1[0]->def().device());
    Node* one = graph->AddNode(one_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    one->set_assigned_device_name(transpose_in_1[0]->assigned_device_name());
 
    DataType shape_dtype = transpose_in_1[0]->output_type(0);
    string add_name = prefix + "/Add";
    std::vector<NodeDefBuilder::NodeOut> add_inputs;
    const Edge* param_to_transpose = nullptr;
    transposes[0]->input_edge(1, &param_to_transpose);
    add_inputs.emplace_back(param_to_transpose->src()->name(),
                            param_to_transpose->src_output(), shape_dtype);
    add_inputs.emplace_back(one_name, 0, shape_dtype);
    NodeDefBuilder add_builder(add_name, "Add");
    add_builder.Input(add_inputs[0]);
    add_builder.Input(add_inputs[1]);
    NodeDef add_node;
    status =
        add_builder
            .Attr("T", shape_dtype)
            .Finalize(&add_node);
    if (!status.ok()) {
      LOG(ERROR) << "BiasAdd node construction failed with" << status;
      return false;
    }
    add_node.set_device(transpose_in_1[0]->def().device());
    Node* add = graph->AddNode(add_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    add->set_assigned_device_name(transpose_in_1[0]->assigned_device_name());
    graph->AddEdge(param_to_transpose->src(),
        param_to_transpose->src_output(), add, 0);
    graph->AddEdge(one, 0, add, 1);

    string zero_name = prefix + "Transpose/Concat/zero";
    NodeDefBuilder zero_builder(zero_name, "Const");
    NodeDef zero_node;
    Tensor t_zero(DT_INT32, TensorShape({1}));
    auto zero_data = t_zero.tensor<int, 1>();
    zero_data(0) = 0;
    status =
        zero_builder
            .Attr("dtype", t_zero.dtype())
            .Attr("value", t_zero)
            .Finalize(&zero_node);
    if (!status.ok()) {
      LOG(ERROR) << "Const node construction failed with" << status;
      return false;
    }
    zero_node.set_device(transpose_in_1[0]->def().device());
    Node* zero = graph->AddNode(zero_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    zero->set_assigned_device_name(transpose_in_1[0]->assigned_device_name());

    // Add a Concat Op to generate new shape
    string zero_scalar_name = prefix + "/Transpose/Concat/concat_dim";
    NodeDefBuilder zero_scalar_builder(zero_scalar_name, "Const");
    NodeDef zero_scalar_node;
    Tensor t_zero_scalar(int(0));
    status =
        zero_scalar_builder
            .Attr("dtype", t_zero_scalar.dtype())
            .Attr("value", t_zero_scalar)
            .Finalize(&zero_scalar_node);
    if (!status.ok()) {
      LOG(ERROR) << "Const node construction failed with" << status;
      return false;
    }
    zero_scalar_node.set_device(transpose_in_1[0]->def().device());
    Node* zero_scalar = graph->AddNode(zero_scalar_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    zero_scalar->set_assigned_device_name(
        transpose_in_1[0]->assigned_device_name());

    string concat_name = prefix + "/Transpose/Concat";
    NodeDefBuilder concat_builder(concat_name, "Concat");
    NodeDefBuilder::NodeOut concat_dim(zero_scalar->name(), 0, shape_dtype);
    std::vector<NodeDefBuilder::NodeOut> concat_inputs;
    concat_inputs.emplace_back(zero_name, 0, shape_dtype);
    concat_inputs.emplace_back(add_name, 0, shape_dtype);
    concat_builder.Input(concat_dim);
    concat_builder.Input(concat_inputs);
    NodeDef concat_node;
    status =
        concat_builder
            .Attr("N", 2)
            .Attr("T", shape_dtype)
            .Finalize(&concat_node);
    if (!status.ok()) {
      LOG(ERROR) << "Concat node construction failed with" << status;
      return false;
    }
    concat_node.set_device(transpose_in_1[0]->def().device());
    Node* concat = graph->AddNode(concat_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    concat->set_assigned_device_name(
        transpose_in_1[0]->assigned_device_name());
    graph->AddEdge(zero_scalar, 0, concat, 0);
    graph->AddEdge(zero, 0, concat, 1);
    graph->AddEdge(add, 0, concat, 2);

    // Add a new Transpose Op
    string transpose_name = prefix + "/Transpose";
    NodeDefBuilder transpose_builder(transpose_name, "Transpose");
    std::vector<NodeDefBuilder::NodeOut> transpose_inputs;
    const Edge* to_unpack;
    unpack->input_edge(0, &to_unpack);
    int src_output = to_unpack->src_output();
    Node* unpack_in = to_unpack->src();
    transpose_inputs.emplace_back(unpack_in->name(), src_output,
                                  transposes[0]->input_type(0));
    transpose_inputs.emplace_back(concat->name(), 0, shape_dtype);
    transpose_builder.Input(transpose_inputs[0]);
    transpose_builder.Input(transpose_inputs[1]);
    NodeDef transpose_node;
    status =
        transpose_builder
            .Attr("T", transposes[0]->input_type(0))
            .Attr("Tperm", shape_dtype)
            .Finalize(&transpose_node);
    if (!status.ok()) {
      LOG(ERROR) << "Reshape node construction failed with" << status;
      return false;
    }
    transpose_node.set_device(transposes[0]->def().device());
    Node* transpose = graph->AddNode(transpose_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    transpose->set_assigned_device_name(
        transposes[0]->assigned_device_name());
    graph->AddEdge(unpack_in, src_output, transpose, 0);
    graph->AddEdge(concat, 0, transpose, 1);
 
    graph->UpdateEdge(transpose, 0, unpack, 0);
    for (Node* t : transposes) {
      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : t->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      const Edge* e = nullptr;
      t->input_edge(0, &e);
      int src_output = e->src_output();
      for (unsigned int i = 0; i < dst_nodes.size(); i++) {
        graph->UpdateEdge(unpack, src_output, dst_nodes[i], dst_inputs[i]);
      }
      graph->RemoveNode(t);
    }
    changed= true;
  }
  return changed;
}

// ->Unpack->BinaryOp*n to ->BinaryOp->Unpack
bool FuseBinaryOpsAfterUnpack(Graph* graph) {
  static int count = 0;
  VLOG(2) << "FuseBinaryOpsAfterUnpack";
  bool changed = false;
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  const std::unordered_set<string> binary_op_set = GetBinaryOps();
  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack") continue;
    // group ops based on its inputs
    std::vector<Node*> binary_ops;
    std::map<string, std::vector<Node*>> binary_ops_m;
    std::vector<std::pair<Node*, int>> common_extra_input;
    const Node* another_node = nullptr;
    bool can_fuse = true;
    string binary_type;
    for (Node* out : node->out_nodes()) {
      string out_type = out->type_string();
      if (binary_type.empty()) {
        binary_type = out->type_string();
        if (binary_op_set.find(binary_type) == binary_op_set.end()) {
          can_fuse = false;
          break;
        }
      }
      if (out->type_string() != binary_type) {
        can_fuse = false;
        break;
      }

      if (out->num_inputs() < 2) {
        can_fuse = false;
        break;
      }

      bool has_same_another_input = true;
      bool another_input_from_unpack_or_const = true;
      bool has_common_extra_input = true;
      for (int i = 0; i < out->num_inputs(); ++i) {
        const Edge* edge = nullptr;
        out->input_edge(i, &edge);
        Node* n = edge->src();
        if (i < 2) {
          if (n != node) {
            if (another_node == nullptr) {
              string type = n->type_string();
              if (type == "Unpack" || type == "Const") {
                another_node = n;
              } else {
                another_input_from_unpack_or_const = false;
                break;
              }
            } else {
              if (n != another_node) {
                has_same_another_input = false;
              }
            }
          }
        } else {
          size_t vec_idx = i - 2;
          if (vec_idx >= common_extra_input.size()) {
            common_extra_input.emplace_back(std::make_pair(n, edge->src_output()));
          } else {
            if (n != common_extra_input[vec_idx].first ||
                edge->src_output() != common_extra_input[vec_idx].second) {
              has_common_extra_input = false;
            }
          }
        }
      }

      if (has_same_another_input && another_input_from_unpack_or_const && has_common_extra_input)
        binary_ops.push_back(out);
    }

    if (!can_fuse || binary_ops.size() < 2) continue;

    for (Node* b : binary_ops) {
      for (int i = 0; i < std::min(b->num_inputs(), 2); ++i) {
        const Node* n = nullptr;
        b->input_node(i, &n);
        if (n != node) {
          string key;
          if (another_node->type_string() == "Unpack") {
            key = n->name();
          } else {
            TensorShapeProto s = n->def().attr().at("value").
                                 tensor().tensor_shape();
            for (int i = 0; i < s.dim_size(); i++) {
              key += std::to_string(s.dim(i).size()) + ",";
            }
          }
          if (binary_ops_m.find(key) != binary_ops_m.end()) {
            binary_ops_m[key].push_back(b);
          } else {
            std::vector<Node*> ins;
            ins.push_back(b);
            binary_ops_m[key] = ins;
          }
          break;
        }
      }
    }
    VLOG(2) << "FuseBinaryOpsAfterUnpack: found pattern";

    // Fuse binary_ops in each group.
    // For each group, do:
    // (1) add two Pack nodes to stack inputs on both sides respectively,
    // (2) add a new node to replace old binary_ops, and
    // (3) add a Unpack node to split result.
    DataType dtype = node->output_type(0);
    std::map<string, std::vector<Node*>>::iterator iter;
    iter = binary_ops_m.begin();
    while (iter != binary_ops_m.end()) {
      std::vector<Node *> *binary_ops_group = &(iter->second);
      if (binary_ops_group->size() < 2) iter++;
      std::sort(binary_ops_group->begin(), binary_ops_group->end(),
                [node](Node *a, Node *b) {
                  const Edge *e = nullptr;
                  a->input_edge(0, &e);
                  int a_src_output = e->src_output();
                  b->input_edge(0, &e);
                  int b_src_output = e->src_output();
                  return a_src_output < b_src_output;
                });
      std::vector<const Edge *> inputs[2];
      VLOG(2) << "the following ops are fused: ";
      for (Node *b : *binary_ops_group) {
        for (int i = 0; i < 2; ++i) {
          VLOG(2) << b->name();

          const Edge *e = nullptr;
          b->input_edge(i, &e);
          inputs[e->dst_input()].push_back(e);
        }
      }

      // Add two Pack nodes to group on two sides, respectively
      Node *packs[2];
      string pack_names[2];
      string prefix = "GemmOptimizer/FuseBinaryOpsAfterUnpack/" +
                      std::to_string(count++);
      pack_names[0] = prefix + "/Pack_0";
      pack_names[1] = prefix + "/Pack_1";
      Status status;
      for (int i = 0; i < 2; i++) {
        std::vector<NodeDefBuilder::NodeOut> pack_inputs;
        for (const Edge *e : inputs[i]) {
          string s = e->src()->name();
          pack_inputs.emplace_back(s, e->src_output(), dtype);
        }
        NodeDefBuilder pack_builder(pack_names[i], "Pack");
        pack_builder.Input(pack_inputs);
        NodeDef pack_node;
        status =
            pack_builder
                .Attr("N", (int) inputs[i].size())
                .Attr("T", dtype)
                .Attr("axis", 0)
                .Finalize(&pack_node);
        if (!status.ok()) {
          LOG(ERROR) << "Pack node construction failed with" << status;
          return false;
        }
        pack_node.set_device(inputs[i][0]->src()->def().device());
        packs[i] = graph->AddNode(pack_node, &status);
        if (!status.ok()) {
          LOG(ERROR) << "Adding node failed " << status;
          return false;
        }
        packs[i]->set_assigned_device_name(
            inputs[i][0]->src()->assigned_device_name());
        for (unsigned int j = 0; j < inputs[i].size(); j++) {
          graph->AddEdge(inputs[i][j]->src(), inputs[i][j]->src_output(),
                         packs[i], j);
        }
      }

      // Add a new BatchMatMulV2
      std::vector<NodeDefBuilder::NodeOut> binary_op_inputs;
      binary_op_inputs.emplace_back(pack_names[0], 0, dtype);
      binary_op_inputs.emplace_back(pack_names[1], 0, dtype);
      if (!common_extra_input.empty()) {
        for (size_t i = 0; i < common_extra_input.size(); ++i) {
          const Node *extra_node = common_extra_input[i].first;
          int extra_src_idx = common_extra_input[i].second;
          binary_op_inputs.emplace_back(extra_node->name(), extra_src_idx, extra_node->output_type(extra_src_idx));
        }
      }
      string binary_op_name = prefix;
      string type = (*binary_ops_group)[0]->type_string();
      string new_type;
      if (type == "MatMul" ||
          type == "BatchMatMul" ||
          type == "BatchMatMulV2") {
        binary_op_name += "/BatchMatMulV2";
        new_type = "BatchMatMulV2";
      } else if (type == "IndicatorMatMul") {
        binary_op_name += "/ParallelIndicatorMatMul";
        new_type = "ParallelIndicatorMatMul";
      } else {
        binary_op_name += "/" + type;
        new_type = type;
      }
      NodeDefBuilder binary_op_builder(binary_op_name, new_type);
      for (size_t i = 0; i < binary_op_inputs.size(); ++i) {
        binary_op_builder.Input(binary_op_inputs[i]);
      }
      NodeDef binary_op_node;
      bool transpose_a = false;
      bool transpose_b = false;
      if (type == "MatMul") {
        transpose_a = (*binary_ops_group)[0]->def().attr().at("transpose_a").b();
        transpose_b = (*binary_ops_group)[0]->def().attr().at("transpose_b").b();
      } else if (type == "BatchMatMul" || type == "BatchMatMulV2" || type == "IndicatorMatMul") {
        transpose_a = (*binary_ops_group)[0]->def().attr().at("adj_x").b();
        transpose_b = (*binary_ops_group)[0]->def().attr().at("adj_y").b();
      }

      if (type == "MatMul" ||
          type == "BatchMatMul" ||
          type == "BatchMatMulV2") {
        status =
            binary_op_builder
                .Attr("adj_x", transpose_a)
                .Attr("adj_y", transpose_b)
                .Attr("T", dtype)
                .Finalize(&binary_op_node);

      } else if (type == "IndicatorMatMul") {
        status =
            binary_op_builder
                .Attr("adj_x", transpose_a)
                .Attr("adj_y", transpose_b)
                .Attr("parallel_num", (int) (*binary_ops_group).size())
                .Attr("T", dtype)
                .Finalize(&binary_op_node);
      } else {
        status =
            binary_op_builder
                .Attr("T", dtype)
                .Finalize(&binary_op_node);
      }
      if (!status.ok()) {
        LOG(ERROR) << "BatchMatMulV2 node construction failed with" << status;
        return false;
      }
      binary_op_node.set_device((*binary_ops_group)[0]->def().device());
      Node* binary_op = graph->AddNode(binary_op_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      binary_op->set_assigned_device_name((*binary_ops_group)[0]->
                                          assigned_device_name());
      graph->AddEdge(packs[0], 0, binary_op, 0);
      graph->AddEdge(packs[1], 0, binary_op, 1);

      // Add an Unpack node to split result
      string unpack_name = prefix + "/Unpack" ;
      NodeDefBuilder::NodeOut unpack_input(binary_op_name, 0, dtype);
      NodeDefBuilder unpack_builder(unpack_name, "Unpack");
      unpack_builder.Input(unpack_input);
      NodeDef unpack_node;
      status =
          unpack_builder
              .Attr("num", (int)(*binary_ops_group).size())
              .Attr("T", dtype)
              .Attr("axis", 0)
              .Finalize(&unpack_node);
      if (!status.ok()) {
        LOG(ERROR) << "Unpack node construction failed with" << status;
        return false;
      }
      unpack_node.set_device((*binary_ops_group)[0]->def().device());
      Node* unpack = graph->AddNode(unpack_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      unpack->set_assigned_device_name((*binary_ops_group)[0]->
                                       assigned_device_name());
      graph->AddEdge(binary_op, 0, unpack, 0);
   
      // Add edges to forward split results to nodes after original binary_ops,
      // and remove original binary_ops
      int index = 0;
      for (Node* b : *binary_ops_group) {
        std::vector<Node*> dst_nodes;
        std::vector<int> dst_inputs;
        for (const Edge* e : b->out_edges()) {
          dst_nodes.push_back(e->dst());
          dst_inputs.push_back(e->dst_input());
        }
        for (unsigned int i = 0; i < dst_nodes.size(); i++) {
          graph->UpdateEdge(unpack, index, dst_nodes[i], dst_inputs[i]);
        }
        graph->RemoveNode(b);
        index++;
      }
 
      changed = true;
      iter++;
    }
  }
 
  return changed;
}

// ->Unpack->UnaryOp*n to ->UnaryOp->Unpack
bool FuseUnaryOpsAfterUnpack(Graph* graph) {
  VLOG(2) << "FuseUnaryOpsAfterUnpack";
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  const std::unordered_set<string> unary_op_set = GetUnaryOps();
  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack") continue;
    Node* unpack = node;
    std::vector<Node*> unary_ops;
    bool can_reorder = true;
    string unary_type;
    for (Node* out : unpack->out_nodes()) {
      if (unary_type.empty()) {
        unary_type = out->type_string();
        if (unary_op_set.find(unary_type) == unary_op_set.end()) {
          can_reorder = false;
          break;
        }
      }
      if (out->type_string() != unary_type) {
        can_reorder = false;
        break;
      }
      unary_ops.push_back(out);
    }
    if (!can_reorder || unary_ops.size() < 2) continue;
    VLOG(2) << "FuseUnaryOpsAfterUnpack: found pattern";
    bool is_first = true;
    for (Node* u : unary_ops) {
      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : u->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      const Edge* e = nullptr;
      u->input_edge(0, &e);
      int src_output = e->src_output();
      for (unsigned int i = 0; i < dst_nodes.size(); i++) {
        graph->UpdateEdge(unpack, src_output, dst_nodes[i], dst_inputs[i]);
      }
      if (is_first) {
        is_first = false;
      } else {
        graph->RemoveNode(u);
      }
    }
    const Edge* to_unpack = nullptr;
    unpack->input_edge(0, &to_unpack);
    graph->UpdateEdge(to_unpack->src(), to_unpack->src_output(),
                      unary_ops[0], 0);
    graph->UpdateEdge(unary_ops[0], 0, unpack, 0);
    changed = true;
  }

  return changed;
}

void RemoveDeadUnpacksAndPacks(Graph* graph) {
  VLOG(2) << "RemoveDeadUnpacksAndPacks";
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack" &&
        node->type_string() != "Pack") continue;
    if (node->out_edges().size() == 0) {
      graph->RemoveNode(node);
    }
  }
}

bool RemoveUnpacksAndPacks(Graph* graph) {
  VLOG(2) << "RemoveUnpacksAndPacks";
  bool changed = false;
  static int count = 0;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack") continue;
    Node* unpack = node;

    bool can_remove = true;
    int num_split = -1;
    int size_per_split = -1;
    std::map<Node*, int> unpack_split_map;
    for (Node* out : unpack->out_nodes()) {
      if (unpack_split_map.find(out) != unpack_split_map.end()) continue;
      if (out->type_string() != "Pack") {
        can_remove = false;
        break;
      }
      // Check Unpack's outputs are grouped evenly
      if (num_split == -1) {
        int unpack_num = unpack->num_outputs();
        size_per_split = out->num_inputs();
        num_split = unpack_num / size_per_split;
        if ((size_per_split *  num_split) != unpack_num) {
          can_remove = false;
          break;
        }
      }

      // Make sure each Pack takes a full group of Unpack's outputs.
      // This is realized by using the following two checks:
      int src_output = -1;
      for (int i = 0; i < out->num_inputs(); i++) {
        const Edge* e = nullptr;
        out->input_edge(i, &e);
        int temp = e->src_output();
        if (src_output != -1) {
          // (1) check order
          if (temp != (src_output + 1)) {
            can_remove = false;
            break;
          }
        } else {
          // (2) check size and begin index
          if ((temp % size_per_split != 0) ||
              (out->num_inputs() != size_per_split)) {
            can_remove = false;
            break;
          }
          // This Unpack is replaced by Split:(temp/size_per_split)
          unpack_split_map[out] = temp / size_per_split;
        }
        src_output = temp;
      }
      if (!can_remove) break;
    }

    if (!can_remove || unpack_split_map.size() == 0) continue;
    VLOG(2) << "RemoveUnpacksAndPacks: found pattern";

    // for Unpack->Pack*n, insert a Split Op, and
    // repalace Packs' outputs with Split's outputs.
    const Edge* to_unpack = nullptr;
    unpack->input_edge(0, &to_unpack);
    Node* unpack_in = to_unpack->src();
    int unpack_src_output = to_unpack->src_output();
    Node* split = nullptr;
    if (num_split > 1) {
      string prefix = "GemmOptimizer/RemoveUnpacksAndPacks/" +
                      std::to_string(count++);
      string zero_name = prefix + "/Split/zero";
      NodeDefBuilder zero_builder(zero_name, "Const");
      NodeDef zero_node;
      Tensor t_zero(int(0));
      Status status =
          zero_builder
              .Attr("dtype", t_zero.dtype())
              .Attr("value", t_zero)
              .Finalize(&zero_node);
      if (!status.ok()) {
        LOG(ERROR) << "Const node construction failed with" << status;
        return false;
      }
      zero_node.set_device(unpack->def().device());
      Node* zero = graph->AddNode(zero_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      zero->set_assigned_device_name(unpack->assigned_device_name());

      std::vector<NodeDefBuilder::NodeOut> split_inputs;
      split_inputs.emplace_back(zero_name, 0, t_zero.dtype());
      split_inputs.emplace_back(unpack_in->name(), unpack_src_output,
                                unpack_in->output_type(unpack_src_output));
      string split_name = prefix + "/Split";
      NodeDefBuilder split_builder(split_name, "Split");
      split_builder.Input(split_inputs[0]);
      split_builder.Input(split_inputs[1]);
      NodeDef split_node;
      status =
          split_builder
              .Attr("T", unpack_in->output_type(unpack_src_output))
              .Attr("num_split", num_split)
              .Finalize(&split_node);
      if (!status.ok()) {
        LOG(ERROR) << "Split node construction failed with" << status;
        return false;
      }
      split_node.set_device(unpack->def().device());
      split = graph->AddNode(split_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      split->set_assigned_device_name(unpack->assigned_device_name());
      graph->AddEdge(zero, 0, split, 0);
      graph->AddEdge(unpack_in, unpack_src_output, split, 1);
    }

    // for Unpack->Pack pair, update graph directly
    std::map<Node*, int>::iterator it;
    for (it = unpack_split_map.begin();
         it != unpack_split_map.end(); it++) {
      Node* pack = it->first;

      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : pack->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      for (unsigned int i = 0; i < dst_nodes.size(); i++) {
        if (num_split == 1) {
         graph->UpdateEdge(unpack_in, unpack_src_output,
                           dst_nodes[i], dst_inputs[i]);
        } else {
         graph->UpdateEdge(split, it->second,
                           dst_nodes[i], dst_inputs[i]);

        }
      }
      graph->RemoveNode(pack);
    }
    changed = true;
  }
  RemoveDeadUnpacksAndPacks(graph);
  return changed;
}

bool FuseGatherBeforeMatMul(Graph* graph) {
  VLOG(2) << "FuseGatherBeforeMatMul";
  
  static int count = 0;
  bool changed = false;
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  for (Node* node : nodes) {
    if (node->type_string() != "BatchMatMulV2") continue;
    Node *matmul = node;
    // Check Gather
    Node *gather = nullptr;
    matmul->input_node(0, &gather);
    if (!graph->IsValidNode(gather).ok()) continue;
    string type = gather->type_string();
    if (type != "GatherV2") continue;
    // Check axis
    Node* axis = nullptr;
    gather->input_node(2, &axis);
    if (!axis) continue;
    type = axis->type_string();
    if (type != "Const") continue;
    if (axis->def().attr().at("value").i() != 0) continue;

    // Prepare op name
    string prefix = "GemmOptimizer/FuseGatherBeforeMatMul/" + matmul->name();
    string op_name = prefix + "/IndicatorMatMul_" + std::to_string(count++);

    // Build NodeDef
    std::vector<NodeDefBuilder::NodeOut> ind_matmul_inputs;
    Node *input_nodes[3] = {nullptr, nullptr, nullptr};
    int input_idx[3] = {0, 0, 0};

    const Edge* gather_input_0;
    gather->input_edge(0, &gather_input_0);
    input_nodes[0] = gather_input_0->src();
    input_idx[0] = gather_input_0->src_output();

    const Edge* matmul_input_1;
    matmul->input_edge(1, &matmul_input_1);
    input_nodes[1] = matmul_input_1->src();
    input_idx[1] = matmul_input_1->src_output();

    const Edge* gather_input_1;
    gather->input_edge(1, &gather_input_1);
    input_nodes[2] = gather_input_1->src();
    input_idx[2] = gather_input_1->src_output();

    if (!input_nodes[0] || !input_nodes[1] || !input_nodes[2]) continue;
    DataType dtype = input_nodes[0]->output_type(0);
    DataType ind_dtype = input_nodes[2]->output_type(0);
    ind_matmul_inputs.emplace_back(input_nodes[0]->name(), input_idx[0], dtype);
    ind_matmul_inputs.emplace_back(input_nodes[1]->name(), input_idx[1], dtype);
    ind_matmul_inputs.emplace_back(input_nodes[2]->name(), input_idx[2], ind_dtype);
    NodeDefBuilder ind_matmul_builder(op_name, "IndicatorMatMul");
    ind_matmul_builder.Input(ind_matmul_inputs[0]);
    ind_matmul_builder.Input(ind_matmul_inputs[1]);
    ind_matmul_builder.Input(ind_matmul_inputs[2]);

    NodeDef ind_matmul_def;
    bool transpose_a = false, transpose_b = false;
    if (matmul->def().attr().find("adj_x") != matmul->def().attr().end()) {
      transpose_a = matmul->def().attr().at("adj_x").b();
    }
    if (matmul->def().attr().find("adj_y") != matmul->def().attr().end()) {
      transpose_b = matmul->def().attr().at("adj_y").b();
    }
    Status status = ind_matmul_builder
        .Attr("adj_x", transpose_a)
        .Attr("adj_y", transpose_b)
        .Attr("T", dtype)
        .Device(matmul->def().device())
        .Finalize(&ind_matmul_def);
    if (!status.ok()) {
      LOG(ERROR) << "IndicatorMatMul node construction failed with " << status;
      return changed;
    }

    // Insert IndicatorMatMul node
    Node *ind_matmul_node = graph->AddNode(ind_matmul_def, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return changed;
    }
    // Update input edge
    graph->AddEdge(input_nodes[0], input_idx[0], ind_matmul_node, 0);
    graph->AddEdge(input_nodes[1], input_idx[1], ind_matmul_node, 1);
    graph->AddEdge(input_nodes[2], input_idx[2], ind_matmul_node, 2);
    // Update output edge
    UpdateAllEdge(graph, ind_matmul_node, matmul);
    // Remove useless node
    auto RemoveNodeSafely = [&](Node* node) {
      if (node->out_edges().empty()) {
        graph->RemoveNode(node);
      }
    };
    RemoveNodeSafely(matmul);
    RemoveNodeSafely(gather);
    RemoveNodeSafely(axis);
    changed = true;
  }
  return changed;
}

void FuseGatherGemm(Graph* graph) {
  bool gemm_fusion = true;
  ReadBoolFromEnvVar("TF_ENABLE_GEMM_FUSION", true, &gemm_fusion);
  if (!gemm_fusion) return;

  if (!FuseGatherBeforeMatMul(graph)) return;
  while(1) {
    bool graph_changed =
        FuseBinaryOpsAfterUnpack(graph) ||
        FuseTransposesAfterUnpack(graph) ||
        FuseUnaryOpsAfterUnpack(graph) ||
        RemoveUnpacksAndPacks(graph);
    if (!graph_changed) break;
  }
}
}  // end namespace

Status GemmOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  VLOG(1) << "GemmOptimizer is on.";
  static int pass = 0;
  if (VLOG_IS_ON(1)) {
    DumpGraphDefToFile("before_gemm", item.graph);
    std::fstream f;
    f.open("before_gemm_" + std::to_string(pass) + ".pb",
           std::fstream::out | std::fstream::binary);
    f << item.graph.SerializeAsString();
    f.close();
  }

  // convert graphdef to graph
  FunctionLibraryDefinition flib(OpRegistry::Global(), item.graph.library());
  Graph graph(flib);
  Status status = ConvertGraphDefToGraph(GraphConstructorOptions(),
                                         item.graph, &graph);
  if (!status.ok()) {
    LOG(WARNING) << "ConvertGraphDefToGraph failed: " << status.ToString();
    *optimized_graph = item.graph;
    return Status::OK();
  }

  FuseGatherGemm(&graph);

  // convert graph to graphdef
  graph.ToGraphDef(optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();

  if (VLOG_IS_ON(2)) {
    DumpGraphDefToFile("after_gemm", *optimized_graph);
    std::fstream f;
    f.open("after_gemm_" + std::to_string(pass) + ".pb",
           std::fstream::out | std::fstream::binary);
    f << optimized_graph->SerializeAsString();
    f.close();
  }
  pass++;
  return Status::OK();
}

void GemmOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
