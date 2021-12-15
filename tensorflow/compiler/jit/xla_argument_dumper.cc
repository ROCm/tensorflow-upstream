#include <numeric>
#include <fstream>
#include "absl/base/call_once.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/compiler/jit/xla_argument_dumper.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"

namespace tensorflow {

XlaArgumentDumper::XlaArgumentDumper() {
  tensorflow::ReadStringFromEnvVar("TF_XLA_ARGUMENT_CACHE_DIR", "",
                                   &cache_dir_);
}

bool XlaArgumentDumper::AsProto(
    const std::vector<XlaCompiler::Argument>& args,
    XlaArgumensProto& protos) {
  for (const auto& arg: args) {
    XlaArgumentProto* proto = protos.add_xla_arguments();

    if (arg.kind == XlaCompiler::Argument::kConstant) {
      proto->set_kind(CacheKind::kConstant);
      // constant value
      TensorProto* constant_value = proto->mutable_constant_value(); 
      arg.constant_value.AsProtoTensorContent(constant_value);
    } else if (arg.kind == XlaCompiler::Argument::kParameter) {
      proto->set_kind(CacheKind::kParameter);
    } else {
      LOG(ERROR) << "Not support xla shape cache! argument type=" << arg.kind;
      return false;
    }

    // type
    proto->set_type(arg.type);

    // tensor_shape
    TensorShape tensor_shape;
    TensorShapeProto* tensor_shape_proto = proto->mutable_shape();
    if (absl::holds_alternative<xla::Shape>(arg.shape)) {
      xla::Shape xla_shape = absl::get<xla::Shape>(arg.shape);
      if (!XLAShapeToTensorShape(xla_shape, &tensor_shape).ok()) {
         VLOG(0) << "convert shape fail";
         return false;
      }
    } else {
      tensor_shape = absl::get<TensorShape>(arg.shape);
    }
    tensor_shape.AsProto(tensor_shape_proto);
  }
  return true;
}

Status XlaArgumentDumper::DumpXlaArguments(
    const std::vector<XlaCompiler::Argument>& args,
    const uint64 graph_key,
    const string& uuid) {
  if(cache_dir_.empty()) return Status::OK();

  auto env = tensorflow::Env::Default();
 
  string dir = tensorflow::io::JoinPath(cache_dir_, std::to_string(graph_key));
  string file_path = tensorflow::io::JoinPath(dir, std::to_string(tensorflow::Hash64(uuid)) + ".shape");

  if (env->FileExists(file_path).ok()) {
    VLOG(0) << "File exist, return " << file_path;
    return Status::OK();
  }
  
  if (!env->IsDirectory(dir).ok()) {
    auto status = env->RecursivelyCreateDir(dir);
    if (!status.ok() && !env->IsDirectory(dir).ok()) {
      LOG(ERROR) << "Could not create directory " << dir
                 << " for dumping xla shapes: " << status;
      return errors::Unimplemented("Dump xla shape failed ", uuid);
    }
  }

  XlaArgumensProto proto;
  if (!AsProto(args, proto)) return errors::Unimplemented("Dump xla shape failed");

  VLOG(0) << "Dump " << uuid << "; cache dir " << file_path; 
  return WriteTextProto(env, file_path, proto);
}

Status XlaArgumentDumper::ParseFromFile(
    const std::shared_ptr<InputsShapeInfo>& base,
    const uint64 graph_key,
    std::vector<std::vector<XlaCompiler::Argument>>& args_array,
    std::vector<std::shared_ptr<InputsShapeInfo>>& inputs_shape_info_array) {
  if(cache_dir_.empty()) return Status::OK();

  args_array.clear();
  inputs_shape_info_array.clear();
  auto env = tensorflow::Env::Default();
  string dir = tensorflow::io::JoinPath(cache_dir_, std::to_string(graph_key));
  std::vector<string> files;
  if(!env->GetChildren(dir, &files).ok()) return errors::Unimplemented("Get cache files failed", dir);

  for (auto file_path: files) {
    file_path = tensorflow::io::JoinPath(dir, file_path);
    VLOG(1) << "Get file " << file_path;
    XlaArgumensProto protos;
    auto s =tensorflow::ReadTextProto(env, file_path, &protos);
    if (!s.ok()) return s;

    std::vector<XlaCompiler::Argument> args;
    FromProto(protos, args);
    inputs_shape_info_array.push_back(BuildInputsShapeInfo(base, args)); 
    args_array.push_back(args);
  }
  return Status::OK();
}

std::shared_ptr<InputsShapeInfo> XlaArgumentDumper::BuildInputsShapeInfo(
      const std::shared_ptr<InputsShapeInfo>& base, 
      const std::vector<XlaCompiler::Argument>& args) {
  if(base == nullptr) return nullptr;

  std::shared_ptr<InputsShapeInfo> inputs = std::make_shared<InputsShapeInfo>();
  inputs->is_cpu_device = base->is_cpu_device;
  inputs->input_names = base->input_names;
  inputs->output_names = base->output_names;
  inputs->var_indexs = base->var_indexs;
  inputs->const_indexs = base->const_indexs;
  inputs->input_shapes.reserve(args.size());
  inputs->input_tensors.reserve(args.size()); 
  for (int i = 0; i < args.size(); i++) {
    TensorShape shape = absl::get<TensorShape>(args[i].shape);
    inputs->input_shapes.push_back(shape);
    if(std::find(inputs->const_indexs.begin(), inputs->const_indexs.end(), i) != inputs->const_indexs.end()) {
      inputs->input_tensors.push_back(args[i].constant_value);
    } else {
      inputs->input_tensors.push_back(Tensor(args[i].type, shape));
    }
  }
  inputs->refresh_size();
  return inputs;
}

bool XlaArgumentDumper::FromProto(
    const XlaArgumensProto& protos,
    std::vector<XlaCompiler::Argument>& args) {
  args.clear();
  for (int i = 0; i < protos.xla_arguments_size(); i++) {
    const auto& proto = protos.xla_arguments(i);
    XlaCompiler::Argument arg;
  
    // type
    arg.type = proto.type();
   
    // shape
    TensorShape tensor_shape(proto.shape());
    arg.shape = tensor_shape;

    // kind
    if (proto.kind() == CacheKind::kConstant) {
      arg.kind = XlaCompiler::Argument::kConstant;
      // constant_value
      arg.constant_value.FromProto(proto.constant_value());
    } else if (proto.kind() == CacheKind::kParameter) {
      arg.kind = XlaCompiler::Argument::kParameter;
    }

    args.push_back(arg);
  } 
  return true;
}

} // namespace tensorflow
