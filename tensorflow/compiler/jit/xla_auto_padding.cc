#include <numeric>
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
#include "tensorflow/compiler/jit/xla_auto_padding.h"
#include "tensorflow/compiler/jit/xla_compilation_cache.h"

namespace tensorflow {

using namespace xla_padding_rule;

XlaAutoPadding::XlaAutoPadding(XlaCompilationCache* cache, std::string name) :
    cache_(cache),
    name_(name) {
  arg_dumper_ = std::make_shared<XlaArgumentDumper>();
}

Status XlaAutoPadding::CompileImpl(
        const XlaCompiler::Options& options,
        const NameAttrList& function,
        std::vector<XlaCompiler::Argument> args,
        const std::function<Status(
                 XlaCompiler* compiler,
                 XlaCompiler::CompilationResult*,
                 const XlaCompiler::CompileOptions& compile_options,
                 const NameAttrList& function,
                 absl::Span<const XlaCompiler::Argument> args,
                 std::shared_ptr<InputsShapeInfo> inputs_shape_info)>& compile_fn,
        const XlaCompiler::CompileOptions& compile_options,
        absl::optional<int64> compile_threshold,
        std::shared_ptr<InputsShapeInfo> inputs_shape_info){
  auto ret = cache_->CompileImpl(options, function, args,
                     compile_options, compile_fn,
                     compile_threshold,
                     nullptr,
                     nullptr,
                     inputs_shape_info);
  // dump xla arguments for warm up
  if (inputs_shape_info){
    arg_dumper_->DumpXlaArguments(args, graph_key_, inputs_shape_info->padded_shape_uuid());
  }
  return ret;
}

Status XlaAutoPadding::FillAndFindCacheShape(
  const std::map<int, Tensor>& constant_args,
  const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
  std::vector<XlaCompiler::Argument>* args,
  std::shared_ptr<InputsShapeInfo> inputs_shape_info) {
  FillInputShapes(constant_args, variable_args, 
                    ctx, args, inputs_shape_info);
  return GetPaddingCachedShape(inputs_shape_info);
}

Status XlaAutoPadding::Warmup(
    const XlaCompiler::Options& options, const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args,
    const XlaCompiler::CompileOptions& compile_options,
    const std::function<Status(XlaCompiler* compiler,
            XlaCompiler::CompilationResult*,
            const XlaCompiler::CompileOptions& compile_options,
            const NameAttrList& function,
            absl::Span<const XlaCompiler::Argument> args,
            std::shared_ptr<InputsShapeInfo> inputs_shape_info)>& compile_fn,
    absl::optional<int64> compile_threshold,
    std::shared_ptr<InputsShapeInfo> inputs_shape_info) {
  has_warmup_ = true;

  // Build signature for cluster
  TF_ASSIGN_OR_RETURN(XlaCompilationCache::Signature signature, XlaCompilationCache::BuildSignatureNoShape(function, args));
  graph_key_ = tensorflow::Hash64(name_ + "_" + signature.HumanString());

  // Read warmup files
  std::vector<std::vector<XlaCompiler::Argument>> args_array;
  std::vector<std::shared_ptr<InputsShapeInfo>> inputs_shape_info_array;
  arg_dumper_->ParseFromFile(inputs_shape_info, graph_key_, 
                                args_array, inputs_shape_info_array);

  LOG(INFO) << name_ << " warmup " << name_ + "_" + signature.HumanString() << " with hash:" << graph_key_
            << " cache file size " << args_array.size();

  // Warmup by files
  for(int i = 0; i < args_array.size(); i++) {
    LOG(INFO) << name_ << " warmup from file " << inputs_shape_info_array[i]->uuid();
    CompileImpl(options, function, args_array[i], compile_fn,
                compile_options, compile_threshold, inputs_shape_info_array[i]);
  }
  LOG(INFO) << name_ << " warmup from file finish! total cache size=" << args_array.size(); 
  return Status::OK();
}

Status XlaAutoPadding::Compile(
    const XlaCompiler::Options& options, const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args,
    const XlaCompiler::CompileOptions& compile_options,
    const std::function<Status(XlaCompiler* compiler,
            XlaCompiler::CompilationResult*,
            const XlaCompiler::CompileOptions& compile_options,
            const NameAttrList& function,
            absl::Span<const XlaCompiler::Argument> args,
            std::shared_ptr<InputsShapeInfo> inputs_shape_info)>& compile_fn,
    absl::optional<int64> compile_threshold,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable,
    std::shared_ptr<InputsShapeInfo> inputs_shape_info) {
  *out_executable = nullptr;
  *out_compilation_result = nullptr;
  // If the backend compile is running, we dont schedule new task to it.
  // Because the compile may take several seconds, during which time,
  // we may schedule lots of same task

  // If we already known the cluster cannot compile, return
  if (inputs_shape_info->graph_properties != nullptr &&
      inputs_shape_info->graph_properties->GetXlaPaddingState() == PaddingState::INVALID) {
    LOG(WARNING) << "[" << name_ << "]: cannot compile, return";
    return Status(tensorflow::error::UNIMPLEMENTED, "Cannot compile");
  }

  // If this is first compile and has warmup file, then dont compile this
  // beacuse first compile is warmup data, the inputs shape has no use
  {
    mutex_lock lock(warmup_mu_);
    if (! has_warmup_) {
      Warmup(options, function, args, compile_options, compile_fn,
          compile_threshold, inputs_shape_info);
      return Status::OK();
    }
  }

  std::vector<XlaCompiler::Argument> unconst_args;
  unconst_args.reserve(args.size());
  for (auto& arg: args) {
    unconst_args.push_back(arg);
  }

  LOG(INFO) << name_ << " begin to compile " 
          << " " << inputs_shape_info->DebugString();
  XlaAutoPadding::CompileImpl(
                     options, function, unconst_args, compile_fn,
                     compile_options, compile_threshold, 
                     inputs_shape_info);
  return Status::OK();
}

Status XlaAutoPadding::FillInputShapes(
    const std::map<int, Tensor>& constant_args,
    const std::map<int, OptionalTensor>& variable_args, 
    OpKernelContext* ctx,
    std::vector<XlaCompiler::Argument>* args,
    std::shared_ptr<InputsShapeInfo> inputs_shape_info) {

  inputs_shape_info->input_tensors.resize(ctx->num_inputs());
  inputs_shape_info->input_shapes.resize(ctx->num_inputs());
  inputs_shape_info->var_indexs.clear();
  inputs_shape_info->var_indexs.reserve(ctx->num_inputs());
  inputs_shape_info->const_indexs.clear();
  inputs_shape_info->const_indexs.reserve(ctx->num_inputs());
  for (int64 input_num = 0; input_num < ctx->num_inputs(); ++input_num) {
    XlaCompiler::Argument& arg = (*args)[input_num];
    
    if (constant_args.count(input_num) > 0) {
      const Tensor& input = constant_args.at(input_num);
      inputs_shape_info->input_tensors[input_num] = input;
      inputs_shape_info->input_shapes[input_num] = input.shape();
      inputs_shape_info->const_indexs.push_back(input_num);
    }else if (variable_args.count(input_num) == 0) {
      const Tensor& input = ctx->input(input_num);
      if (input.NumElements() <= 0) continue;

      inputs_shape_info->input_shapes[input_num] = input.shape();
      inputs_shape_info->input_tensors[input_num] = input;
      inputs_shape_info->var_indexs.push_back(input_num);
    } 
  }
  inputs_shape_info->refresh_size();
  return Status::OK();
}

void XlaAutoPadding::ConvertXLAShapeToTensors(
    const absl::Span<const XlaCompiler::Argument>& args,
    std::vector<Tensor>& feed) {
  for (auto arg: args) {
    if (arg.kind == XlaCompiler::Argument::kConstant) {
      feed.push_back(arg.constant_value);
      continue;
    }

    if (absl::holds_alternative<xla::Shape>(arg.shape)) {
      xla::Shape xla_shape = absl::get<xla::Shape>(arg.shape);
      TensorShape tensor_shape;
      if (XLAShapeToTensorShape(xla_shape, &tensor_shape).ok()) {
        Tensor tensor(arg.type, tensor_shape);
        feed.push_back(tensor);
      }
    } else {
      TensorShape tensor_shape = absl::get<TensorShape>(arg.shape);
      Tensor tensor(arg.type, tensor_shape);
      feed.push_back(tensor);
    }
  }
}

Status XlaAutoPadding::CompileCallback(
 const XlaCompiler::Options& options,
    absl::Span<const XlaCompiler::Argument> args,
    XlaCompiler::CompilationResult* out_compilation_result,
    xla::LocalExecutable* out_executable,
    std::shared_ptr<InputsShapeInfo> inputs_shape_info) {
  if (inputs_shape_info == nullptr) return Status::OK();

  // For the first time compile, init shape infer entity
  // For single cluster, the size of thread_pools and shape infer
  // Entitys must be the same, so that each thread can use its own entiy
  {
    mutex_lock lock(graph_properties_mu_);
    if (graph_properties_ == nullptr) {
      InitShapeInferEntity(inputs_shape_info);
    }
  }

  // Add Tensor shape args to cached_args_
  std::vector<Tensor> cached_tensors;
  ConvertXLAShapeToTensors(args, cached_tensors);
  // Lock cached_input_shapes_ to insert new shape
  FillShapeInferCtx(inputs_shape_info, options.device_type.type_string());
  inputs_shape_info->out_compilation_result = out_compilation_result;
  inputs_shape_info->out_executable = out_executable;

  // After padding, check if the cluster is valid for padding
  std::vector<int> base_dims;
  {
    mutex_lock lock(cached_input_shapes_mu_);      
    base_dims.assign(cached_inputs_shapes_[0]->all_dims.begin(), cached_inputs_shapes_[0]->all_dims.end());
  }
  TF_RETURN_IF_ERROR(ValidateCluster(base_dims, inputs_shape_info));

  VLOG(1) << name_ << " Compile Finish! " << inputs_shape_info->DebugString();
  {
    mutex_lock lock(cached_input_shapes_mu_);    
    // remove unused cache
    for (int i = 0; i < cached_inputs_shapes_.size(); i++) {
      if (cached_inputs_shapes_[i]->first_request) {
        cached_inputs_shapes_.erase(cached_inputs_shapes_.begin() + i);
        break;
      }
    }  

    // Replace origin pushed ptr, whose out_executable is nullptr 
    bool exist = false;
    for(int i = 0; i < cached_inputs_shapes_.size(); i++) {
      if (cached_inputs_shapes_[i]->padded_shape_uuid() == inputs_shape_info->padded_shape_uuid()) {
        cached_inputs_shapes_[i] = inputs_shape_info;
        exist = true;
      }
    }
    if (!exist) {
      cached_inputs_shapes_.push_back(inputs_shape_info);
      SortCaches();
    }

    // Print the cached shapes
    LOG(INFO) << name_ << " print XLA cached shapes**********************";
    for (int i = 0 ; i < cached_inputs_shapes_.size(); i++) {
      LOG(INFO) << name_ << " cached " << i + 1 << "/" 
                << cached_inputs_shapes_.size() 
                << " " << cached_inputs_shapes_[i]->DebugString();
    }
  }
  
  return Status::OK();
}

void XlaAutoPadding::InitShapeInferEntity(
      std::shared_ptr<InputsShapeInfo> inputs_shape_info) {
  CHECK(inputs_shape_info != nullptr) << name_;
  CHECK(inputs_shape_info->graph != nullptr) << name_;
  CHECK(graph_properties_ == nullptr) << name_;

  const auto& input_names = inputs_shape_info->input_names;
  VLOG(0) << "####### Begin to Init GrapplerItem." << name_;
  //Init the compiler_cache grapper_item when first compile the xla cluseter
  std::vector<std::pair<string, Tensor>> feed;
  feed.reserve(input_names.size());
  if (input_names.size() == inputs_shape_info->input_tensors.size() && 
      input_names.size() > 0) {
    for(int i = 0; i < input_names.size(); i++) {
      feed.push_back({input_names[i], inputs_shape_info->input_tensors[i]});
    }
  }

  grappler::GrapplerItem item;
  inputs_shape_info->graph->ToGraphDef(&item.graph);
  item.fetch = inputs_shape_info->output_names;
  item.feed = feed;
  graph_properties_ = std::make_shared<grappler::GraphProperties>(item);
  graph_properties_->InferStatically(
                                false, /*assume_valid_feeds*/
                                false, /*aggressive_shape_inference*/
                                false, /*include_input_tensor_values*/
                                false, /*include_output_tensor_values*/
                                inputs_shape_info->input_tensors,
                                96
  );
  VLOG(1) << "####### Finish Init GrapplerItem! " << name_;
}

inline Status XlaAutoPadding::ValidateCluster(
    const std::vector<int>& base_dims,
    std::shared_ptr<InputsShapeInfo> inputs_shape_info) {
  
  CHECK(base_dims.size() == inputs_shape_info->all_dims.size());
  std::stringstream diff_flag;
  for (int i = 0; i < base_dims.size(); i++) {
    if ( base_dims[i] == inputs_shape_info->all_dims[i]) {
      // "0" meas dim equal
      diff_flag << "0";
    } else {
      // "1" meas dim not equal
      diff_flag << "1";
    }
  }

  const string signature = diff_flag.str();
  const auto& properties = inputs_shape_info->graph_properties; 
  const auto state = properties->GetXlaPaddingState(signature);

  if (state == PaddingState::VALID) {
    return Status::OK();
  } else if(state == PaddingState::UNKNOWN) {
    TF_RETURN_IF_ERROR(
        inputs_shape_info->graph_properties->InferStaticallyFastMode(
            inputs_shape_info->input_tensors,
            inputs_shape_info->inferred_shape_protos));
    //TODO Validate shape inference is correct
  }

  if (state == PaddingState::INVALID) {
    LOG(WARNING) << name_ << " XLA Padding Check failed";
    return Status(tensorflow::error::UNIMPLEMENTED, "Cannot compile");
  } else {
    LOG(INFO) << name_ << " XLA Padding Check Succ";
    properties->SetXlaPaddingState(signature, PaddingState::VALID);
    return Status::OK();
  }
}

inline void XlaAutoPadding::pad_to_power(int& val, 
    const std::vector<std::vector<int>>& shape_rules) {
  for(const auto shape_rule: shape_rules) {
    if (val >= shape_rule[0] && val <= shape_rule[1]) {
      int div = (val - 1) / shape_rule[2];
      val = (div + 1) * shape_rule[2];
      return;
    }
  }

  if (val <= 8) {
    val = 8;
  } else if (val <= 32) {
    val = 32;
  } else if (val >= 1500) {
    int div = (val - 1) / 200;
    val = (div + 1) * 200;
  } else {
    int div = (val - 1) / 64;
    val = (div + 1) * 64;
  }
}

inline int compare_all_dims(std::shared_ptr<InputsShapeInfo> inputs,
      std::shared_ptr<InputsShapeInfo> cached) {
  bool equal = true;
  for (int i = 0; i < inputs->all_dims.size(); i++) {
    if (inputs->all_dims[i] > cached->all_dims[i]) {
      return 1;
    } else if (inputs->all_dims[i] < cached->all_dims[i]) {
      equal = false;
    }
  }
  return equal ? 0 : -1;
}

inline std::shared_ptr<InputsShapeInfo> XlaAutoPadding::PaddingInputs(
    std::shared_ptr<InputsShapeInfo> inputs) { 
  // Padding inputs to cached shapes

  VLOG(1) << name_ << " PaddingInputs before " << inputs->DebugString();
  // find the most possible cache, the biggest cache is the most possible
  int possible_cache = cached_inputs_shapes_.size() - 1;
  for (int up = 0; up < cached_inputs_shapes_.size(); up++) {
    if (cached_inputs_shapes_[up]->size < inputs->size) continue;

    possible_cache = up;
    break;
  }

  int origin_size = inputs->size;
  bool all_dims_same = true;
  for (int i = 0; i < inputs->all_dims.size(); i++) {
    if (inputs->all_dims[i] == cached_inputs_shapes_[possible_cache]->all_dims[i]) continue;

    pad_to_power(inputs->all_dims[i], inputs->auto_padding_shape);
    all_dims_same = false;
  }
  if (all_dims_same) {
    cached_inputs_shapes_[possible_cache]->hit_times++;
    VLOG(1) << name_ << " has no dynamic input " << inputs->DebugString();
    return cached_inputs_shapes_[possible_cache];
  }

  inputs->refresh_shape();

  // make sure each dim is smaller than cached
  for (; possible_cache < cached_inputs_shapes_.size(); possible_cache++) {
    int compare = compare_all_dims(inputs, cached_inputs_shapes_[possible_cache]);
    if (compare == 0) {
      cached_inputs_shapes_[possible_cache]->hit_times++;
      VLOG(1) << name_ << " just hit " << cached_inputs_shapes_[possible_cache]->DebugString();
      return cached_inputs_shapes_[possible_cache];
    }
    else if (compare < 0) break;
  }

  if (possible_cache == cached_inputs_shapes_.size()) {
    VLOG(1) << name_ << " bigger than largest chache! input " << inputs->DebugString();
    return nullptr;
  }

  const auto& cached = cached_inputs_shapes_[possible_cache]; 
  VLOG(1) << name_ << " cached=" << cached->size << ", origin=" << origin_size << ", padded= "
          << inputs->size;
  if ((cached->size - origin_size) > (int)((inputs->size - origin_size) * 1.2)) {
    VLOG(1) << name_ << " padding too much!";
    VLOG(1) << name_ << " raw input:" << inputs->uuid() << " size=" << origin_size;
    VLOG(1) << name_ << " just hit :" << inputs->DebugString() << " size=" << inputs->size;
    VLOG(1) << name_ << " final to : " << cached->DebugString() << " size=" << cached->size;
    return nullptr;
  }

  cached->hit_times++;     
  inputs->copy_shapes(cached);
  VLOG(1) << name_ << " PaddingInputs after " << inputs->DebugString();
  return cached;
}

inline bool XlaAutoPadding::CheckConstEqual(
   const std::shared_ptr<InputsShapeInfo> cached_inputs,
   const std::shared_ptr<InputsShapeInfo> inputs) {
  /// For const inputs, the const value must be same
  /// Otherwise, it will cause xla cache miss
  for (int i = 0; i < inputs->const_indexs.size(); i++) {
    const Tensor& input = inputs->input_tensors[i];
    const Tensor& cached = cached_inputs->input_tensors[i];
    if (!input.SameAs(cached)) {
      LOG(WARNING) << "Input  tensor " << input.DebugString();
      LOG(WARNING) << "Cached tensor " << cached.DebugString();
      LOG(WARNING) << "[" << name_ << "]: Input " << i <<" Const arg not same, can not compile the cluster";
      return false;
    }
  }
  return true;
}

Status XlaAutoPadding::GetPaddingCachedShape(
    std::shared_ptr<InputsShapeInfo> inputs_shape_info) {

  mutex_lock lock(cached_input_shapes_mu_);
  VLOG(1) << "[" << name_ << "]: Input shape: " << inputs_shape_info->DebugString();
  if (cached_inputs_shapes_.size() == 0) {
    // keep first request shape as base, to compare which dim is dynamic
    // but dont compile
    inputs_shape_info->first_request = true;
    cached_inputs_shapes_.push_back(inputs_shape_info);
    return Status::OK();
  }
  
  inputs_shape_info->first_request = false;
  // If const args not same, means this cluster can not use xla
  if (!CheckConstEqual(cached_inputs_shapes_[0], inputs_shape_info)) {
    LOG(ERROR) << name_ << " CheckConstEqual fail";
    return Status(tensorflow::error::UNIMPLEMENTED, "Cannot compile");
  }
  
  auto cached = PaddingInputs(inputs_shape_info);
  if (cached != nullptr) {
    inputs_shape_info->graph_properties = cached->graph_properties;
    inputs_shape_info->is_cpu_device = cached->is_cpu_device;
    inputs_shape_info->out_compilation_result = cached->out_compilation_result;
    inputs_shape_info->out_executable = cached->out_executable;
    inputs_shape_info->hit_times = cached->hit_times;
    VLOG(1) << name_ << " hit cache " << inputs_shape_info->DebugString();
  } else {
    VLOG(1) << name_ << " Add New Xla Cache " << inputs_shape_info->DebugString();
    cached_inputs_shapes_.push_back(inputs_shape_info);

    SortCaches();
  }
  return Status::OK();
}

Status XlaAutoPadding::SortCaches() {
  std::sort(cached_inputs_shapes_.begin(), cached_inputs_shapes_.end(),
           [](const std::shared_ptr<InputsShapeInfo> shape1, const std::shared_ptr<InputsShapeInfo> shape2) {
             return shape1->size < shape2->size;
  });
  return Status::OK();
}

}
