#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/blaze_xla_predictor.h"
#include "tensorflow/core/util/env_var.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/gpu_utils.h"
using tensorflow::se::Event;
#endif

namespace tensorflow {
const char* const kOutputShape = "_output_shapes";
const char* const kShape = "shape";

InputNodeMap BlazeXlaPredictor::ToInputNodeMap() {
  InputNodeMap node_map;
  for (int i = 0; i < graph_def_.node_size(); ++i) {
    auto& node = graph_def_.node(i);
    node_map_[node.name()] = node;
    for (int j = 0; j < node.input_size(); ++j) {
      auto& input = node.input(j);
      auto iter = node_map.find(input);
      if (iter == node_map.end()) {
        std::vector<NodeDef> node_list;
        node_list.push_back(node);
        node_map[input] = std::move(node_list);
      } else {
        iter->second.push_back(node);
      }
    }
  }
  return node_map;
}

Status BlazeXlaPredictor::FindBlackPaddingInputs() {
  std::set<std::string> no_warmup;
  for (const auto& black_input : blaze_run_options_.no_warmup_inputs()) {
    no_warmup.insert(black_input);
  }
  skip_padding_.resize(input_names_.size());
  auto node_map = ToInputNodeMap();

  for (int i = 0; i < input_names_.size(); ++i) {
    skip_padding_[i] = false;
    if (no_warmup.find(input_names_[i]) != no_warmup.end()) {
      skip_padding_[i] = true;
    }
  }
  return Status::OK();
}

Status BlazeXlaPredictor::InitXlaWarmup() {
  if (blaze_run_options_.warmup_batchsize_size() == 0) {
    return errors::Internal("xla not setting warmup batchsize");
  }

  std::vector<int> warm;
  warm.reserve(blaze_run_options_.warmup_batchsize_size());
  for (int i = 0; i < blaze_run_options_.warmup_batchsize_size(); ++i) {
    warm.push_back(blaze_run_options_.warmup_batchsize(i));
  }

  std::sort(warm.begin(), warm.end());
  for (auto val : warm) {
    if (val <= 0) {
      LOG(ERROR) << "exo warmup batchsize " << val << " invalid";
      return errors::Internal("warmuup batchsize ", val, " invalid");
    }
  }

  batch_sizes_ = std::move(warm);
  return Status::OK();
}

Status BlazeXlaPredictor::Warmup() {
  warmuped_ = false;
  return Status::OK();
}

Status BlazeXlaPredictor::Warmup(OpKernelContext* ctx) {
  if (warmuped_) {
    return Status::OK();
  }
  int num_inputs = ctx->num_inputs();
  std::vector<Tensor> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(ctx->input(i));
  }

  int batchsize = InferBatchSize(inputs);
  if (batchsize == -1) {
    return errors::Internal("Cannot infer inputs' batchsize");
  }
  auto max_bs = batch_sizes_[batch_sizes_.size() - 1];
  std::vector<Tensor> padded_inputs(num_inputs);
  Status status;
  if (same_device_) {
    status = PadToStatic(inputs, &padded_inputs,
        batchsize, max_bs, ctx);
  } else {
    status = PadToStaticCPUToGPU(inputs, &padded_inputs,
        batchsize, max_bs, ctx);
  }
  if (!status.ok()) {
    return status;
  }

  for (auto bs : batch_sizes_) {
    VLOG(0) << "begin warmup " << bs;
    auto start_us = Env::Default()->NowMicros();
    int pad_to_batchsize = bs;

    VLOG(1) << "batchsize = " << batchsize
        << ", pad_to_batchsize = " << pad_to_batchsize;

    // Pad inputs
    std::vector<Tensor> sliced_inputs;
    sliced_inputs.reserve(num_inputs);
    for (int i = 0; i < padded_inputs.size(); ++i) {
      const TensorShape& shape = inputs[i].shape();
      int64 first_dim = shape.dim_size(0);
      first_dim = (first_dim == 1) ? 1 : bs;
      if (skip_padding_[i] || (bs > 1 && first_dim == 1)) {
        sliced_inputs.push_back(padded_inputs[i]);
      } else {
        sliced_inputs.push_back(padded_inputs[i].Slice(0, bs));
      }
    }
    // Call SessionRun
    std::vector<Tensor> padded_outputs;
    status = session_->RunCallable(
         handle_, sliced_inputs, &padded_outputs, nullptr);
    if (!status.ok()) {
      return status;
    }
    auto end_us = Env::Default()->NowMicros();
    VLOG(0) << "batch " <<  pad_to_batchsize << " has warmuped; cost us: " << (end_us - start_us);
  }
  return Status::OK();
}

Status BlazeXlaPredictor::CheckShape(const TensorShapeProto& shape) {
  for (int i = 0; i < shape.dim_size(); ++i) {
    if (shape.dim(i).size() <= 0) {
      return errors::Internal("shape size invalid ",  shape.DebugString());
    }
  }
  return Status::OK();
}

Status BlazeXlaPredictor::PrepareData() {
  TF_RETURN_IF_ERROR(FindBlackPaddingInputs());
  TF_RETURN_IF_ERROR(InitXlaWarmup());

  return Status::OK();
}

int BlazeXlaPredictor::InferBatchSize(const std::vector<Tensor>& tensors) {
  int batchsize = -1;
  for (size_t i = 0; i < tensors.size(); ++i) {
    VLOG(1) << "Shape of input " << i << ": "
            << tensors[i].shape().DebugString();
    if (skip_padding_[i]) continue;
    int dims = tensors[i].shape().dims();
    if (dims == 0) continue;
    int64 first_dim = tensors[i].shape().dim_size(0);
    if (batchsize == -1 || (batchsize == 1 && first_dim != 1)) {
      batchsize = first_dim;
    }
    if (batchsize != 1 && first_dim != 1 && first_dim != batchsize) {
      batchsize = -1;
      VLOG(1) << "Cannot infer batchsize: tensors have different dim_size(0).";
      break;
    }
  }
  return batchsize;
}

Status BlazeXlaPredictor::PadToStaticCPUToGPU(const std::vector<Tensor>& inputs,
                                      std::vector<Tensor>* padded_inputs,
                                      int batchsize, int pad_to_batchsize,
                                      OpKernelContext* ctx) {
  for (int i = 0; i < inputs.size(); ++i) {
    VLOG(1) << "Shape of input " << i << ": "
            << inputs[i].shape().DebugString();
    TensorShape pad_to_shape;
    const TensorShape& shape = inputs[i].shape();
    pad_to_shape = shape;
    int64 first_dim = shape.dim_size(0);
    if (!skip_padding_[i]) {
      first_dim = (first_dim == 1) ? 1 : pad_to_batchsize;
    }
    pad_to_shape.set_dim(0, first_dim);
    Tensor padded_tensor(blaze_allocator_, inputs[i].dtype(), pad_to_shape);
    (*padded_inputs)[i] = padded_tensor;
    const uint8* input_ptr = (uint8*)GetTensorAddress(&inputs[i]);
    uint8* padded_ptr = (uint8*)GetTensorAddress(&(*padded_inputs)[i]);
    uint64 input_size = GetTensorSize(&inputs[i]);
    uint64 padded_size = GetTensorSize(&(*padded_inputs)[i]);
    if (input_ptr == nullptr || padded_ptr == nullptr ||
        input_size == 0 || padded_size == 0) {
      return errors::Internal(
          "Error when getting input address or size");
    }
#if GOOGLE_CUDA
      auto padded_dev_ptr = AsDeviceMemory(padded_ptr, padded_size);
      if (DataTypeIsInteger(inputs[i].dtype())) {
        bool copy_status =
            GetStream()->ThenMemZero(&padded_dev_ptr, padded_size).ok();
        if (!copy_status) {
          return errors::Internal("MemZero failed.");
        }
      }
      bool copy_status =
          GetStream()->ThenMemcpy(&padded_dev_ptr, input_ptr, input_size).ok();
      if (!copy_status) {
        return errors::Internal("MemcpyH2D for padding inputs failed.");
      }
#endif

    VLOG(1) << "Shape of padded_input " << i << ": "
            << (*padded_inputs)[i].shape().DebugString();
  }
  return Status::OK();
}

Status BlazeXlaPredictor::PadToStatic(const std::vector<Tensor>& inputs,
                                      std::vector<Tensor>* padded_inputs,
                                      int batchsize, int pad_to_batchsize,
                                      OpKernelContext* ctx) {
  for (int i = 0; i < inputs.size(); ++i) {
    VLOG(1) << "Shape of input " << i << ": "
            << inputs[i].shape().DebugString();
    AllocatorAttributes alloc_attrs;
    alloc_attrs.set_on_host(ctx->input_memory_type(i) == HOST_MEMORY);
    TensorShape pad_to_shape;
    const TensorShape& shape = inputs[i].shape();
    pad_to_shape = shape;
    int64 first_dim = shape.dim_size(0);
    first_dim = (first_dim == 1)? 1 : pad_to_batchsize;
    if (first_dim == 1 || skip_padding_[i]) {
      (*padded_inputs)[i] = inputs[i];
      VLOG(1) << "Shape of padded_input " << i << ": "
              << (*padded_inputs)[i].shape().DebugString();
      continue;
    }
    pad_to_shape.set_dim(0, first_dim);
    Status allocate_status =
        ctx->allocate_temp(inputs[i].dtype(),
                           pad_to_shape,
                           &(*padded_inputs)[i], alloc_attrs);
    if (!allocate_status.ok()) {
      return allocate_status;
    }
    const uint8* input_ptr = (uint8*)GetTensorAddress(&inputs[i]);
    uint8* padded_ptr = (uint8*)GetTensorAddress(&(*padded_inputs)[i]);
    uint64 input_size = GetTensorSize(&inputs[i]);
    uint64 padded_size = GetTensorSize(&(*padded_inputs)[i]);
    if (input_ptr == nullptr || padded_ptr == nullptr ||
        input_size == 0 || padded_size == 0) {
      return errors::Internal(
          "Error when getting input address or size");
    }
    if (device_type_ == DEVICE_GPU && ctx->input_memory_type(i) == DEVICE_MEMORY) {
#if GOOGLE_CUDA
      auto* stream = ctx->op_device_context()->stream();
      auto input_dev_ptr = AsDeviceMemory(input_ptr, input_size);
      auto padded_dev_ptr = AsDeviceMemory(padded_ptr, padded_size);
      if (DataTypeIsInteger(inputs[i].dtype())) {
        bool copy_status =
            stream->ThenMemZero(&padded_dev_ptr, padded_size).ok();
        if (!copy_status) {
          return errors::Internal("MemZero failed.");
        }
      }
      bool copy_status =
          stream->ThenMemcpyD2D(&padded_dev_ptr, input_dev_ptr, input_size).ok();
      if (!copy_status) {
        return errors::Internal("MemcpyD2D for padding inputs failed.");
      }
#endif
    } else {
      std::memset(padded_ptr, 0, padded_size);
      std::memcpy(padded_ptr, input_ptr, input_size);
    }

    VLOG(1) << "Shape of padded_input " << i << ": "
            << (*padded_inputs)[i].shape().DebugString();
  }
  return Status::OK();
}

const int kUnPadding = 1;
Status BlazeXlaPredictor::SliceToDynamic(const std::vector<Tensor>& padded_outputs,
                                         int batchsize, int pad_to_batchsize,
                                         std::vector<Tensor>& outputs, OpKernelContext* ctx) {
  for (int i = 0; i < padded_outputs.size(); ++i) {
    VLOG(1) << "Shape of padded_output " << i << ": "
            << padded_outputs[i].shape().DebugString();
    TensorShape slice_to_shape = padded_outputs[i].shape();
    if (slice_to_shape.dim_size(0) == kUnPadding) {
      outputs.push_back(padded_outputs[i]);
      continue;
    }
    if (slice_to_shape.dim_size(0) != pad_to_batchsize) {
      return errors::Internal(
          "Shape error, cannot slice output: padded_output shape = " +
          slice_to_shape.DebugString() +
          ", pad_to_batchsize = " +
          std::to_string(pad_to_batchsize));
    }
    outputs.push_back(padded_outputs[i].Slice(0, batchsize));
  }
  return Status::OK();
}

Status BlazeXlaPredictor::SliceToDynamicCPU(const std::vector<Tensor>& padded_outputs,
                                         int batchsize, int pad_to_batchsize,
                                         std::vector<Tensor>& outputs, OpKernelContext* ctx) {
#if GOOGLE_CUDA
  for (int i = 0; i < padded_outputs.size(); ++i) {
    VLOG(1) << "Shape of padded_output " << i << ": "
            << padded_outputs[i].shape().DebugString();
    TensorShape slice_to_shape = padded_outputs[i].shape();
    if (slice_to_shape.dim_size(0) != pad_to_batchsize) {
      return errors::Internal(
          "Shape error, cannot slice output: padded_output shape = " +
          slice_to_shape.DebugString() +
          ", pad_to_batchsize = " +
          std::to_string(pad_to_batchsize));
    }
    const auto& tmp_tensor = padded_outputs[i].Slice(0, batchsize);
    auto device_context = blaze_device_->tensorflow_gpu_device_info()->default_context;
    uint8* tmp_ptr = (uint8*)GetTensorAddress(&tmp_tensor);
    uint64 tmp_size = GetTensorSize(&tmp_tensor);
    auto tmp_dev_ptr = AsDeviceMemory(tmp_ptr, tmp_size);
    Tensor tensor;
    AllocatorAttributes alloc_attrs;
    alloc_attrs.set_on_host(true);
    alloc_attrs.set_gpu_compatible(true);
    TF_RETURN_IF_ERROR(ctx->allocate_temp(tmp_tensor.dtype(),
          tmp_tensor.shape(), &tensor, alloc_attrs));
    uint8* host_add = (uint8*)GetTensorAddress(&tensor);
    auto stream = GetStream();
    stream->ThenMemcpy(host_add, tmp_dev_ptr, tmp_size);
    auto event = std::make_shared<Event>(stream->parent());
    if (!event->Init()) {
      LOG(ERROR) << "event init failed!";
      return errors::Internal("SliceToDynamic GPU2CPU failed event init");
    }
    stream->ThenRecordEvent(event.get());
    stream->ThenSynchronizeEvent(event.get());

    outputs.push_back(tensor);
  }
#endif
  return Status::OK();
}

Status BlazeXlaPredictor::Compute(OpKernelContext* ctx) {
  // Infer inputs' batchsize

  if (TF_PREDICT_FALSE(!warmuped_)) {
    if (warmuping_) {
      return errors::Internal("Blaze kernel warmuping");
    }
    VLOG(0) << "Begin warmup";
    mutex_lock l(warmup_mu_);
    warmuping_ = true;
    auto st = Warmup(ctx);
    if (!st.ok()) {
      warmuping_ = false;
      return st;
    }
    warmuped_ = true;
    warmuping_ = false;
  }

  int num_inputs = ctx->num_inputs();
  std::vector<Tensor> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(ctx->input(i));
  }
  int batchsize = InferBatchSize(inputs);
  if (batchsize == -1) {
    return errors::Internal("Cannot infer inputs' batchsize");
  }

  int pad_to_batchsize = batchsize;
  bool found_bs = false;
  for (int n : batch_sizes_) {
    if (n >= batchsize) {
      pad_to_batchsize = n;
      found_bs = true;
      break;
    }
  }

  if (TF_PREDICT_FALSE(!found_bs)) {
    mutex_lock l(batch_size_mu_);
    pad_to_batchsize = AddNewBatchSize(batchsize);
  }

  VLOG(1) << "batchsize = " << batchsize
          << ", pad_to_batchsize = " << pad_to_batchsize;

  if (pad_to_batchsize != batchsize) {
    // Pad inputs
    std::vector<Tensor> padded_inputs(num_inputs);
    Status status;
    if (same_device_) {
      status = PadToStatic(inputs, &padded_inputs,
        batchsize, pad_to_batchsize, ctx);
    } else {
      status = PadToStaticCPUToGPU(inputs, &padded_inputs,
        batchsize, pad_to_batchsize, ctx);
    }
    if (!status.ok()) {
      return status;
    }

    // Call SessionRun
    std::vector<Tensor> padded_outputs;
    TF_RETURN_IF_ERROR(session_->RunCallable(
            handle_, padded_inputs, &padded_outputs, nullptr));

    // Unpad outputs
    std::vector<Tensor> outputs;
    outputs.reserve(padded_outputs.size());
    if (same_device_) {
      status = SliceToDynamic(padded_outputs, batchsize, pad_to_batchsize, outputs, ctx);
    } else {
      status = SliceToDynamicCPU(padded_outputs, batchsize, pad_to_batchsize, outputs, ctx);
    }
    if (!status.ok()) {
      return status;
    }
    for (int i = 0; i < outputs.size(); ++i) {
      ctx->set_output(i, outputs[i]);
    }
  } else {
    VLOG(1) << "Skip padding: input bathsize = " << batchsize
            << ", input pad_to_batchsize = " << pad_to_batchsize;
    std::vector<Tensor> outputs;
    std::vector<Tensor> real_inputs(inputs.size());

    TF_RETURN_IF_ERROR(PrepareInputs(inputs, &real_inputs, ctx));
    TF_RETURN_IF_ERROR(session_->RunCallable(
            handle_, real_inputs, &outputs, nullptr));

    std::vector<Tensor> real_outputs(outputs.size());
    TF_RETURN_IF_ERROR(PrepareOutputs(outputs, &real_outputs, ctx));
    for (int i = 0; i < real_outputs.size(); ++i) {
      ctx->set_output(i, real_outputs[i]);
    }
  }
  return Status::OK();
}

int BlazeXlaPredictor::AddNewBatchSize(int padded_size) {
  int last_size = batch_sizes_[batch_sizes_.size() - 1];
  if (padded_size <= last_size) {
    return last_size;
  }

  int add_size = last_size;
  while(add_size < padded_size) {
    add_size += last_size;
  }

  batch_sizes_.push_back(add_size);
  return add_size;
}
}
