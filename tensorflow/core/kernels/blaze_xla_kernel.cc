/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/kernels/blaze_predictor.h"
#include "tensorflow/core/kernels/blaze_xla_predictor.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/compiler/jit/flags.h"

namespace tensorflow {

class BlazeXlaOp : public AsyncOpKernel {
 public:
  explicit BlazeXlaOp(OpKernelConstruction* context);
  ~BlazeXlaOp() {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override;

  Status ParseRunOptions(BlazeKernelOptions& run_options);
 private:
  Status ParseAttr();
  void InitPredictor(OpKernelConstruction* context);

  void Schedule(OpKernelContext* ctx, const DoneCallback& done, uint64 begin, bool is_first=true);
  void ComputeNormal(OpKernelContext* context, const DoneCallback& done);
  void ComputeNull(OpKernelContext* context, const DoneCallback& done);
  int GetBatchSizeUnsafe(OpKernelContext* context);

  DeviceType device_type_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::string blaze_option_path_;
  std::string graph_def_path_;
  std::string graph_def_str_;
  string device_string_;
  std::vector<DataType> input_types_;
  
  std::string device_;
  GraphDef graph_def_;
  std::unique_ptr<BlazePredictor> predictor_;
  BlazeKernelOptions blaze_run_options_;
  Env* env_;
  mutex tracing_mu_;
  mutex benchmark_mu_;
  mutex running_mu_;
  std::atomic<int> benchmark_counter_;
  int wait_ns_;

  tensorflow::thread::ThreadPool pool_;
  std::atomic<int> running_counter_;
  const int kBlazeRunningCount_;
  const int kScheduleFactor_ = 2;

  std::atomic<int> waiting_counter_;
  const int kMaxWaitingCount_;

  static std::atomic<int> total_waiting_counter_;
  static std::atomic<int> total_running_counter_;
};

std::atomic<int> BlazeXlaOp::total_waiting_counter_(0);
std::atomic<int> BlazeXlaOp::total_running_counter_(0);

int BlazeThreadsCount() {
  const int kDefaultDenseThreadsNum = 2;
  int64 dense_threads_num;
  ReadInt64FromEnvVar("BLAZE_THREADS_NUM", kDefaultDenseThreadsNum, &dense_threads_num);
  VLOG(0) << "blaze set thread pool size " << dense_threads_num;
  return (int)dense_threads_num;
}

int BlazeWatingCount() {
  const int kDefaultWaitingCount = 10;
  int64 dense_waiting_num;
  ReadInt64FromEnvVar("DENSE_MAX_WAITING_COUNT", kDefaultWaitingCount, &dense_waiting_num);
  VLOG(0) << "blaze max waiting count " << dense_waiting_num;
  return (int)dense_waiting_num;
}

void BlazeXlaOp::InitPredictor(OpKernelConstruction* context) {
  auto config = blaze_run_options_.mutable_config_proto();
  config->set_allow_soft_placement(true);
  config->mutable_gpu_options()->set_allow_growth(true);

  if (blaze_run_options_.use_single_threaded_executor()) {
    config->mutable_experimental()->set_executor_type("SINGLE_THREADED_EXECUTOR");
  }
  LOG(INFO) << "Blaze create with options " << blaze_run_options_.DebugString();
  if (blaze_run_options_.xla_compilation()) {
    auto jitLevel = OptimizerOptions::ON_1;
    {
      tensorflow::BuildXlaOpsPassFlags* flags =
          tensorflow::GetBuildXlaOpsPassFlags();
      flags->tf_xla_enable_lazy_compilation = false;
    }
    {
      tensorflow::MarkForCompilationPassFlags* flags =
          tensorflow::GetMarkForCompilationPassFlags();
      flags->tf_xla_cpu_global_jit = true;
      flags->tf_xla_min_cluster_size = 1;
    }
    config->mutable_graph_options()->mutable_optimizer_options()->set_global_jit_level(jitLevel);
    predictor_ = absl::make_unique<BlazeXlaPredictor>(input_names_, output_names_,
                                       graph_def_, device_, blaze_run_options_,
                                       device_string_, input_types_, context);
  } else {
    predictor_ = absl::make_unique<BlazePredictor>(input_names_, output_names_,
                                    graph_def_, device_, blaze_run_options_,
                                    device_string_, input_types_, context);
  }
}

BlazeXlaOp::BlazeXlaOp(OpKernelConstruction* context)
    : AsyncOpKernel(context), device_type_(context->device_type().type()), 
    pool_(Env::Default(), "blaze_kernel", BlazeThreadsCount() * 2), running_counter_(0),
    kBlazeRunningCount_(BlazeThreadsCount()), waiting_counter_(0),
    kMaxWaitingCount_(BlazeWatingCount()) {
  OP_REQUIRES_OK(context, context->GetAttr("input_names", &input_names_));
  OP_REQUIRES_OK(context, context->GetAttr("output_names", &output_names_));
  OP_REQUIRES_OK(context, context->GetAttr("graph_def", &graph_def_path_));
  OP_REQUIRES_OK(context, context->GetAttr("blaze_option_path", &blaze_option_path_));
  OP_REQUIRES_OK(context, context->GetAttr("InT", &input_types_));
  OP_REQUIRES_OK(context, ParseAttr());
  device_string_ = context->device_type().type_string();
  device_ = context->def().device();
  InitPredictor(context);
  OP_REQUIRES_OK(context, predictor_->InitSession());
  env_ = Env::Default();
  benchmark_counter_ = 0;
  wait_ns_ = blaze_run_options_.wait_ms() * 1000000;
}

Status BlazeXlaOp::ParseAttr() {
  if (ReadTextProto(Env::Default(), blaze_option_path_,
                     &blaze_run_options_).ok()) {
    VLOG(0) << "Parse blaze options from file succ";
  } else {
    if (!::tensorflow::protobuf::TextFormat::ParseFromString(
            blaze_option_path_, &blaze_run_options_)) {
      LOG(ERROR) << "Parse proto from " << blaze_option_path_ << " failed";
      return errors::Internal("parse proto from ", blaze_option_path_,  " failed");
    }
    LOG(INFO) << "Parse blaze options succ " << blaze_run_options_.DebugString();
  }

  if (!ReadTextProto(Env::Default(), graph_def_path_,
                     &graph_def_).ok()) {
    if (!ReadBinaryProto(Env::Default(), graph_def_path_,
                       &graph_def_).ok()) {
      return errors::Internal("parse proto from ", graph_def_path_,  " failed");
    }
  }

  graph_def_str_ = graph_def_.DebugString();
  
  return Status::OK();
}

void BlazeXlaOp::ComputeNormal(OpKernelContext* ctx, const DoneCallback& done) {
  auto begin = env_->NowNanos();
  Schedule(ctx, done, begin);
}

void BlazeXlaOp::ComputeNull(OpKernelContext* context, const DoneCallback& done) {
  auto batch_size = GetBatchSizeUnsafe(context);
  Tensor *output;
  context->allocate_output(0, {batch_size, 2}, &output);
  done();
}

void BlazeXlaOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  switch(blaze_run_options_.run_mode()) {
    case BlazeKernelOptions::DEFAULT: {
      ComputeNormal(ctx, std::move(done));
      break;
    }
    case BlazeKernelOptions::SKIP: {
      ComputeNull(ctx, std::move(done));
      break;
    }
    default: {
      ComputeNormal(ctx, std::move(done));
    }
  }
}

//unsafe func to infer batchsize, just for testing
int BlazeXlaOp::GetBatchSizeUnsafe(OpKernelContext* context) {
  for (int i = 0; i < context->num_inputs(); ++i) {
    const auto& shape = context->input(i).shape();
    if (shape.dims() != 0 && shape.dim_size(0) != 1) {
      return shape.dim_size(0);
    }
  }
  return 1;
}

void BlazeXlaOp::Schedule(OpKernelContext* ctx, const DoneCallback& done, uint64 begin, bool is_first) {
  auto schedule_func = [this, ctx, done, begin] {
    this->Schedule(ctx, done, begin, false);
  };
  if (running_counter_ >= kBlazeRunningCount_) {
    auto schedule_time = env_->NowNanos();
    if (wait_ns_ > 0) {
      OP_REQUIRES_ASYNC(ctx, schedule_time - begin <= wait_ns_,
          errors::Internal("blaze wait too long ", schedule_time - begin),
          done);
    } else {
      OP_REQUIRES_ASYNC(ctx, waiting_counter_ < kMaxWaitingCount_,
          errors::Internal("waiting pool is full ", waiting_counter_.load()),
          done);
    }
    if (is_first) { ++waiting_counter_; ++total_waiting_counter_; }
    pool_.Schedule(std::move(schedule_func));
  } else {
    if (!is_first) { --waiting_counter_; --total_waiting_counter_; }
    ++running_counter_;
    ++total_running_counter_;
    pool_.Schedule([this, ctx, done, begin] {
      auto schedule_time = env_->NowNanos();
      if (wait_ns_ > 0) {
        if (schedule_time - begin > wait_ns_) { --running_counter_; --total_running_counter_; }
        OP_REQUIRES_ASYNC(ctx, schedule_time - begin <= wait_ns_,
                          errors::DeadlineExceeded("blaze wait too long ", schedule_time - begin),
                          done);
      }
      Status status = predictor_->Compute(ctx);
      --running_counter_;
      --total_running_counter_;
      --total_waiting_counter_;
      OP_REQUIRES_ASYNC(ctx, status.ok(), status, done);
      done();
  });
} 
}

REGISTER_KERNEL_BUILDER(Name("BlazeXlaOp").Device(DEVICE_CPU), BlazeXlaOp);
REGISTER_KERNEL_BUILDER(Name("BlazeXlaOp").Device(DEVICE_GPU), BlazeXlaOp);

}  // namespace tensorflow
