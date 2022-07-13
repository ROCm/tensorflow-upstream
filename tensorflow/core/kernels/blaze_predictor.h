#ifndef TENSORFLOW_CORE_KERNELS_BLAZE_PREDICOTR_H_
#define TENSORFLOW_CORE_KERNELS_BLAZE_PREDICOTR_H_

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

//Base blaze predictor, for normal run/(mlir)
class BlazePredictor {
 public:
  BlazePredictor(OpKernelConstruction* ctx);
  BlazePredictor(const std::vector<std::string>& input_names,
                          const std::vector<std::string>& output_names,
                          const GraphDef& graph_def, const std::string& device,
                          const BlazeKernelOptions& options, const string& device_string,
                          const std::vector<DataType>& input_types,
                          OpKernelConstruction* ctx = nullptr) :
    input_names_(input_names), output_names_(output_names),
    graph_def_(graph_def), request_device_(device),
    blaze_run_options_(options), device_type_(device_string),
    input_types_(input_types), ctx_(ctx) {}

    virtual ~BlazePredictor() {}

  virtual Status Compute(OpKernelContext* ctx);
  virtual void ComputeNull(OpKernelContext* ctx) {}
  //session must created in constructor function, otherwise in compute function
  //it will cost lots of time the first time
  virtual Status InitSession();

  Session* GetSession() {
    return session_.get();
  }
 protected:
  // read from tensor proto
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  GraphDef graph_def_;
  std::string request_device_;
  BlazeKernelOptions blaze_run_options_;
  DeviceType device_type_;
  std::vector<DataType> input_types_;

  std::string blaze_option_path_;
  std::string graph_def_str_;
  std::string device_;
  OpKernelConstruction* ctx_;
  //runtime options
  std::unique_ptr<Session> session_;
  Session::CallableHandle handle_;

  //for cpu->gpu
  std::string blaze_real_deive_;
  bool same_device_;
  Device* blaze_device_;
  Allocator* blaze_allocator_;
  stream_executor::Stream* stream_;
  int vgpu_id_;

 private:
  Status ParseAttr(const std::string& device);
  virtual Status PrepareData() {
    return Status::OK();
  }

  virtual Status PrepareGraph(GraphDef& graph_def);
  virtual Status GenSessionOptions(SessionOptions& options);
  virtual Status MakeCallable();
  virtual Status Warmup();
  void SetDeviceInGraphDef(const std::string device_name, GraphDef* graph_def);

  Status SetDeviceInfo(OpKernelConstruction* ctx);

 private:
  const char* const kBlazeRealDevice = "_blaze_real_device";
  Status CopyTensorCPUToGPU(const std::vector<Tensor>& inputs,
                            std::vector<Tensor>* real_inputs,
                            OpKernelContext* ctx);
  Status CopyTensorGPUToCPU(const std::vector<Tensor>& gpu_tensors,
                            std::vector<Tensor>* cpu_tensors,
                            OpKernelContext* ctx);

 protected:
  stream_executor::Stream* GetStream() const;
  Status PrepareInputs(const std::vector<Tensor>& inputs,
      std::vector<Tensor>* real_inputs, OpKernelContext* ctx);

  Status PrepareOutputs(const std::vector<Tensor>& outputs,
      std::vector<Tensor>* real_outputs, OpKernelContext* ctx);

#define TYPECASE_0(dt, X, Y)                                    \
  case dt: {                                                  \
    return (void*)X->flat<EnumToDataType<dt>::Type>().data(); \
  }

void* GetTensorAddress(const Tensor* tensor_ptr) {
  auto tensor_type = tensor_ptr->dtype();
  switch (tensor_type) {
    TYPECASE_0(DT_FLOAT, tensor_ptr, dest_ptr);
    TYPECASE_0(DT_HALF, tensor_ptr, dest_ptr);
    TYPECASE_0(DT_INT8, tensor_ptr, dest_ptr);
    TYPECASE_0(DT_INT32, tensor_ptr, dest_ptr);
    TYPECASE_0(DT_INT64, tensor_ptr, dest_ptr);
    default: {
      LOG(ERROR) << "Unsupported Data type " << DataTypeString(tensor_type);
      return nullptr;
    }
  }
}

#define TYPECASE_1(dt, X, Y)                                    \
  case dt: {                                                  \
    return X->flat<EnumToDataType<dt>::Type>().size() * sizeof(EnumToDataType<dt>::Type); \
  }
uint64 GetTensorSize(const Tensor* tensor_ptr) {
  auto tensor_type = tensor_ptr->dtype();
  switch (tensor_type) {
    TYPECASE_1(DT_FLOAT, tensor_ptr, dest_ptr);
    TYPECASE_1(DT_HALF, tensor_ptr, dest_ptr);
    TYPECASE_1(DT_INT8, tensor_ptr, dest_ptr);
    TYPECASE_1(DT_INT32, tensor_ptr, dest_ptr);
    TYPECASE_1(DT_INT64, tensor_ptr, dest_ptr);
    default: {
      LOG(ERROR) << "Unsupported Data type " << DataTypeString(tensor_type);
      return 0;
    }
  }
}
};
}
#endif //end TENSORFLOW_CORE_KERNELS_BLAZE_PREDICOTR_H_
