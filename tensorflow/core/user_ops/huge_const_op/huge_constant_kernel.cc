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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"

#include <fstream>
#include "./huge_constant_kernel.h"
#include "./npy.h"

#ifdef TENSORFLOW_USE_SYCL
#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#endif  // TENSORFLOW_USE_SYCL


using DEVICE_CPU = Eigen::ThreadPoolDevice;
using DEVICE_GPU = Eigen::GpuDevice;
using Eigen::half;

namespace tensorflow {


HugeConstantOp::HugeConstantOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), initialized_(false) {

  DataType attr_dtype;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &attr_dtype));
  TensorShape attr_shape;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &attr_shape));
  string filename;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("path", &filename));

  std::ifstream stream(filename, std::ifstream::binary);
  OP_REQUIRES(ctx, stream, 
              errors::NotFound("Fail to open file: ", 
                               filename));
  
  std::string np_dtype_str;
  std::vector<unsigned long> np_shape;
  bool fortran_order;
  try {
    npy::LoadMetaFromNumpy(stream, np_dtype_str, np_shape, fortran_order);
  } catch (const std::exception& e) {
    OP_REQUIRES(ctx, false, 
                errors::Internal("Fail to parse npy file: ", 
                                      e.what()));
  }

  OP_REQUIRES(ctx, !fortran_order, 
              errors::Unimplemented("Fortran order NOT supported."));

  //check shape
  for (int i = 0; i < np_shape.size(); ++i) {
    OP_REQUIRES(ctx, np_shape[i] == attr_shape.dim_size(i), 
                errors::Internal("attr_shape and np_shape NOT match in dim ", i));
  }


  //check type
  std::string expect_dtype_str;
#define CHECK_NP_TYPE(SCALAR) \
  expect_dtype_str = npy::has_typestring<SCALAR>::dtype.str(); \
  OP_REQUIRES(ctx, np_dtype_str == expect_dtype_str, \
              errors::Internal("DataType mismatch: ", np_dtype_str, "!=", expect_dtype_str)); 

  switch (attr_dtype) {
    case DT_HALF:
      CHECK_NP_TYPE(half);
      break;
    case DT_FLOAT:
      CHECK_NP_TYPE(float);
      break;
    case DT_DOUBLE:
      CHECK_NP_TYPE(double);
      break;
    case DT_INT32:
      CHECK_NP_TYPE(int32);
      break;
    case DT_INT64:
      CHECK_NP_TYPE(int64);
      break;
    default:
      OP_REQUIRES(ctx, false, 
                  errors::Unimplemented("Unsupported DataType."));
      break;
  }

  //create tensor
  tensor_ = Tensor(attr_dtype, attr_shape);

  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "filename: " << filename
              << ", dtype: " << DataTypeString(attr_dtype)
              << ", shape: " << attr_shape.DebugString()
              << ", total_bytes" <<tensor_.TotalBytes();
  }
  
  //read stream to tensor data
  void* tensor_data = nullptr;
  switch (attr_dtype) {
    case DT_HALF:
      tensor_data = tensor_.flat<half>().data();
      break;
    case DT_FLOAT:
      tensor_data = tensor_.flat<float>().data();
      break;
    case DT_DOUBLE:
      tensor_data = tensor_.flat<double>().data();
      break;
    case DT_INT32:
      tensor_data = tensor_.flat<int32>().data();
      break;
    case DT_INT64:
      tensor_data = tensor_.flat<int64>().data();
      break;
    default:
      break;
  }

  stream.read(reinterpret_cast<char*>(tensor_data), tensor_.TotalBytes());
};

void HugeConstantOp::Compute(OpKernelContext* ctx) {
  if (!initialized_) {

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
    Device* gpu_device = dynamic_cast<Device*>(ctx->device());
    GPUDeviceContext* gpu_device_context = dynamic_cast<GPUDeviceContext*>(ctx->op_device_context());
    if (gpu_device!=nullptr && gpu_device_context!=nullptr) {
      //allocate tensor on GPU

      if (VLOG_IS_ON(1)) LOG(INFO) << "Copy CPU tensor to GPU";
      Tensor copy = Tensor(gpu_device->GetAllocator(AllocatorAttributes()), 
                           tensor_.dtype(), tensor_.shape(), AllocationAttributes());
      OP_REQUIRES(ctx, copy.IsInitialized(), errors::ResourceExhausted( 
                    "OOM when allocating tensor of type ", DataTypeString(tensor_.dtype()), "and shape ", tensor_.shape().DebugString()
                  ));

      //copy Host tensor data onto GPU tensor
      Notification n;
      Status status;
      gpu_device_context->CopyCPUTensorToDevice(
        &tensor_, gpu_device, &copy, 
        [&n, &status](const Status& s) {
          status = s;
          n.Notify();
        },
        true);
      n.WaitForNotification();
      OP_REQUIRES_OK(ctx, status);

      tensor_ = std::move(copy);
    } else {
      if (VLOG_IS_ON(1)) LOG(WARNING) << "Fail to Copy CPU tensor to GPU, ptrs: "
                                      << gpu_device <<" & "<<gpu_device_context;
    }
#endif

    initialized_ = true;
  }
  ctx->set_output(0, tensor_);
  if (TF_PREDICT_FALSE(ctx->track_allocations())) {
    ctx->record_persistent_memory_allocation(tensor_.AllocatedBytes());
  }
};

HugeConstantOp::~HugeConstantOp() {};

#define REGISTER_CPU_KERNEL(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("HugeConst").Device(DEVICE_CPU).TypeConstraint<TYPE>("dtype"), \
      HugeConstantOp);
REGISTER_CPU_KERNEL(Eigen::half);
REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);
REGISTER_CPU_KERNEL(int32);
REGISTER_CPU_KERNEL(int64);
#undef REGISTER_CPU_KERNEL

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
#define REGISTER_GPU_KERNEL(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("HugeConst").Device(DEVICE_GPU).TypeConstraint<TYPE>("dtype"), \
      HugeConstantOp);
REGISTER_GPU_KERNEL(Eigen::half);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
REGISTER_GPU_KERNEL(int32);
REGISTER_GPU_KERNEL(int64);
#undef REGISTER_GPU_KERNEL
#endif

} //namespaece tensorflow 
