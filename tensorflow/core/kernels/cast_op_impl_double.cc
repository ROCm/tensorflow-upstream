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

#include "tensorflow/core/kernels/cast_op_impl.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

CastFunctorType GetCpuCastFromDouble(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, CPUDevice, double);
  CAST_CASE(CPUDevice, double, float8_e5m2);
  CAST_CASE(CPUDevice, double, float8_e4m3fn);
  CAST_CASE(CPUDevice, double, float8_e5m2fnuz);
  CAST_CASE(CPUDevice, double, float8_e4m3fnuz);
  return nullptr;
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
CastFunctorType GetGpuCastFromDouble(DataType dst_dtype) {
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
  CAST_CASE(GPUDevice, double, bfloat16);
#else
  CURRY_TYPES3(CAST_CASE, GPUDevice, double);
#endif
  CAST_CASE(GPUDevice, double, float8_e5m2);
  CAST_CASE(GPUDevice, double, float8_e4m3fn);
  CAST_CASE(GPUDevice, double, float8_e5m2fnuz);
  CAST_CASE(GPUDevice, double, float8_e4m3fnuz);
  return nullptr;
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


}  // namespace tensorflow
