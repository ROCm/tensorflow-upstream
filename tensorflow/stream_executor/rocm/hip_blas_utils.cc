/* Copyright 2023 The OpenXLA Authors.

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

#include "tensorflow/stream_executor/rocm/hip_blas_utils.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/compiler/xla/util.h"

namespace stream_executor {
namespace rocm {

xla::Status ToStatus(hipblasStatus_t status, const char* prefix) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return xla::InternalError("%s: HipblasLt error %d", 
        static_cast<int>(status));
  }
  return xla::Status::OK();
}

hipDataType AsHipblasDataType(blas::DataType type) {
  switch (type) {
    case blas::DataType::kHalf:
      return HIP_R_16F;
    case blas::DataType::kBFloat16:
      return HIP_R_16BF;
    case blas::DataType::kFloat:
      return HIP_R_32F;
    case blas::DataType::kDouble:
      return HIP_R_64F;
    case blas::DataType::kInt8:
      return HIP_R_8I;
    case blas::DataType::kInt32:
      return HIP_R_32I;
    case blas::DataType::kComplexFloat:
      return HIP_C_32F;
    case blas::DataType::kComplexDouble:
      return HIP_C_64F;
    default:
      LOG(FATAL) << "unknown data type";
  }
}

hipblasComputeType_t AsHipblasComputeType(blas::ComputationType type) {
  if (type == blas::ComputationType::kF32)
      //|| type == blas::ComputationType::kTF32AsF32)
    return HIPBLAS_COMPUTE_32F;
  else
    LOG(FATAL) << "unsupported hipblaslt computation type";
}

hipblasOperation_t AsHipblasOperation(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return HIPBLAS_OP_N;
    case blas::Transpose::kTranspose:
      return HIPBLAS_OP_T;
    case blas::Transpose::kConjugateTranspose:
      return HIPBLAS_OP_C;
  }
}

}  // namespace rocm
}  // namespace stream_executor
