#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

#include "tensorflow/core/kernels/hip_gpu_layer_norm_op_gpu_layer_norm.h"

#ifdef TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/stream_executor/rocm/rocm_stream.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
class GpuLayerNorm: public OpKernel {
 public:
  explicit GpuLayerNorm(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }
  
  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& gamma = ctx->input(1);
    const Tensor& beta = ctx->input(2);
    int rank = ctx->input(0).dims();
    int batch_size, rows, cols; 
    Tensor* out = nullptr;
    if (rank == 2) {
      batch_size = input.dim_size(0);
      rows = 1;
      cols = input.dim_size(1);
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch_size, cols}, &out));
    } else if(rank == 3) {
      batch_size = input.dim_size(0);
      rows = input.dim_size(1);
      cols = input.dim_size(2);
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch_size, rows, cols}, &out));
    } else{
      return;
    }

    if (out->NumElements() == 0)
      return;

    auto* input_ptr = input.template flat<T>().data();
    auto* gamma_ptr = gamma.template flat<T>().data();
    auto* beta_ptr = beta.template flat<T>().data();
    auto* output_ptr = out->template flat<T>().data();
    LaunchGpuLayerNorm(ctx, input_ptr, gamma_ptr, beta_ptr, batch_size, rows, cols, output_ptr);
  }
  
};

REGISTER_KERNEL_BUILDER(
    Name("GpuLayerNorm").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    GpuLayerNorm<Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("GpuLayerNorm").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    GpuLayerNorm<float>);
} // namespace tensorflow
#endif  // GOOGLE_CUDA
