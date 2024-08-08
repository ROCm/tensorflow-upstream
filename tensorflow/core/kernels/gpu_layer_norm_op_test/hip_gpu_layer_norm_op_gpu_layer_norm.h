#ifndef TENSORFLOW_CORE_KERNELS_TILE_SPLIT_CONCAT_NORM3_H
#define TENSORFLOW_CORE_KERNELS_TILE_SPLIT_CONCAT_NORM3_H

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

// Forward declaration.
class OpKernelContext;

#ifdef TENSORFLOW_USE_ROCM
template <typename T >
void LaunchGpuLayerNorm (OpKernelContext* ctx,
                         const T *input,
                         const T *gamma,
                         const T *beta,
                         const int batch_size,
                         const int rows,
                         const int cols,
                         T *output);

#endif // GOOGLE_CUDA
} // namespace tensorflow

#endif
