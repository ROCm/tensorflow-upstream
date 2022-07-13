//
// Created by qiaoxj on 2020/9/7.
//

#ifndef TENSORFLOW_CO_ACTION_OP_H
#define TENSORFLOW_CO_ACTION_OP_H

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Scalar>
struct LaunchCoAction {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                  const Tensor& in_a, const Tensor& in_b, Tensor* out,
                  int64 batch_a, int64 batch_b, int64 paralle_num,
                  int64 pow_num);
};
template <typename Device, typename Scalar, typename TIndex>
struct LaunchCoActionIndicator {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                  const Tensor& in_a, const Tensor& in_b,
                  const Tensor& indicator, Tensor* out, int64 batch_a,
                  int64 batch_b, int64 paralle_num, int64 pow_num);
};

#if GOOGLE_CUDA
template <typename Scalar>
struct LaunchCoAction<GPUDevice, Scalar> {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                  const Tensor& in_a, const Tensor& in_b, Tensor* out,
                  int64 batch_a, int64 batch_b, int64 paralle_num,
                  int64 pow_num);
};
template <typename Scalar, typename TIndex>
struct LaunchCoActionIndicator<GPUDevice, Scalar, TIndex> {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                  const Tensor& in_a, const Tensor& in_b,
                  const Tensor& indicator, Tensor* out, int64 batch_a,
                  int64 batch_b, int64 paralle_num, int64 pow_num);
};

#endif

}  // namespace tensorflow

#endif  // TENSORFLOW_CO_ACTION_OP_H
