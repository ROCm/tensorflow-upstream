//
// Created by qiaoxj on 2020-09-07.
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/co_action_op.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

template <typename T>
struct HalfAsFloat {
  typedef T type;
};

template <>
struct HalfAsFloat<Eigen::half> {
  typedef float type;
};

template <typename Scalar, typename TIndex>
struct IMatmulParam {
  Scalar* A;
  Scalar* B;
  Scalar* C;
  TIndex* indicators;
  int batch_a;
  int parallel_num;
};

template <typename Scalar, int SUM_NUM>
__forceinline__ __device__ Scalar warpReduceSum(Scalar val) {
  for (int offset = SUM_NUM / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

template <typename Scalar, int M_SIZE, int N_SIZE>
__forceinline__ __device__ Scalar blockReduceSum(Scalar val, int n) {
  __shared__ static Scalar s_data[N_SIZE][32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  val = warpReduceSum<Scalar, 32>(val);
  if (lane == 0) {
    s_data[n][wid] = val;
  }
  __syncthreads();
  if (wid == 0) {
    val = (threadIdx.x <= blockDim.x / warpSize) ? s_data[n][lane] : 0;
    if (M_SIZE > 128) {
      val = warpReduceSum<Scalar, 8>(val);
    } else if (M_SIZE > 64) {
      val = warpReduceSum<Scalar, 4>(val);
    } else if (M_SIZE > 32) {
      val = warpReduceSum<Scalar, 2>(val);
    }
  }
  return val;
}

template <bool use_indicator, typename Scalar, typename TIndex, int POW_NUM,
          int M_SIZE, int K_SIZE, int N_SIZE>
__global__ void ComputeCoActionIndicator(IMatmulParam<Scalar, TIndex> param) {
  int ind = 0;
  if (use_indicator) {
    ind = (int)param.indicators[blockIdx.x];
    if (ind < 0 || ind >= param.batch_a) {
      ind = 0;
    }
  }
  Scalar* A = param.A +
              (ind * param.parallel_num + blockIdx.y) * M_SIZE * K_SIZE +
              threadIdx.x * K_SIZE;
  Scalar* B = param.B +
              (blockIdx.x * param.parallel_num + blockIdx.y) * K_SIZE * N_SIZE;
  Scalar* C =
      param.C +
      ((blockIdx.x * param.parallel_num + blockIdx.y) * POW_NUM + blockIdx.z) *
          N_SIZE;

  // step 1: load matrix b to shared memory
  __shared__ Scalar Bs[K_SIZE * N_SIZE];
  if (threadIdx.x < K_SIZE * N_SIZE) {
    Bs[threadIdx.x] = B[threadIdx.x];
  }
  __syncthreads();

  // step 2: pow + concat + matmul
  float C_local[N_SIZE] = {0.0f};
#pragma unroll
  for (int k = 0; k < K_SIZE; k++) {
    float a_val = float(A[k]);
#pragma unroll
    for (int n = 0; n < N_SIZE; n++) {
      if (blockIdx.z == 0) {
        C_local[n] += a_val * float(Bs[k * N_SIZE + n]);
      } else {
        C_local[n] += a_val * a_val * float(Bs[k * N_SIZE + n]);
      }
    }
  }
  // step 3: tanh and wrap reduce.
#pragma unroll
  for (int n = 0; n < N_SIZE; n++) {
    C_local[n] = tanhf(C_local[n]);
    C_local[n] = blockReduceSum<float, M_SIZE, N_SIZE>(C_local[n], n);
    if (threadIdx.x == 0) {
      C[n] = Scalar(C_local[n]);
    }
  }
}

template <typename Scalar>
Status LaunchCoAction<GPUDevice, Scalar>::operator()(
    OpKernelContext* context, int64 m, int64 n, int64 k, const Tensor& in_a,
    const Tensor& in_b, Tensor* out, int64 batch_a, int64 batch_b,
    int64 paralle_num, int64 pow_num) {
  IMatmulParam<Scalar, int64> param;
  param.A = const_cast<Scalar*>(in_a.template flat<Scalar>().data());
  param.B = const_cast<Scalar*>(in_b.template flat<Scalar>().data());
  param.C = out->template flat<Scalar>().data();
  param.indicators = nullptr;
  param.batch_a = batch_a;
  param.parallel_num = paralle_num;
  dim3 grid_dim(batch_b, paralle_num, pow_num);
  dim3 block_dim(m);
  const auto& d = context->eigen_device<GPUDevice>();
  int shared_memory_size = k * n * sizeof(Scalar) + 32 * n * sizeof(float);
  if (m == 50 && k == 5 && n == 4 && pow_num == 2) {
    TF_CHECK_OK(GpuLaunchKernel(
        ComputeCoActionIndicator<false, Scalar, int64, 2, 50, 5, 4>, grid_dim,
        block_dim, shared_memory_size, d.stream(), param));
  } else if (m == 150 && k == 5 && n == 4 && pow_num == 2) {
    TF_CHECK_OK(GpuLaunchKernel(
        ComputeCoActionIndicator<false, Scalar, int64, 2, 150, 5, 4>, grid_dim,
        block_dim, shared_memory_size, d.stream(), param));
  } else {
    return errors::InvalidArgument("Unsupported m, k, n, pow_num: ", m, k, n,
                                   pow_num);
  }
  return Status::OK();
}

template <typename Scalar, typename TIndex>
Status LaunchCoActionIndicator<GPUDevice, Scalar, TIndex>::operator()(
    OpKernelContext* context, int64 m, int64 n, int64 k, const Tensor& in_a,
    const Tensor& in_b, const Tensor& indicator, Tensor* out, int64 batch_a,
    int64 batch_b, int64 paralle_num, int64 pow_num) {
  IMatmulParam<Scalar, TIndex> param;
  param.A = const_cast<Scalar*>(in_a.template flat<Scalar>().data());
  param.B = const_cast<Scalar*>(in_b.template flat<Scalar>().data());
  param.C = out->template flat<Scalar>().data();
  param.indicators =
      const_cast<TIndex*>(indicator.template flat<TIndex>().data());
  param.batch_a = batch_a;
  param.parallel_num = paralle_num;
  dim3 grid_dim(batch_b, paralle_num, pow_num);
  dim3 block_dim(m);
  const auto& d = context->eigen_device<GPUDevice>();
  int shared_memory_size = k * n * sizeof(Scalar) + 32 * n * sizeof(float);
  if (m == 50 && k == 5 && n == 4 && pow_num == 2) {
    TF_CHECK_OK(GpuLaunchKernel(
        ComputeCoActionIndicator<true, Scalar, TIndex, 2, 50, 5, 4>, grid_dim,
        block_dim, shared_memory_size, d.stream(), param));
  } else if (m == 150 && k == 5 && n == 4 && pow_num == 2) {
    TF_CHECK_OK(GpuLaunchKernel(
        ComputeCoActionIndicator<true, Scalar, TIndex, 2, 150, 5, 4>, grid_dim,
        block_dim, shared_memory_size, d.stream(), param));
  } else {
    return errors::InvalidArgument("Unsupported m, k, n, pow_num: ", m, k, n,
                                   pow_num);
  }
  return Status::OK();
}

template struct LaunchCoAction<GPUDevice, Eigen::half>;
template struct LaunchCoAction<GPUDevice, float>;
template struct LaunchCoActionIndicator<GPUDevice, Eigen::half, int32>;
template struct LaunchCoActionIndicator<GPUDevice, float, int32>;
template struct LaunchCoActionIndicator<GPUDevice, Eigen::half, int64>;
template struct LaunchCoActionIndicator<GPUDevice, float, int64>;
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
