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
#include "tensorflow/core/util/env_var.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/stream_executor.h"
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"

#if GOOGLE_CUDA
constexpr int kWarpSize = 32;
#else
constexpr int kWarpSize = 64;
#endif

#define MAKE_SHARED2(T, Name, N, M)                                     \
  __shared__ __align__(alignof(T)) char Name##_raw[N * M * sizeof(T)]; \
  typedef T(*Name##_Accessor)[M];                                      \
  auto Name = reinterpret_cast<Name##_Accessor>(Name##_raw)

#define MAKE_SHARED(T, Name, N)                                     \
  __shared__ __align__(alignof(T)) char Name##_raw[N * sizeof(T)]; \
  auto Name = reinterpret_cast<T*>(Name##_raw)


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
    val += __shfl_xor(val, offset);
  return val;
}

template <typename Scalar, int M_SIZE, int N_SIZE>
__forceinline__ __device__ Scalar blockReduceSum(Scalar val, int n) {
  MAKE_SHARED2(Scalar, s_data, N_SIZE, kWarpSize);
  __syncthreads();
  int lane = threadIdx.x % kWarpSize;
  int wid = threadIdx.x / kWarpSize;
  val = warpReduceSum<Scalar, kWarpSize>(val);
  if (lane == 0) {
    s_data[n][wid] = val;
  }
  __syncthreads();
  if (wid == 0) {
    val = (threadIdx.x <= blockDim.x / kWarpSize) ? s_data[n][lane] : 0;
    if (M_SIZE > kWarpSize*4) {
      val = warpReduceSum<Scalar, 8>(val);
    } else if (M_SIZE > kWarpSize*2) {
      val = warpReduceSum<Scalar, 4>(val);
    } else if (M_SIZE > kWarpSize) {
      val = warpReduceSum<Scalar, 2>(val);
    }
  }
  return val;
}

template <typename Scalar, int M_SIZE, int N_SIZE>
__forceinline__ __device__ Scalar blockReduceSum_Phase1(Scalar* s_data, Scalar val, int n) {
  int lane = threadIdx.x % kWarpSize;
  int wid = threadIdx.x / kWarpSize;
  val = warpReduceSum<Scalar, kWarpSize>(val);
  return val;
}

template <typename Scalar, int M_SIZE, int N_SIZE>
__forceinline__ __device__ Scalar blockReduceSum_Phase2(Scalar* s_data, int n) {
  int lane = threadIdx.x % kWarpSize;
  int wid = threadIdx.x / kWarpSize;
  Scalar val = 0;
  if (wid == 0) {
    val = (threadIdx.x <= blockDim.x / kWarpSize) ? s_data[n*kWarpSize+lane] : 0;
    if (M_SIZE > kWarpSize*4) {
      val = warpReduceSum<Scalar, 8>(val);
    } else if (M_SIZE > kWarpSize*2) {
      val = warpReduceSum<Scalar, 4>(val);
    } else if (M_SIZE > kWarpSize) {
      val = warpReduceSum<Scalar, 2>(val);
    }
  }
  return val;
}

template <bool use_indicator, typename Scalar, typename TIndex, int POW_NUM,
          int M_SIZE, int K_SIZE, int N_SIZE>
__global__ void ComputeCoActionIndicator(IMatmulParam<Scalar, TIndex> param, int global_flag) {
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
  MAKE_SHARED(Scalar,Bs,K_SIZE*N_SIZE);
  if (threadIdx.x < K_SIZE * N_SIZE) {
    Bs[threadIdx.x] = B[threadIdx.x];
  }
  __syncthreads();

  // step 2: pow + concat + matmul
  float C_local[N_SIZE] = {0.0f};

  // ~46 us (dropping: 240 us -> 194 us)
  if (threadIdx.x < M_SIZE) {
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
  }
  __syncthreads();

  constexpr int nMaxWarps = 3;
  //MAKE_SHARED2(float, s_data, N_SIZE, kWarpSize);
  __shared__ float s_data[N_SIZE*nMaxWarps];

  // step 3: tanh and wrap reduce.
  for (int n = 0; n < N_SIZE; n++)
    C_local[n] = tanhf(C_local[n]);

  int lane = threadIdx.x % kWarpSize;
  int wid = threadIdx.x / kWarpSize;

  int nWarps = blockDim.x / kWarpSize;

    // in the slow mode, executed as M=150, K=5, N=4, POW_NUM=2, grid size (1494000 150)
    // ( 9960 blocks, 498 blocks/CU )
    // Each thread executes 6*N_SIZE = 24 shfl 
    // ~109 us (dropping: 131 us)

    //float val = global_flag ? blockReduceSum_Phase1<float, M_SIZE, N_SIZE>(s_data, C_local[n], n) : 0.f;
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
      for (int n = 0; n < N_SIZE; n++)
        C_local[n] += __shfl_xor(C_local[n], offset);
  //for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
  //    for (int n = 0; n < N_SIZE; n++)
  //      C_local[n] += __shfl_xor(C_local[n], offset);

    // ~16 us (dropping: 224 us)
  if (lane == 0)
    for (int n = 0; n < N_SIZE; n++)
      s_data[n*nMaxWarps+wid] = C_local[n];

  __syncthreads();

#pragma unroll
  for (int n = 0; n < N_SIZE; n++) {
    // ~20 us (dropping: 240 us -> 220 us)
    /*
    C_local[n] = blockReduceSum_Phase2<float, M_SIZE, N_SIZE>(s_data, n);
    if (threadIdx.x == 0) {
      C[n] = Scalar(C_local[n]);
    }
    */
    if (wid == 0) {
      float val = (threadIdx.x < nWarps) ? s_data[n*nMaxWarps+lane] : 0;
      if (M_SIZE > kWarpSize*4) {
        val = warpReduceSum<float, 8>(val);
      } else if (M_SIZE > kWarpSize*2) {
        val = warpReduceSum<float, 4>(val);
      } else if (M_SIZE > kWarpSize) {
        val = warpReduceSum<float, 2>(val);
      }
      if (threadIdx.x == 0) {
        C[n] = Scalar(val);
      }
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
  dim3 block_dim((m+kWarpSize-1) & ~(kWarpSize-1));
  const auto& d = context->eigen_device<GPUDevice>();
  int shared_memory_size = 0; //k * n * sizeof(Scalar) + kWarpSize * n * sizeof(float);
  if (m == 50 && k == 5 && n == 4 && pow_num == 2) {
    TF_CHECK_OK(GpuLaunchKernel(
        ComputeCoActionIndicator<false, Scalar, int64, 2, 50, 5, 4>, grid_dim,
        block_dim, shared_memory_size, d.stream(), param, 0));
  } else if (m == 150 && k == 5 && n == 4 && pow_num == 2) {
    TF_CHECK_OK(GpuLaunchKernel(
        ComputeCoActionIndicator<false, Scalar, int64, 2, 150, 5, 4>, grid_dim,
        block_dim, shared_memory_size, d.stream(), param, 0));
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

  static bool do_blas_logging = false;
  static bool blas_logging_set = false;
  if(!blas_logging_set) {
      tensorflow::ReadBoolFromEnvVar("TF_ROCBLAS_TRACE", false, &do_blas_logging);
      blas_logging_set = true;
  }  
  IMatmulParam<Scalar, TIndex> param;
  param.A = const_cast<Scalar*>(in_a.template flat<Scalar>().data());
  param.B = const_cast<Scalar*>(in_b.template flat<Scalar>().data());
  param.C = out->template flat<Scalar>().data();
  param.indicators =
      const_cast<TIndex*>(indicator.template flat<TIndex>().data());
  param.batch_a = batch_a;
  param.parallel_num = paralle_num;
  dim3 grid_dim(batch_b, paralle_num, pow_num);
  dim3 block_dim((m+kWarpSize-1) & ~(kWarpSize-1));
  const auto& d = context->eigen_device<GPUDevice>();
  int shared_memory_size = k * n * sizeof(Scalar) + kWarpSize * n * sizeof(float);
  if (m == 50 && k == 5 && n == 4 && pow_num == 2) {
    TF_CHECK_OK(GpuLaunchKernel(
        ComputeCoActionIndicator<true, Scalar, TIndex, 2, 50, 5, 4>, grid_dim,
        block_dim, shared_memory_size, d.stream(), param, 0));
  } else if (m == 150 && k == 5 && n == 4 && pow_num == 2) {
    TF_CHECK_OK(GpuLaunchKernel(
        ComputeCoActionIndicator<true, Scalar, TIndex, 2, 150, 5, 4>, grid_dim,
        block_dim, shared_memory_size, d.stream(), param, 0));
  } else {
    return errors::InvalidArgument("Unsupported m, k, n, pow_num: ", m, k, n,
                                   pow_num);
  }
    if(do_blas_logging) {
      const Scalar* pa = param.A;
      //int nA = batch_a * param.parallel_num * m * k;
      const Scalar* pb = param.B;
      const Scalar* pc = param.C;
      int nC = pow_num * n * paralle_num * batch_b;
      printf("ComputeCoActionIndicator: %p %p %p   %f %f %f %f -> %f %f  .. %f %f\n", 
        pa, pb, pc,  
        float(pa[0]), float(pa[1]), float(pb[0]), float(pb[1]), float(pc[0]), float(pc[1]),
        float(pc[nC-2]), float(pc[nC-1])
        //checksum(pa, ctx.m*ctx.k*ctx.batch_count), checksum(pb, ctx.n*ctx.k*ctx.batch_count),
        //checksum(pc, ctx.m*ctx.n*ctx.batch_count)
        );
      fflush(stdout);
      if (!isfinite(float(pc[0])))
        exit(0);
    }

  return Status::OK();
}

template struct LaunchCoAction<GPUDevice, Eigen::half>;
template struct LaunchCoAction<GPUDevice, float>;
template struct LaunchCoActionIndicator<GPUDevice, Eigen::half, int32>;
template struct LaunchCoActionIndicator<GPUDevice, float, int32>;
template struct LaunchCoActionIndicator<GPUDevice, Eigen::half, int64>;
template struct LaunchCoActionIndicator<GPUDevice, float, int64>;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}  // namespace tensorflow
