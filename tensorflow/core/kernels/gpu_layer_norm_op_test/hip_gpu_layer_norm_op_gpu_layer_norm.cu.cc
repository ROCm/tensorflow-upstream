#include "hip/hip_runtime.h"
#if TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/platform/rocm.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/kernels/hip_gpu_layer_norm_op_gpu_layer_norm.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
#define FINAL_MASK 0xffffffff
template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  #pragma unroll
  for(int mask = 16; mask > 0; mask >>= 1)
    val += GpuShuffleXorSync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T warpReduceSum_t4(T val)
{
  #pragma unroll
  for(int mask = 2; mask > 0; mask >>= 1)
    val += GpuShuffleXorSync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T warpReduceSum_t8(T val)
{
  #pragma unroll
  for(int mask = 4; mask > 0; mask >>= 1)
    val += GpuShuffleXorSync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T warpReduceSum_t16(T val)
{
  #pragma unroll
  for(int mask = 8; mask > 0; mask >>= 1)
    val += GpuShuffleXorSync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;
  __syncthreads();
 
  int num_warps = blockDim.x >> 5; 
  if (blockDim.x % 32 != 0)
    num_warps += 1;

  val = (threadIdx.x < (num_warps )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}

template <typename T>
__global__ void fusedGpuLayerNormOpKernelWarp(const T *pInMat, 
                               const T *gamma,
                               const T *beta,
                               const float y,
                               const int batch_size,
                               const int rows,
                               const int cols,
                               T *pOutMat) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ float s_mean, s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float tmp = 0.f;
  if( threadIdx.x < cols){
      tmp = float(pInMat[i])  ;
  }
 
  if (cols ==4)
      mean = warpReduceSum_t4<float>(tmp);
  else if (cols ==8)
      mean = warpReduceSum_t8<float>(tmp);
  else if (cols ==16)
      mean = warpReduceSum_t16<float>(tmp);
  else if (cols ==32)
      mean = warpReduceSum<float>(tmp);
 
  if (threadIdx.x == 0)
      s_mean = mean / cols;
  __syncthreads();

  if( threadIdx.x < cols){
      variance = float(tmp - s_mean) * float(tmp - s_mean);
  }

  if (cols ==4)
      variance = warpReduceSum_t4<float>(variance);
  else if (cols ==8)
      variance = warpReduceSum_t8<float>(variance);
  else if (cols ==16)
      variance = warpReduceSum_t16<float>(variance);
  else if (cols ==32)
      variance = warpReduceSum<float>(variance);

  if (threadIdx.x == 0)
      s_variance = variance / cols ;
  __syncthreads();

  float sum = rsqrtf(s_variance  + y);

  sum = sum * float(gamma[i % cols]);


  mean = float(beta[i % cols]) - sum * s_mean;
  sum = sum * tmp + mean;
   
  pOutMat[i] = T(sum);
}

template <typename T>
__global__ void fusedGpuLayerNormOpKernelBlock(const T *pInMat, 
                               const T *gamma,
                               const T *beta,
                               const float y,
                               const int batch_size,
                               const int rows,
                               const int cols,
                               T *pOutMat) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  // Normalize
  __shared__ float s_mean, s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float tmp = 0.f;
  if( threadIdx.x < cols){
      tmp = float(pInMat[i])  ;
  }
 
  mean = blockReduceSum<float>(tmp);
 
  if (threadIdx.x == 0)
      s_mean = mean / cols;
  __syncthreads();

  if( threadIdx.x < cols){
      variance = float(tmp - s_mean) * float(tmp - s_mean);
  }

  variance = blockReduceSum<float>(variance);

  if (threadIdx.x == 0)
      s_variance = variance / cols ;
  __syncthreads();

  float sum = rsqrtf(s_variance  + y);

  sum = sum * float(gamma[i % cols]);

  mean = float(beta[i % cols]) - sum * s_mean;

  sum = sum * tmp + mean;
   
  pOutMat[i] = T(sum);
}

template <typename T>
__global__ void fusedGpuLayerNormOpKernelBlock2(const T *pInMat, 
                                                const T *gamma,
                                                const T *beta,
                                                const float y,
                                                const int batch_size,
                                                const int rows,
                                                const int cols,
                                                T *pOutMat) {

  // int i = threadIdx.x + blockIdx.x * blockDim.x;

  // Normalize
  __shared__ float s_mean, s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float p_sum = 0.f;
  float p_var = 0.f;

  int loop = (cols + 1023) / 1024  ; 
  float tmp[10];
  for (int l = 0 ; l < loop ; l ++){
    tmp[l] = 0.f;
    int i = threadIdx.x + l * 1024;
    int in_offset = i + blockIdx.y * cols + blockIdx.z * rows * cols; 
    if( i < cols){
      tmp[l] = float(pInMat[in_offset]);
    }
  }

  for (int l = 0 ; l < loop ; l ++){
    float tmp1 = 0.f;
    int i = threadIdx.x + l * 1024;
    if( i < cols){
      tmp1 = tmp[l];
    }
    // __syncthreads();
    mean = blockReduceSum<float>(tmp1);
    __syncthreads();
    p_sum += mean;
  }
 
  if (threadIdx.x == 0)
    s_mean = p_sum / cols;
  __syncthreads();

  for (int l = 0 ; l < loop ; l ++){
    float tmp1 = 0.f;
    float variance = 0.f;
    int i = threadIdx.x + l * 1024;
    if( i < cols){
      tmp1 = tmp[l];
      variance = float(tmp1 - s_mean) * float(tmp1 - s_mean);
    }
    variance = blockReduceSum<float>(variance);
    __syncthreads();
    p_var += variance;
  }

  if (threadIdx.x == 0)
    s_variance = p_var / cols ;
  __syncthreads();

  for (int l = 0 ; l < loop ; l ++){
    int i = threadIdx.x + l * 1024;
    int in_offset = i + blockIdx.y * cols + blockIdx.z * rows * cols; 
    float sum = rsqrtf(s_variance  + y);
    if( i < cols){
      sum = sum * float(gamma[i]);
      mean = float(beta[i]) - sum * s_mean;
      sum = sum * tmp[l] + mean;
      pOutMat[in_offset] = T(sum);
    }
  }
}



template <typename T>
void LaunchGpuLayerNorm(OpKernelContext *ctx,
                                const T *input,
                                const T *gamma,
                                const T *beta,
                                const int batch_size,
                                const int rows,
                                const int cols,
                                T *output) {
  int32 size = batch_size * rows * cols;
  const GPUDevice d = ctx->eigen_device<GPUDevice>();
  int32 thread_per_block, block_count;   
  float y = 1e-12;
  
  if (cols < 32){
      thread_per_block = cols ;   
      block_count = size / thread_per_block;
      fusedGpuLayerNormOpKernelWarp<T><<<block_count, thread_per_block, 0, d.stream()>>>(input,
                                                                                 gamma,
                                                                                 beta,
                                                                                 y,
                                                                                 batch_size,
                                                                                 rows,
                                                                                 cols,
                                                                                 output);
  } else if (cols <= 1024) {
      thread_per_block = cols ;   
      block_count = size / thread_per_block;
      fusedGpuLayerNormOpKernelBlock<T><<<block_count, thread_per_block, 0, d.stream()>>>(input,
                                                                                 gamma,
                                                                                 beta,
                                                                                 y,
                                                                                 batch_size,
                                                                                 rows,
                                                                                 cols,
                                                                                 output);
  } else {
      dim3 block_dims(1024, 1, 1);
      dim3 grid_dims( 1 , rows, batch_size);

      fusedGpuLayerNormOpKernelBlock2<T><<<grid_dims, block_dims, 0, d.stream()>>>(input,
                                                                              gamma,
                                                                              beta,
                                                                              y,
                                                                              batch_size,
                                                                              rows,
                                                                              cols,
                                                                              output);
  }


  hipError_t status = hipGetLastError();
  if (status != hipSuccess) {
    LOG(ERROR) << "GpuLayerNorm kernel launch failed with error code " << static_cast<int>(status) << " rows " << rows << ", cols " << cols << "block_count " << block_count << " ,thread_per_block " << thread_per_block;
  }
}



template void LaunchGpuLayerNorm<float>(OpKernelContext *,
                                        const float *,
                                        const float *,
                                        const float *,
                                        const int,
                                        const int,
                                        const int,
                                        float *);
template void LaunchGpuLayerNorm<Eigen::half>(OpKernelContext *,
                                              const Eigen::half *,
                                              const Eigen::half *,
                                              const Eigen::half *,
                                              const int,
                                              const int,
                                              const int,
                                              Eigen::half *) ;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
