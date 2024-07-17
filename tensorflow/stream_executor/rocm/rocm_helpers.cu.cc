#include <hip/hip_runtime.h>
#include <limits>
namespace stream_executor {
namespace gpu {

__global__ void rocm_Broadcast_fp32Kernel(float* dst, int dst_stride,
                                          int batches, float* src, int size) {
  dst += blockIdx.y * 4 * dst_stride + blockIdx.z * dst_stride * batches;
  src += blockIdx.z * size;
  float* dst2 = dst + dst_stride;
  float* dst3 = dst + dst_stride * 2;
  float* dst4 = dst + dst_stride * 3;
  bool b2 = (blockIdx.y * 4 + 1 < batches);
  bool b3 = (blockIdx.y * 4 + 2 < batches);
  bool b4 = (blockIdx.y * 4 + 3 < batches);
  for (int i = threadIdx.x + blockIdx.x * 256; i < size;
       i += blockDim.x * gridDim.x) {
    dst[i] = src[i];
    if (b2) {
      dst2[i] = src[i];
    }
    if (b3) {
      dst3[i] = src[i];
    }
    if (b4) {
      dst4[i] = src[i];
    }
  }
}


/** 
 *  Receives a device side buffer of size 'size*src_batches' floats.
 *  Broadcasts each 'size' block into 'batches' copies in the destination buffer,
 *  at destination stride 'dst_stride':
 * 
 *     Input:    [A] [B] [C] (each at offset 'size' from each other)
 *     Output:   
 *       [A]  @ address 0
 *       [A]  @ address 'dst_stride'
 *       [A]  @ address 'dst_stride*2'
 *       [B]  @ address 'dst_stride*3'
*/
void rocm_Broadcast_fp32(void* stream, float* dst, int dst_stride, int batches,
                         int src_batches, float* src, int size) {
  int x_blocks = (size + 255) / 256;
  hipLaunchKernelGGL(rocm_Broadcast_fp32Kernel,
                     dim3(x_blocks, (batches + 3) / 4, src_batches),
                     min(256, (int)size), 0, (hipStream_t)stream, dst,
                     dst_stride, batches, src, size);
}

__global__ void rocm_Broadcast_rank3_fp32Kernel(
          float* pdst, int dst_stride, int batches, float* psrc, int size,
          int rank_3_dim, int rank_3_step_dst, int rank_3_step_src) {
  for(int j = 0; j<rank_3_dim; j++) {
    float* dst = pdst + blockIdx.y * 4 * dst_stride + blockIdx.z * dst_stride * batches + j * rank_3_step_dst;
    float* src = psrc + blockIdx.z * size + j*rank_3_step_src;
    float* dst2 = dst + dst_stride;
    float* dst3 = dst + dst_stride * 2;
    float* dst4 = dst + dst_stride * 3;
    bool b2 = (blockIdx.y * 4 + 1 < batches);
    bool b3 = (blockIdx.y * 4 + 2 < batches);
    bool b4 = (blockIdx.y * 4 + 3 < batches);
    for (int i = threadIdx.x + blockIdx.x * 256; i < size;
         i += blockDim.x * gridDim.x) {
      dst[i] = src[i];
      if (b2) {
        dst2[i] = src[i];
      }
      if (b3) {
        dst3[i] = src[i];
      }
      if (b4) {
        dst4[i] = src[i];
      }
    }
  }
}

void rocm_Broadcast_rank3_fp32(void* stream, float* dst, int dst_stride, int batches,
                         int src_batches, float* src, int size,
                         int rank_3_dim, int rank_3_step_dst, int rank_3_step_src) {
  int x_blocks = (size + 255) / 256;
  hipLaunchKernelGGL(rocm_Broadcast_rank3_fp32Kernel,
                     dim3(x_blocks, (batches + 3) / 4, src_batches),
                     min(256, (int)size), 0, (hipStream_t)stream, dst,
                     dst_stride, batches, src, size,
                     rank_3_dim, rank_3_step_dst, rank_3_step_src);
}


__global__ void rocm_Broadcast_general_kernel(void* pdst, const void** ppsrc, int size) 
{
  const float* src = reinterpret_cast<const float*>(ppsrc[blockIdx.x]);
  size >>= 2;
  float* dst = reinterpret_cast<float*>(pdst) + blockIdx.x*size;
  for (int i = threadIdx.x; i < size; i += 256)
    dst[i] = src[i];
}

void rocm_Broadcast_general(void* stream, 
                void* pdst, const void** ppsrc, int size, int batches) {
  int x_blocks = (size + 255) / 256;
  hipLaunchKernelGGL(rocm_Broadcast_general_kernel,
                     batches,
                     min(256, (int)size/4), 0, (hipStream_t)stream, 
                     pdst, ppsrc, size);
}


};  // namespace gpu
};  // namespace stream_executor
