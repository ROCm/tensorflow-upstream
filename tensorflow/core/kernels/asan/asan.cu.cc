#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
__global__ void
set1(int n, int *p)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i == n) {
      p[i] = 1;
    }
}

void RunMemset(Tensor input, int n1, int n2){
    set1<<<dim3(n1), dim3(n2), 0, 0>>>(input.flat<int>().size() + 1, input.flat<int>().data());
    // GPU_LAUNCH_KERNEL(set1, dim3(n1), dim3(n2), 0, nullptr, input.flat<int>().data());
}
}