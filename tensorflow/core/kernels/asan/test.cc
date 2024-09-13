#include <cstdlib>
// #include <hip/hip_runtime.h>
#include <vector>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/asan/asan.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class AsanTest : public OpsTestBase {};

TEST_F(AsanTest, Simple) {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));
    int m = 100;
    int n1 = 120;
    int n2 = 100;
    int *dp;
    Tensor input(allocator(), DT_INT32, TensorShape({m}));
    RunMemset(input, n1, n2);
    device_->Sync();

}
}

// TEST_F(AsanTest, Simple2) {
//     int m = 100;
//     int n1 = 10;
//     int n2 = 10;
//     int c = 99;
//     int *dp;
//     hipMalloc(&dp, m*sizeof(int));
//     hipLaunchKernelGGL(set1, dim3(n1), dim3(n2), 0, 0, dp);
//     int *hp = (int*)malloc(c * sizeof(int));
//     hipMemcpy(hp, dp, m*sizeof(int), hipMemcpyDeviceToHost);
//     hipDeviceSynchronize();
//     hipFree(dp);
//     free(hp);
// }
