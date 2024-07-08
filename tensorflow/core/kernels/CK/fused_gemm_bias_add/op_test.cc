#include <functional>
#include <iostream>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {
class GemmBiasAddTest : public OpsTestBase {
 protected:
  void RunUnfusedTest(const std::vector<Eigen::half>& mat_A,
                      const std::vector<Eigen::half>& mat_B,
                      const std::vector<Eigen::half>& mat_C,
                      std::vector<Eigen::half>& mat_D, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
      std::vector<float> tmp;
      for (int n = 0; n < N; n++) {
        float psum = 0.f;
        for (int k = 0; k < K; k++) {
          float areg = float(mat_A[m * K + k]);
          float breg = float(mat_B[n * K + k]);
          psum += areg * breg;
        }
        psum += float(mat_C[n]);
        mat_D[m * N + n] = Eigen::half(psum);
      }
    }
  }

  void RunCKTest(const std::vector<Eigen::half>& mat_A,
                 const std::vector<Eigen::half>& mat_B,
                 const std::vector<Eigen::half>& mat_C,
                 std::vector<Eigen::half>& mat_D, int M, int N, int K) {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));

    TF_EXPECT_OK(NodeDefBuilder("fused_gemm_bias_add", "FusedGemmBiasAdd")
                     .Input(FakeInput(DT_HALF))
                     .Input(FakeInput(DT_HALF))
                     .Input(FakeInput(DT_HALF))
                     //  .Input(FakeInput(DT_HALF))
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    AddInputFromArray<Eigen::half>(TensorShape({M, K}), mat_A);  // 0
    AddInputFromArray<Eigen::half>(TensorShape({N, K}), mat_B);  // 1
    AddInputFromArray<Eigen::half>(TensorShape({M, N}), mat_C);  // 1

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), DT_HALF, TensorShape({M, N}));
    test::FillValues<Eigen::half>(&expected, mat_D);
    Tensor actual = *GetOutput(0);
    test::ExpectTensorNear<Eigen::half>(expected, actual, 0.0005);
  }
};

TEST_F(GemmBiasAddTest, Half) {
  srand(10);

  std::vector<Eigen::half> mat_A;
  std::vector<Eigen::half> mat_B;
  std::vector<Eigen::half> mat_C;
  std::vector<Eigen::half> mat_D;

  int K = 256;
  int M = 3;
  int N = 256;

  for (int i = 0; i < M * K; ++i) {
    mat_A.push_back(Eigen::half(rand() / double(RAND_MAX) - 0.5));
  }
  for (int i = 0; i < N * K; ++i) {
    mat_B.push_back(Eigen::half(rand() / double(RAND_MAX) - 0.5));
  }
  for (int i = 0; i < M * N; ++i) {
    mat_C.push_back(Eigen::half(rand() / double(RAND_MAX) - 0.5));
  }
  for (int i = 0; i < M * N; ++i) {
    mat_D.push_back(Eigen::half(128));
  }

  RunUnfusedTest(mat_A, mat_B, mat_C, mat_D, M, N, K);
  RunCKTest(mat_A, mat_B, mat_C, mat_D, M, N, K);
}

}  // namespace
}  // namespace tensorflow
