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
class FusedGemmBiasAddTest : public OpsTestBase {
 protected:
  void RunUnfusedTest(const std::vector<Eigen::half>& mat_A,
                      const std::vector<Eigen::half>& mat_B,
                      const std::vector<Eigen::half>& mat_C,
                      std::vector<Eigen::half>& mat_D, int K, int M, int N) {
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

  void RunFusedGemmBiasAddTest(const std::vector<Eigen::half>& mat_A,
                               const std::vector<Eigen::half>& mat_B,
                               const std::vector<Eigen::half>& mat_C,
                               const std::vector<Eigen::half>& mat_D, int k,
                               int m, int n) {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));

    TF_EXPECT_OK(NodeDefBuilder("fused_gemm_bias_add", "FusedGemmBiasAdd")
                     .Input(FakeInput(DT_HALF))  // 0 q
                     .Input(FakeInput(DT_HALF))  // 1 k
                     .Input(FakeInput(DT_HALF))  // 2
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    AddInputFromArray<Eigen::half>(TensorShape({m, k}), mat_A);  // 0
    AddInputFromArray<Eigen::half>(TensorShape({n, k}), mat_B);  // 1
    AddInputFromArray<Eigen::half>(TensorShape({n}), mat_C);

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), DT_HALF, TensorShape({m, n}));
    test::FillValues<Eigen::half>(&expected, mat_D);

    test::ExpectTensorNear<Eigen::half>(expected, *GetOutput(0), 0.001);
  }
};

TEST_F(FusedGemmBiasAddTest, Half) {
  int k = 256;
  int m = 3;
  int n = 256;

  srand(10);
  std::vector<Eigen::half> mat_A;
  std::vector<Eigen::half> mat_B;
  std::vector<Eigen::half> mat_C;

  std::vector<Eigen::half> mat_D;

  for (int i = 0; i < m * k; ++i) {
    mat_A.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < n * k; ++i) {
    mat_B.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < n; ++i) {
    mat_C.push_back(Eigen::half(0));
  }
  for (int i = 0; i < m * n; ++i) {
    mat_D.push_back(Eigen::half(128));
  }

  RunUnfusedTest(mat_A, mat_B, mat_C, mat_D, k, m, n);
  RunFusedGemmBiasAddTest(mat_A, mat_B, mat_C, mat_D, k, m, n);
}

}  // namespace
}  // namespace tensorflow
