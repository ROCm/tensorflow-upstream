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
class GemmLayernormGemmTest : public OpsTestBase {
 protected:
  void RunUnfusedGemmLayernormGemmTest(const std::vector<Eigen::half>& mat_A0,
                                       const std::vector<Eigen::half>& mat_B0,
                                       const std::vector<Eigen::half>& mat_C,
                                       const std::vector<Eigen::half>& mat_B1,
                                       const std::vector<Eigen::half>& Gamma,
                                       const std::vector<Eigen::half>& Beta,
                                       std::vector<Eigen::half>& mat_D, int K,
                                       int M, int N0, int N1, int head_num,
                                       float lrelu_alpha, bool do_layer_norm,
                                       bool do_leaky_relu) {
    std::vector<Eigen::half> tmp0;
    std::vector<Eigen::half> ln_res;
    float eps = 1e-12;
    int M1 = M * head_num;
    int K1 = N0 / head_num;
    int head_sz = N0 / head_num;
    float mul = 1.0f / sqrtf(head_sz * 1.f);
    // gemm 0
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N0; n++) {
        float psum = 0.f;
        for (int k = 0; k < K; k++) {
          float areg = float(mat_A0[m * K + k]);
          float breg = float(mat_B0[n * K + k]);
          psum += areg * breg;
        }
        psum += float(mat_C[n]);
        tmp0.push_back(Eigen::half(psum));
      }
    }
    // layernorm
    for (int m = 0; m < M1; m++) {
      float sum0 = 0.f;
      for (int k = 0; k < K1; k++) {
        sum0 += float(tmp0[m * K1 + k]);
      }
      float mean = sum0 / float(K1);
      float var = 0.f;
      for (int k = 0; k < K1; k++) {
        var +=
            (float(tmp0[m * K1 + k]) - mean) * (float(tmp0[m * K1 + k]) - mean);
      }
      var = 1.f / float(sqrt(var / float(K1) + eps));
      for (int k = 0; k < K1; k++) {
        ln_res.push_back(
            Eigen::half((var * float(Gamma[k])) * float(tmp0[m * K1 + k]) +
                        (float(Beta[k]) - (var * float(Gamma[k])) * mean)));
        // std::cout<<"ln_res"<<ln_res[k]<<std::endl;
      }
    }
    // gemm 1 + softmax
    // [M, N0] * [N1, N0] = [M, N1]
    // [M, N0] --> [M, head, N0/head]
    // [N1, N0] --> [N1, head, N0/head]
    // K1 = N0 / head_num;
    // output: [batch, head_num, seq]
    // output: [M, head_num, N1]
    for (int m = 0; m < M; m++) {
      for (int h = 0; h < head_num; h++) {
        // float sum = 0.f;
        for (int n = 0; n < N1; n++) {
          float psum = 0.f;
          for (int k = 0; k < K1; k++) {
            float areg = float(ln_res[m * N0 + h * N0 / head_num + k]);
            float breg = float(mat_B1[n * N0 + h * N0 / head_num + k]);
            psum += areg * breg;
          }
          float res = float(psum) * float(mul);
          mat_D[m * head_num * N1 + h * N1 + n] = Eigen::half(res);
        }
      }
    }
  }

  void RunGemmLayernormGemmTest(const std::vector<Eigen::half>& mat_A0,
                                const std::vector<Eigen::half>& mat_B0,
                                const std::vector<Eigen::half>& mat_C,
                                const std::vector<Eigen::half>& mat_B1,
                                const std::vector<Eigen::half>& Gamma,
                                const std::vector<Eigen::half>& Beta,
                                const std::vector<Eigen::half>& mat_D, int K,
                                int M, int N0, int N1, int head_num,
                                float lrelu_alpha, bool do_layer_norm,
                                bool do_leaky_relu) {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));

    TF_EXPECT_OK(NodeDefBuilder("gemm_layernorm_gemm", "GemmLayernormGemm")
                     .Input(FakeInput(DT_HALF))
                     .Input(FakeInput(DT_HALF))
                     .Input(FakeInput(DT_HALF))
                     .Input(FakeInput(DT_HALF))
                     .Input(FakeInput(DT_HALF))
                     .Input(FakeInput(DT_HALF))
                     .Attr("head_num", head_num)
                     .Attr("lrelu_alpha", lrelu_alpha)
                     .Attr("do_layer_norm", do_layer_norm)
                     .Attr("do_leaky_relu", do_leaky_relu)
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    int n = N0 / head_num;

    AddInputFromArray<Eigen::half>(TensorShape({M, 1, K}), mat_A0);  // 0
    AddInputFromArray<Eigen::half>(TensorShape({N0, K}), mat_B0);    // 1
    AddInputFromArray<Eigen::half>(TensorShape({N0}), mat_C);
    AddInputFromArray<Eigen::half>(TensorShape({n}), Beta);
    AddInputFromArray<Eigen::half>(TensorShape({n}), Gamma);
    AddInputFromArray<Eigen::half>(TensorShape({1, N1, N0}), mat_B1);

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), DT_HALF, TensorShape({M, head_num, N1}));
    test::FillValues<Eigen::half>(&expected, mat_D);

    test::ExpectTensorNear<Eigen::half>(expected, *GetOutput(0), 0.003);
  }
};

TEST_F(GemmLayernormGemmTest, Half) {
  srand(10);

  std::vector<Eigen::half> mat_A0;
  std::vector<Eigen::half> mat_B0;
  std::vector<Eigen::half> mat_C;
  std::vector<Eigen::half> mat_B1;
  std::vector<Eigen::half> Gamma;
  std::vector<Eigen::half> Beta;
  std::vector<Eigen::half> mat_D;
  int head_num = 8;
  int head_size = 32;
  int K = 32;
  int M = 4;
  int N0 = head_num * head_size;
  int N1 = 48;
  float lrelu_alpha = 0.0001;
  bool do_layer_norm = true;
  bool do_leaky_relu = false;

  // N1 = sequence_length
  // N0 = hidden_units = head_num * head_size
  // N2 = O

  // mat_A0  input0 [M, 1, K]
  // mat_B0  input1 [N0, K]
  // mat_C   input2 [N0]
  // Gamma   input3 [N0 / head_num]
  // Beta    input4 [N0 / head_num]
  // mat_B1  input5 [1, N1, N0]
  // Keymask input6 [1, N1]
  // mat_D   output [M, head_num, N1]

  for (int i = 0; i < M * K; ++i) {
    mat_A0.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < N0 * K; ++i) {
    mat_B0.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < N0; ++i) {
    mat_C.push_back(Eigen::half(0));
  }
  for (int i = 0; i < N0 / head_num; ++i) {
    Gamma.push_back(Eigen::half(1));
  }
  for (int i = 0; i < N0 / head_num; ++i) {
    Beta.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < N1 * N0; ++i) {
    mat_B1.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < M * N1 * head_num; ++i) {
    mat_D.push_back(Eigen::half(128));
  }

  RunUnfusedGemmLayernormGemmTest(mat_A0, mat_B0, mat_C, mat_B1, Gamma, Beta,
                                  mat_D, K, M, N0, N1, head_num, lrelu_alpha,
                                  do_layer_norm, do_leaky_relu);
  RunGemmLayernormGemmTest(mat_A0, mat_B0, mat_C, mat_B1, Gamma, Beta, mat_D, K,
                           M, N0, N1, head_num, lrelu_alpha, do_layer_norm,
                           do_leaky_relu);
}

}  // namespace
}  // namespace tensorflow
