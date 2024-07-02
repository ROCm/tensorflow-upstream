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
class GemmLayernormGemmSoftmaxGemmTest : public OpsTestBase {
 protected:
  void RunUnfusedGemmLayernormGemmSoftmaxGemmTest(
      const std::vector<Eigen::half>& mat_A0,
      const std::vector<Eigen::half>& mat_B0,
      const std::vector<Eigen::half>& mat_C,
      const std::vector<Eigen::half>& mat_B1,
      const std::vector<Eigen::half>& mat_B2,
      const std::vector<Eigen::half>& Gamma,
      const std::vector<Eigen::half>& Beta,
      const std::vector<Eigen::half>& Keymask, std::vector<Eigen::half>& mat_D,
      int K, int M, int N0, int N1, int N2, int B_kv, int head_num,
      float lrelu_alpha, bool do_layer_norm, bool do_leaky_relu,
      bool do_query_mask) {
    std::vector<Eigen::half> tmp0;
    std::vector<float> tmp2;
    std::vector<Eigen::half> ln_res;
    std::vector<Eigen::half> softmax_res;
    float eps = 1e-12;
    int M1 = M * head_num;
    // int N = N2 * head_num;
    int K1 = N0 / head_num;
    int head_sz = N2 / head_num;
    int head_sz_k = N0 / head_num;
    float mul = 1.0f / sqrtf(head_sz_k * 1.f);
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
        if (do_leaky_relu) psum = psum > 0 ? psum : psum * lrelu_alpha;
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
      }
    }
    // gemm 1 + softmax
    // [M, N0] * [B_kv, N1, N0] = [B_kv, M, N1]
    // [M, N0] --> [M, head, N0/head]
    // [B_kv, N1, N0] --> [B_kv, N1, head, N0/head]
    // K1 = N0 / head_num;
    for (int b = 0; b < B_kv; b++) {
      for (int m = 0; m < M; m++) {
        for (int h = 0; h < head_num; h++) {
          float sum = 0.f;
          std::vector<float> localExps;
          for (int n = 0; n < N1; n++) {
            float psum = 0.f;
            for (int k = 0; k < K1; k++) {
              //////
              float areg = float(ln_res[m * N0 + h * N0 / head_num + k]);
              float breg =
                  float(mat_B1[b * N0 * N1 + n * N0 + h * N0 / head_num + k]);
              psum += areg * breg;
            }
            float res = float(psum) * float(mul);
            float fill = float(-4294967296);
            res = Keymask[b * N1 + n] ? res : fill;
            float local = exp(float(res));
            localExps.push_back(local);
            sum += local;
          }
          for (int n = 0; n < N1; n++) {
            if (sum == 0.f)
              softmax_res.push_back(Eigen::half(1. / N1));
            else
              softmax_res.push_back(Eigen::half((localExps[n]) / sum));
          }
        }
      }
    }
    // gemm 2
    // [B_kv, M , N1] * [B_kv, N1, N2] = [B_kv, M , N2]
    // [B_kv, M* head_num, N1] --> [B_kv, M, head_num, N1]
    // [B_kv, N1, N2] --> [B_kv, N1, head_num, N2/head_num]
    for (int b = 0; b < B_kv; b++) {
      for (int m = 0; m < M; m++) {
        for (int h = 0; h < head_num; h++) {
          for (int n = 0; n < (N2 / head_num); n++) {
            float psum = 0.f;
            for (int k = 0; k < N1; k++) {
              float areg = float(softmax_res[b * M * head_num * N1 +
                                             m * head_num * N1 + h * N1 + k]);
              float breg =
                  float(mat_B2[b * N1 * N2 + k * N2 + h * N2 / head_num + n]);
              psum += areg * breg;
            }
            tmp2.push_back(psum);
            mat_D[b * M * N2 + m * N2 + h * N2 / head_num + n] =
                Eigen::half(psum);
          }
        }
      }
    }
  }

  void RunGemmLayernormGemmSoftmaxGemmTest(
      const std::vector<Eigen::half>& mat_A0,
      const std::vector<Eigen::half>& mat_B0,
      const std::vector<Eigen::half>& mat_C,
      const std::vector<Eigen::half>& mat_B1,
      const std::vector<Eigen::half>& mat_B2,
      const std::vector<Eigen::half>& Gamma,
      const std::vector<Eigen::half>& Beta,
      const std::vector<Eigen::half>& Keymask,
      const std::vector<Eigen::half>& mat_D, int K, int M, int N0, int N1,
      int long_seq, int N2, int B_kv, int head_num, float lrelu_alpha,
      bool do_layer_norm, bool do_leaky_relu, bool do_query_mask) {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));

    TF_EXPECT_OK(NodeDefBuilder("gemm_layernorm_gemm_softmax_gemm",
                                "GemmLayernormGemmSoftmaxGemm")
                     .Input(FakeInput(DT_HALF))
                     .Input(FakeInput(DT_HALF))
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
                     .Attr("do_query_mask", do_query_mask)
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    int n = N0 / head_num;

    AddInputFromArray<Eigen::half>(TensorShape({M, 1, K}), mat_A0);  // 0
    AddInputFromArray<Eigen::half>(TensorShape({N0, K}), mat_B0);    // 1
    AddInputFromArray<Eigen::half>(TensorShape({N0}), mat_C);
    AddInputFromArray<Eigen::half>(TensorShape({n}), Beta);
    AddInputFromArray<Eigen::half>(TensorShape({n}), Gamma);
    AddInputFromArray<Eigen::half>(TensorShape({B_kv, N1, N0}), mat_B1);
    AddInputFromArray<Eigen::half>(TensorShape({B_kv, long_seq}), Keymask);
    AddInputFromArray<Eigen::half>(TensorShape({B_kv, N1, N2}), mat_B2);

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), DT_HALF, TensorShape({B_kv * M, 1, N2}));
    test::FillValues<Eigen::half>(&expected, mat_D);

    test::ExpectTensorNear<Eigen::half>(expected, *GetOutput(0), 0.0005);
  }
};

TEST_F(GemmLayernormGemmSoftmaxGemmTest, Half) {
  srand(10);

  std::vector<Eigen::half> mat_A0;
  std::vector<Eigen::half> mat_B0;
  std::vector<Eigen::half> mat_C;
  std::vector<Eigen::half> mat_B1;
  std::vector<Eigen::half> mat_B2;
  std::vector<Eigen::half> Gamma;
  std::vector<Eigen::half> Beta;
  std::vector<Eigen::half> Keymask;
  std::vector<Eigen::half> mat_D;
  int head_num = 8;
  int head_size_k = 8;
  int head_size = 32;

  int K = 696;
  int M = 1;
  int N0 = head_num * head_size_k;
  int N1 = 200;
  int N2 = head_num * head_size;
  int B_kv = 4;
  float lrelu_alpha = 0.0001;
  bool do_layer_norm = true;
  bool do_leaky_relu = false;
  bool do_query_mask = false;

  // N1 = sequence_length
  // N0 = hidden_units = head_num * head_size
  // N2 = O

  // mat_A0  input0 [M, 1, K]
  // mat_B0  input1 [N0, K]
  // mat_C   input2 [N0]
  // Gamma   input3 [N0 / head_num]
  // Beta    input4 [N0 / head_num]
  // mat_B1  input5 [B_kv, N1, N0]
  // Keymask input6 [B_kv, N1]
  // mat_B2  input7 [B_kv, N1, N2]
  // mat_D   output [M, 1, N2]

  for (int i = 0; i < M * K; ++i) {
    mat_A0.push_back(Eigen::half(rand() / double(RAND_MAX) - 0.5));
  }
  for (int i = 0; i < N0 * K; ++i) {
    mat_B0.push_back(Eigen::half(rand() / double(RAND_MAX) - 0.5));
  }
  for (int i = 0; i < N0; ++i) {
    mat_C.push_back(Eigen::half(rand() / double(RAND_MAX) / 10));
  }
  for (int i = 0; i < N0 / head_num; ++i) {
    Gamma.push_back(Eigen::half(-1 * rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < N0 / head_num; ++i) {
    Beta.push_back(Eigen::half(rand() / double(RAND_MAX) / 10));
  }
  for (int i = 0; i < B_kv * N1 * N0; ++i) {
    mat_B1.push_back(Eigen::half(rand() / double(RAND_MAX) - 0.5));
  }
  for (int i = 0; i < B_kv * N1; ++i) {
      Keymask.push_back(Eigen::half((rand() / double(RAND_MAX) - 0.5) > 0));
  }
  for (int i = 0; i < B_kv * N2 * N1; ++i) {
    mat_B2.push_back(Eigen::half(rand() / double(RAND_MAX) - 0.3));
  }
  for (int i = 0; i < B_kv * M * N2; ++i) {
    mat_D.push_back(Eigen::half(128));
  }

  RunUnfusedGemmLayernormGemmSoftmaxGemmTest(
      mat_A0, mat_B0, mat_C, mat_B1, mat_B2, Gamma, Beta, Keymask, mat_D, K, M,
      N0, N1, N2, B_kv, head_num, lrelu_alpha, do_layer_norm, do_leaky_relu,
      do_query_mask);
  RunGemmLayernormGemmSoftmaxGemmTest(
      mat_A0, mat_B0, mat_C, mat_B1, mat_B2, Gamma, Beta, Keymask, mat_D, K, M,
      N0, N1, N1, N2, B_kv, head_num, lrelu_alpha, do_layer_norm, do_leaky_relu,
      do_query_mask);
}

}  // namespace
}  // namespace tensorflow
