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
class GemmRowSoftmaxGemmTest : public OpsTestBase {
 protected:
  // RunGemmRowSoftmaxGemmTest(mat_B0, mat_A0, mat_A1,  Keymask, mat_D, batch,
  // seq, head_num, new_head);
  void RunUnfusedGemmRowSoftmaxGemmTest(const std::vector<Eigen::half>& mat_B0,
                                        const std::vector<Eigen::half>& mat_A0,
                                        const std::vector<Eigen::half>& mat_A1,
                                        const std::vector<Eigen::half>& Keymask,
                                        std::vector<Eigen::half>& mat_D,
                                        int batch, int seq, int head_num,
                                        int new_head) {
    // input0: (batch, head_num, seq)  b0
    // input1: (new_head, head_num)    a0
    // input2: (head_num, new_head)    a1
    // input3: (batch, 1, seq)
    // output: (batch, head_num, seq)
    std::vector<Eigen::half> softmax_res;
    // float mul = 1.0f / sqrtf(head_sz * 1.f);
    for (int b = 0; b < batch; b++) {
      for (int m = 0; m < new_head; m++) {
        float sum = 0.f;
        std::vector<float> localExps;
        for (int n = 0; n < seq; n++) {
          float psum = 0.f;
          for (int k = 0; k < head_num; k++) {
            float areg = float(mat_A0[m * head_num + k]);
            float breg = float(mat_B0[b * head_num * seq + k * seq + n]);
            psum += areg * breg;
          }
          psum = psum > 0 ? psum : 0;
          // float res = float(psum) * float(mul);
          float fill = float(-4294967296);
          float res = Keymask[b * seq + n] ? psum : fill;
          float local = exp(float(res));
          localExps.push_back(local);
          sum += local;
        }
        for (int n = 0; n < seq; n++) {
          if (sum == 0.f)
            softmax_res.push_back(Eigen::half(1. / seq));
          else
            softmax_res.push_back(Eigen::half((localExps[n]) / sum));
        }
      }
    }
    // gemm 1
    // (head_num, new_head)  * (batch , new_head, seq)
    // output: (batch, head_num, seq)
    for (int b = 0; b < batch; b++) {
      for (int m = 0; m < head_num; m++) {
        for (int n = 0; n < seq; n++) {
          float psum = 0.f;
          for (int k = 0; k < new_head; k++) {
            float areg = float(mat_A1[m * new_head + k]);
            float breg = float(softmax_res[b * new_head * seq + k * seq + n]);
            // std::cout<<"test in cpu"<<breg<<std::endl;
            psum += areg * breg;
          }
          psum = psum > 0 ? psum : 0;
          mat_D[b * head_num * seq + m * seq + n] = Eigen::half(psum);
        }
      }
    }
  }

  // RunGemmRowSoftmaxGemmTest(mat_B0, mat_A0, mat_A1,  Keymask, mat_D, batch,
  // seq, head_num, new_head);

  void RunGemmRowSoftmaxGemmTest(const std::vector<Eigen::half>& mat_B0,
                                 const std::vector<Eigen::half>& mat_A0,
                                 const std::vector<Eigen::half>& mat_A1,
                                 const std::vector<Eigen::half>& Keymask,
                                 const std::vector<Eigen::half>& mat_D,
                                 int batch, int seq, int head_num,
                                 int new_head) {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));

    TF_EXPECT_OK(NodeDefBuilder("gemm_row_softmax_gemm", "GemmRowSoftmaxGemm")
                     .Input(FakeInput(DT_HALF))
                     .Input(FakeInput(DT_HALF))
                     .Input(FakeInput(DT_HALF))
                     .Input(FakeInput(DT_HALF))
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    // input0: (batch, head_num, seq)  b0
    // input1: (new_head, head_num)    a0
    // input2: (head_num, new_head)    a1
    // input3: (batch, 1, seq)
    // output: (batch, head_num, seq)

    AddInputFromArray<Eigen::half>(TensorShape({batch, head_num, seq}),
                                   mat_B0);  // 0
    AddInputFromArray<Eigen::half>(TensorShape({new_head, head_num}),
                                   mat_A0);  // 1
    AddInputFromArray<Eigen::half>(TensorShape({head_num, new_head}), mat_A1);
    AddInputFromArray<Eigen::half>(TensorShape({batch, 1, seq}), Keymask);

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), DT_HALF, TensorShape({batch, head_num, seq}));
    test::FillValues<Eigen::half>(&expected, mat_D);

    // test::ExpectTensorNear<Eigen::half>(expected, *GetOutput(0), 0.00001);
  }
};

TEST_F(GemmRowSoftmaxGemmTest, Half) {
  srand(10);

  std::vector<Eigen::half> mat_B0;
  std::vector<Eigen::half> mat_A0;
  std::vector<Eigen::half> mat_A1;
  std::vector<Eigen::half> Keymask;
  std::vector<Eigen::half> mat_D;
  int head_num = 8;
  int new_head = 64;
  int batch = 3;
  int seq = 200;

  // input0: (batch, head_num, seq)  b0
  // input1: (new_head, head_num)    a0
  // input2: (head_num, new_head)    a1
  // input3: (batch, 1, seq)
  // output: (batch, head_num, seq)

  for (int i = 0; i < batch * head_num * seq; ++i) {
    mat_B0.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < new_head * head_num; ++i) {
    mat_A0.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < new_head * head_num; ++i) {
    mat_A1.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < seq; ++i) {
    Keymask.push_back(Eigen::half(1));
  }
  for (int i = 0; i < seq; ++i) {
    Keymask.push_back(Eigen::half(0));
  }
  for (int i = 0; i < seq; ++i) {
    Keymask.push_back(Eigen::half(1));
  }
  for (int i = 0; i < batch * head_num * seq; ++i) {
    mat_D.push_back(Eigen::half(128));
  }

  RunUnfusedGemmRowSoftmaxGemmTest(mat_B0, mat_A0, mat_A1, Keymask, mat_D,
                                   batch, seq, head_num, new_head);
  RunGemmRowSoftmaxGemmTest(mat_B0, mat_A0, mat_A1, Keymask, mat_D, batch, seq,
                            head_num, new_head);
}

}  // namespace
}  // namespace tensorflow