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

class GemvSoftmaxGemvTest : public OpsTestBase {
 protected:
  template <typename T>
  void RunUnfusedGemvSoftmaxGemvTest(const std::vector<T>& mat_A,
                                     const std::vector<T>& mat_B0,
                                     const std::vector<T>& keymask,
                                     const std::vector<T>& mat_B1,
                                     std::vector<T>& mat_D, int head_sz,
                                     int seq, int B) {
    std::vector<float> new_vec_A;
    // for single_head efficient attention
    // input0 (B, 1, head_sz)
    // input1（B, seq, head_sz）
    // input2 (B, seq) // mask shape to check
    // input3 (B, seq, head_sz)
    // output (B, 1, head_sz)

    // output0: ( B, 1, seq)
    // output1: (B, 1, head_sz)
    float mul = 1.0f / sqrtf(head_sz * 1.0f);
    for (int b = 0; b < B; b++) {
      float sum = 0.f;
      std::vector<float> localExps;
      for (int n = 0; n < seq; n++) {
        float psum = 0.f;
        for (int k = 0; k < head_sz; k++) {
          float areg = float(mat_A[b * head_sz + k]);
          float breg = float(mat_B0[b * head_sz * seq + n * head_sz + k]);
          psum += areg * breg;
        }
        float res = float(psum) * float(mul);
        float fill = float(-4294967296);
        res = keymask[n + b * seq] ? res : fill;
        float local = exp(float(res));
        localExps.push_back(local);
        sum += local;
      }
      for (int n = 0; n < seq; n++) {
        float res = float(0);
        if (sum == 0.f)
          res = float(1. / seq);
        else
          res = float((localExps[n]) / sum);
        new_vec_A.push_back(res);
        // std::cout << "test in cpu: " << res << std::endl;
      }
    }
    for (int b = 0; b < B; b++) {
      for (int k = 0; k < head_sz; k++) {
        float psum = 0.f;
        for (int s = 0; s < seq; s++) {
          float areg = float(new_vec_A[b * seq + s]);
          float breg = float(mat_B1[b * head_sz * seq + s * head_sz + k]);
          // std::cout << "test in cpu: " << breg << std::endl;
          psum += areg * breg;
        }
        mat_D[b * head_sz + k] = T(psum);
      }
    }
  }

  template <typename T>
  void RunGemvSoftmaxGemvTest(const std::vector<T>& mat_A,
                              const std::vector<T>& mat_B0,
                              const std::vector<T>& keymask,
                              const std::vector<T>& mat_B1,
                              const std::vector<T>& mat_D, int head_sz, int seq,
                              int b) {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));
    DataType dt = DataTypeToEnum<T>::v();

    TF_EXPECT_OK(NodeDefBuilder("gemv_softmax_gemv", "GemvSoftmaxGemv")
                     .Input(FakeInput(dt))  // 0 q
                     .Input(FakeInput(dt))  // 1 k
                     .Input(FakeInput(dt))  // 2 keymask
                     .Input(FakeInput(dt))  // 3 v
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    AddInputFromArray<T>(TensorShape({b, 1, head_sz}), mat_A);     // 0
    AddInputFromArray<T>(TensorShape({b, seq, head_sz}), mat_B0);  // 1
    AddInputFromArray<T>(TensorShape({b, seq}), keymask);          // 1
    AddInputFromArray<T>(TensorShape({b, seq, head_sz}), mat_B1);  // 1

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), dt, TensorShape({b, 1, head_sz}));
    test::FillValues<T>(&expected, mat_D);

    test::ExpectTensorNear<T>(expected, *GetOutput(0), 0.001);
  }
};

TEST_F(GemvSoftmaxGemvTest, Simple) {
  int head_sz = 128;
  int seq = 32;
  int b = 1000;

  srand(10);
  std::vector<Eigen::half> mat_A;
  std::vector<Eigen::half> mat_B0;
  std::vector<Eigen::half> mat_B1;
  std::vector<Eigen::half> keymask;

  std::vector<Eigen::half> mat_D;

  for (int i = 0; i < b * head_sz; ++i) {
    mat_A.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < b * seq * head_sz; ++i) {
    mat_B0.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < b * seq * head_sz; ++i) {
    mat_B1.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < b * seq; ++i) {
    keymask.push_back(Eigen::half(1));
  }
  for (int i = 0; i < b * head_sz; ++i) {
    mat_D.push_back(Eigen::half(128));
  }

  RunUnfusedGemvSoftmaxGemvTest<Eigen::half>(mat_A, mat_B0, keymask, mat_B1,
                                             mat_D, head_sz, seq, b);
  RunGemvSoftmaxGemvTest<Eigen::half>(mat_A, mat_B0, keymask, mat_B1, mat_D,
                                      head_sz, seq, b);
}

TEST_F(GemvSoftmaxGemvTest, Half) {
  int head_sz = 128;
  int seq = 48;
  int b = 1600;

  srand(10);
  std::vector<Eigen::half> mat_A;
  std::vector<Eigen::half> mat_B0;
  std::vector<Eigen::half> mat_B1;
  std::vector<Eigen::half> keymask;

  std::vector<Eigen::half> mat_D;

  for (int i = 0; i < b * head_sz; ++i) {
    mat_A.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < b * seq * head_sz; ++i) {
    mat_B0.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < b * seq * head_sz; ++i) {
    mat_B1.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < b * seq; ++i) {
    keymask.push_back(Eigen::half(1));
  }
  for (int i = 0; i < b * head_sz; ++i) {
    mat_D.push_back(Eigen::half(128));
  }

  RunUnfusedGemvSoftmaxGemvTest<Eigen::half>(mat_A, mat_B0, keymask, mat_B1,
                                             mat_D, head_sz, seq, b);
  RunGemvSoftmaxGemvTest<Eigen::half>(mat_A, mat_B0, keymask, mat_B1, mat_D,
                                      head_sz, seq, b);
}

}  // namespace
}  // namespace tensorflow