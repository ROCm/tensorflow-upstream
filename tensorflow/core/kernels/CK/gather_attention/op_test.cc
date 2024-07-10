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
class GatherAttentionTest : public OpsTestBase {
 protected:
  template <typename T>
  void RunUnfusedGatherAttentionTest(const std::vector<T>& mat_A,
                                     const std::vector<T>& mat_B0,
                                     const std::vector<T>& keymask,
                                     const std::vector<int>& indices,
                                     const std::vector<T>& mat_B1,
                                     std::vector<T>& mat_D, int head_sz,
                                     int seq, int B, int index, int head_num) {
    std::vector<T> new_mat_B0;
    std::vector<T> new_mat_B1;
    std::vector<T> new_key_mask;
    std::vector<T> new_vec_A;
    for (int i = 0; i < index; i++) {
      for (int s = 0; s < seq; s++) {
        for (int k = 0; k < head_sz * head_num; k++) {
          int tmp = indices[i];
          new_mat_B0.push_back(mat_B0[tmp * seq * head_sz * head_num +
                                      s * head_sz * head_num + k]);
          new_mat_B1.push_back(mat_B1[tmp * seq * head_sz * head_num +
                                      s * head_sz * head_num + k]);
          new_key_mask.push_back(keymask[tmp * seq + s]);
        }
      }
    }
    // K = K / head_num;   // head size
    float mul = 1.0f / sqrtf(head_sz * 1.f);
    for (int h = 0; h < head_num; h++) {
      for (int b = 0; b < index; b++) {
        float sum = 0.f;
        std::vector<float> localExps;
        for (int s = 0; s < seq; s++) {
          float psum = 0.f;
          for (int k = 0; k < head_sz; k++) {
            // (indezx, 1, head_num, head_sz)
            float areg = float(mat_A[head_num * b * head_sz + h * head_sz + k]);
            float breg =
                float(new_mat_B0[head_num * b * seq * head_sz +
                                 head_num * s * head_sz + h * head_sz + k]);
            psum += areg * breg;
          }
          float res = float(psum) * float(mul);
          float fill = float(-4294967296);
          res = new_key_mask[b * seq + s] ? res : fill;
          float local = exp(float(res));
          localExps.push_back(local);
          sum += local;
        }
        for (int s = 0; s < seq; s++) {
          float res = float(0);
          if (sum == 0.f)
            res = float(1. / seq);
          else
            res = float((localExps[s]) / sum);

          new_vec_A.push_back(Eigen::half(res));
        }
      }
    }
    for (int b = 0; b < index; b++) {
      for (int k = 0; k < head_sz; k++) {
        for (int h = 0; h < head_num; h++) {
          float psum = 0.f;
          for (int s = 0; s < seq; s++) {
            float areg = float(new_vec_A[h * index * seq + b * seq + s]);
            // (index, seq, head_num, head_sz)
            float breg =
                float(new_mat_B1[b * head_sz * head_num * seq +
                                 s * head_num * head_sz + h * head_sz + k]);
            psum += areg * breg;
          }
          mat_D[b * head_sz * head_num + h * head_sz + k] = T(psum);
        }
      }
    }
  }

  template <typename T>
  void RunGatherAttentionTest(const std::vector<T>& mat_A,
                              const std::vector<T>& mat_B0,
                              const std::vector<T>& keymask,
                              const std::vector<int>& indices,
                              const std::vector<T>& mat_B1,
                              const std::vector<T>& mat_D, int head_sz, int seq,
                              int B, int index, int head_num) {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));
    DataType dt = DataTypeToEnum<T>::v();

    TF_EXPECT_OK(NodeDefBuilder("gather_attention", "GatherAttention")
                     .Input(FakeInput(dt))        // 0
                     .Input(FakeInput(dt))        // 1
                     .Input(FakeInput(dt))        // 2
                     .Input(FakeInput(DT_INT32))  // 3
                     .Input(FakeInput(dt))        // 4
                     .Attr("head_num", head_num)
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    AddInputFromArray<T>(TensorShape({index, 1, head_sz * head_num}),
                         mat_A);  // 0
    AddInputFromArray<T>(TensorShape({B, seq, head_sz * head_num}),
                         mat_B0);                           // 1
    AddInputFromArray<T>(TensorShape({B, seq}), keymask);   // 1
    AddInputFromArray<int>(TensorShape({index}), indices);  // 1
    AddInputFromArray<T>(TensorShape({B * seq, head_sz * head_num}),
                         mat_B1);  // 1

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), dt,
                    TensorShape({index, 1, head_sz * head_num}));
    test::FillValues<T>(&expected, mat_D);

    test::ExpectTensorNear<T>(expected, *GetOutput(0), 0.001);
  }
};

TEST_F(GatherAttentionTest, Simple2) {
  int head_num = 8;
  int head_sz = 32;
  int b = 3;
  int index = 4;
  int seq = 32;

  srand(9);
  std::vector<Eigen::half> mat_A;
  std::vector<Eigen::half> mat_B0;
  std::vector<Eigen::half> keymask;
  std::vector<int> indices;
  std::vector<Eigen::half> mat_B1;
  std::vector<Eigen::half> mat_D;

  // input0 (head_num * index, 1, head_sz)
  // input1（B, seq, head_sz * head_num）
  // input2 (B, seq)
  // input3 (index)
  // input4 (B, seq, head_sz * head_num)
  // output (index, 1, head_sz * head_num)
  // -----------------------------------------------------------------------
  // gemv0 : (head_num * indices, 1, head_sz) * (B, head_sz * head_num, seq)
  // output0: (head_num * indices, 1, seq)
  // -----------------------------------------------------------------------
  // gemv1 : (head_num * indices, 1, seq) * (B, seq, head_sz * head_num)
  // output1: (indices, 1, head_sz * head_num)

  for (int i = 0; i < index * head_num * head_sz; ++i) {
    mat_A.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < b * seq * head_sz * head_num; ++i) {
    mat_B0.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < b * seq; ++i) {
    keymask.push_back(Eigen::half(1));
  }
  indices.push_back(0);
  indices.push_back(2);
  indices.push_back(0);
  indices.push_back(1);
  for (int i = 0; i < b * seq * head_sz * head_num; ++i) {
    mat_B1.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < index * head_num * head_sz; ++i) {
    mat_D.push_back(Eigen::half(128));
  }

  RunUnfusedGatherAttentionTest<Eigen::half>(mat_A, mat_B0, keymask, indices,
                                             mat_B1, mat_D, head_sz, seq, b,
                                             index, head_num);
  RunGatherAttentionTest<Eigen::half>(mat_A, mat_B0, keymask, indices, mat_B1,
                                      mat_D, head_sz, seq, b, index, head_num);
}
}  // namespace
}  // namespace tensorflow