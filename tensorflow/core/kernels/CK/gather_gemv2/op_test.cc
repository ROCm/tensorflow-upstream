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

class GatherGemv2Test : public OpsTestBase {
 protected:
  template <typename T>
  void RunUnfusedGatherGemv2Test(const std::vector<T>& mat_A,
                                 const std::vector<T>& mat_B0,
                                 const std::vector<int>& indices,
                                 std::vector<T>& mat_D, int head_sz, int seq,
                                 int B, int index, int head_num) {
    std::vector<T> new_mat_B0;
    for (int i = 0; i < index; i++) {
      for (int s = 0; s < seq; s++) {
        for (int k = 0; k < head_sz * head_num; k++) {
          int tmp = indices[i];
          new_mat_B0.push_back(mat_B0[tmp * seq * head_sz * head_num +
                                      s * head_sz * head_num + k]);
        }
      }
    }
    // input0 (index , head_num,  seq)
    // input1（B, seq, head_sz * head_num）
    // input2 (index)
    // output (index, 1, head_sz * head_num)
    for (int h = 0; h < head_num; h++) {
      for (int b = 0; b < index; b++) {
        for (int k = 0; k < head_sz; k++) {
          float psum = 0.f;
          for (int s = 0; s < seq; s++) {
            float areg = float(mat_A[head_num * b * seq + h * seq + s]);
            float breg =
                float(new_mat_B0[head_num * b * seq * head_sz +
                                 head_num * s * head_sz + h * head_sz + k]);
            psum += areg * breg;
          }
          mat_D[b * head_num * head_sz + h * head_sz + k] = T(psum);
        }
      }
    }
  }

  template <typename T>
  void RunGatherGemv2Test(const std::vector<T>& mat_A,
                          const std::vector<T>& mat_B,
                          const std::vector<int>& indices,
                          const std::vector<T>& mat_D, int head_sz, int seq,
                          int B, int index, int head_num) {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));
    DataType dt = DataTypeToEnum<T>::v();

    TF_EXPECT_OK(NodeDefBuilder("gather_gemv2", "GatherGemv2")
                     .Input(FakeInput(dt))  // 0 q
                     .Input(FakeInput(dt))  // 1 k
                     .Input(FakeInput(DT_INT32))
                     .Attr("head_num", head_num)
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());
    // input0 (index , head_num,  seq)
    // input1（B, seq, head_sz * head_num）
    // input2 (index)
    // output (index, 1, head_sz * head_num)

    AddInputFromArray<T>(TensorShape({index, head_num, seq}), mat_A);  // 0
    AddInputFromArray<T>(TensorShape({B, seq, head_sz * head_num}),
                         mat_B);                            // 1
    AddInputFromArray<int>(TensorShape({index}), indices);  // 1

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), dt,
                    TensorShape({index, 1, head_num * head_sz}));
    test::FillValues<T>(&expected, mat_D);

    test::ExpectTensorNear<T>(expected, *GetOutput(0), 0.0001);
  }
};

TEST_F(GatherGemv2Test, Simple3) {
  int head_num = 4;
  int head_sz = 32;
  int b = 3;
  int index = 4;
  int seq = 48;

  srand(9);
  std::vector<Eigen::half> mat_A;
  std::vector<Eigen::half> mat_B;
  std::vector<int> indices;

  std::vector<Eigen::half> mat_D;
  // input0 (index , head_num, seq)
  // input1（B, seq, head_sz * head_num）
  // input2 (index)
  // output (index, 1, head_sz * head_num)

  for (int j = 0; j < index * head_num * seq; ++j) {
    mat_A.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < b * seq * head_num * head_sz; ++i) {
    mat_B.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  indices.push_back(0);
  indices.push_back(1);
  indices.push_back(0);
  indices.push_back(1);
  for (int j = 0; j < index * head_num * head_sz; ++j) {
    mat_D.push_back(Eigen::half(128));
  }

  RunUnfusedGatherGemv2Test<Eigen::half>(mat_A, mat_B, indices, mat_D, head_sz,
                                         seq, b, index, head_num);
  RunGatherGemv2Test<Eigen::half>(mat_A, mat_B, indices, mat_D, head_sz, seq, b,
                                  index, head_num);
}

}  // namespace
}  // namespace tensorflow
