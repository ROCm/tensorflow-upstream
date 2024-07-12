#include <functional>
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
class FusedTileGemmTest : public OpsTestBase {
 protected:
  void RunStandardGemvTest(const std::vector<Eigen::half>& mat_A,
                           const std::vector<Eigen::half>& mat_B,
                           std::vector<Eigen::half>& mat_D, int batch, int seq,
                           int head_sz, int head_num) {
    for (int b = 0; b < batch; b++) {
      for (int h = 0; h < head_num; h++) {
        for (int n = 0; n < head_sz; n++) {
          float psum = 0.f;
          for (int k = 0; k < seq; k++) {
            float areg = float(mat_A[b * seq * head_num + h * seq + k]);  // a0
            float breg =
                float(mat_B[k * head_sz * head_num + head_sz * h + n]);  // b1
            psum += areg * breg;
          }
          mat_D[b * head_sz * head_num + h * head_sz + n] = Eigen::half(psum);
        }
      }
    }
  }

  void RunFusedTileGemmTest(const std::vector<Eigen::half>& mat_A,
                            const std::vector<Eigen::half>& mat_B,
                            const std::vector<Eigen::half>& mat_D, int batch,
                            int seq, int head_sz, int head_num) {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));
    // (batch, head_num, seq) * (1, seq, head_num * head_sz)
    // (batch, 1, head_num * head_sz)

    TF_EXPECT_OK(NodeDefBuilder("fused_tile_gemm", "FusedTileGemm")
                     .Input(FakeInput(DT_HALF))
                     .Input(FakeInput(DT_HALF))
                     .Attr("head_num", head_num)
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    AddInputFromArray<Eigen::half>(TensorShape({batch, head_num, seq}),
                                   mat_A);  // 0
    AddInputFromArray<Eigen::half>(TensorShape({1, seq, head_sz * head_num}),
                                   mat_B);  // 1

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), DT_HALF,
                    TensorShape({batch, 1, head_sz * head_num}));
    test::FillValues<Eigen::half>(&expected, mat_D);
    test::ExpectTensorNear<Eigen::half>(expected, *GetOutput(0), 0.01);
  }
};

TEST_F(FusedTileGemmTest, Half) {
  int batch = 100;
  int seq = 48;
  int head_sz = 32;
  int head_num = 8;

  srand(9);
  std::vector<Eigen::half> mat_A;
  std::vector<Eigen::half> mat_B;
  std::vector<Eigen::half> mat_D;
  // (batch, head_num, seq) * (1, seq, head_num * head_sz)
  // (batch, 1, head_num * head_sz)

  for (int i = 0; i < batch * seq * head_num; ++i) {
    mat_A.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  for (int i = 0; i < seq * head_sz * head_num; ++i) {
    mat_B.push_back(Eigen::half(rand() / double(RAND_MAX)));
  }
  // expect_data
  for (int i = 0; i < batch * head_num * head_sz; ++i) {
    mat_D.push_back(Eigen::half(3));
  }

  // TF_ASSERT_OK(RunOpKernel());
  RunStandardGemvTest(mat_A, mat_B, mat_D, batch, seq, head_sz, head_num);
  RunFusedTileGemmTest(mat_A, mat_B, mat_D, batch, seq, head_sz, head_num);
}
}  // namespace
}  // namespace tensorflow
