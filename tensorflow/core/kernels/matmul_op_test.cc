/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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

namespace tensorflow {

template <typename T>
static Graph* Matmul(int m, int k, int n, bool transpose_a, bool transpose_b,
                     DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<T>().setRandom();
  test::graph::Matmul(g, test::graph::Constant(g, in0),
                      test::graph::Constant(g, in1), transpose_a, transpose_b);
  return g;
}

#define BM_MatmulDev(M, K, N, TA, TB, T, TFTYPE, DEVICE)                       \
  static void BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE( \
      int iters) {                                                             \
    testing::UseRealTime();                                                    \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);        \
    test::Benchmark(#DEVICE, Matmul<T>(M, K, N, TA, TB, TFTYPE)).Run(iters);   \
  }                                                                            \
  BENCHMARK(BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE);

#define BM_Matmul(M, K, N, TA, TB)                                       \
  BM_MatmulDev(M, K, N, TA, TB, Eigen::half, DT_HALF, gpu);                   \
  // BM_MatmulDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, gpu); \
/* Uncomment to enable benchmarks for double/complex128: */              \
// BM_MatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, cpu);                   \
// BM_MatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, cpu); \
// BM_MatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, gpu);                   \
// BM_MatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, gpu);

// Batch size of 1 included for inference.
// Typical fully connected layers
// BM_Matmul(1, 512, 512, false, false);
// BM_Matmul(8, 512, 512, false, false);
// BM_Matmul(16, 512, 512, false, false);
// BM_Matmul(128, 512, 512, false, false);

// BM_Matmul(1, 1024, 1024, false, false);
// BM_Matmul(8, 1024, 1024, false, false);
// BM_Matmul(16, 1024, 1024, false, false);
// BM_Matmul(128, 1024, 1024, false, false);
// BM_Matmul(4096, 4096, 4096, false, false);

// // Backward for fully connected layers
// BM_Matmul(1, 1024, 1024, false, true);
// BM_Matmul(8, 1024, 1024, false, true);
// BM_Matmul(16, 1024, 1024, false, true);
// BM_Matmul(128, 1024, 1024, false, true);

// // Forward softmax with large output size
// BM_Matmul(1, 200, 10000, false, false);
// BM_Matmul(8, 200, 10000, false, false);
// BM_Matmul(20, 200, 10000, false, false);
// BM_Matmul(20, 200, 20000, false, false);

// // Backward softmax with large output size
// BM_Matmul(1, 10000, 200, false, true);
// BM_Matmul(1, 10000, 200, false, false);
// BM_Matmul(8, 10000, 200, false, true);
// BM_Matmul(20, 10000, 200, false, true);
// BM_Matmul(20, 20000, 200, false, true);

// // Test some matrix-vector multiplies.
// BM_Matmul(50, 50, 1, false, false);
// BM_Matmul(50, 50, 1, true, false);
// BM_Matmul(50, 50, 1, false, true);
// BM_Matmul(50, 50, 1, true, true);
// BM_Matmul(500, 500, 1, false, false);
// BM_Matmul(500, 500, 1, true, false);
// BM_Matmul(500, 500, 1, false, true);
// BM_Matmul(500, 500, 1, true, true);
// BM_Matmul(2000, 2000, 1, false, false);
// BM_Matmul(2000, 2000, 1, true, false);
// BM_Matmul(2000, 2000, 1, false, true);
// BM_Matmul(2000, 2000, 1, true, true);

// Test some vector-matrix multiplies.
// BM_Matmul(1, 50, 50, false, false);
// BM_Matmul(1, 50, 50, true, false);
// BM_Matmul(1, 50, 50, false, true);
// BM_Matmul(1, 50, 50, true, true);
// BM_Matmul(1, 500, 500, false, false);
// BM_Matmul(1, 500, 500, true, false);
// BM_Matmul(1, 500, 500, false, true);
// BM_Matmul(1, 500, 500, true, true);
// BM_Matmul(1, 2000, 2000, false, false);
// BM_Matmul(1, 2000, 2000, true, false);
// BM_Matmul(1, 2000, 2000, false, true);
// BM_Matmul(1, 2000, 2000, true, true);
// BM_Matmul(1, 1024, 3224, true, true);

// Test some rank-one products.
// BM_Matmul(50, 1, 50, false, false);
// BM_Matmul(50, 1, 50, true, false);
// BM_Matmul(50, 1, 50, false, true);
// BM_Matmul(50, 1, 50, true, true);
// BM_Matmul(500, 1, 500, false, false);
// BM_Matmul(500, 1, 500, true, false);
// BM_Matmul(500, 1, 500, false, true);
// BM_Matmul(500, 1, 500, true, true);
// BM_Matmul(2000, 1, 2000, false, false);
// BM_Matmul(2000, 1, 2000, true, false);
// BM_Matmul(2000, 1, 2000, false, true);
// BM_Matmul(2000, 1, 2000, true, true);

namespace{
class MatMulfp16Test : public OpsTestBase {
 protected:
void RunNaiveMatmulTest(const std::vector<Eigen::half>& mat_A,
                        const std::vector<Eigen::half>& mat_B,
                        std::vector<Eigen::half>& mat_C,
                        int M,
                        int N,
                        int K){
    for (int m = 0; m < M; m++){
      for (int n = 0; n < N; n++){
        float psum = 0;
        for (int k = 0; k < K; k++){
          psum += float(mat_A[m * K + k]) * float(mat_B[k * N + n]);
        }
        mat_C[m * N + n] = Eigen::half(psum);
      }
    }
}

void RunMatmulTest(const std::vector<Eigen::half>& mat_A,
                  const std::vector<Eigen::half>& mat_B,
                  const std::vector<Eigen::half>& mat_C,
                  int M,
                  int N,
                  int K){
    SetDevice(DEVICE_GPU,
             std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                 "GPU", {}, "/job:a/replica:0/task:0")));
    
    TF_EXPECT_OK(NodeDefBuilder("matmul", "MatMul")
                     .Input(FakeInput(DT_HALF)) //0
                     .Input(FakeInput(DT_HALF)) //1
                     .Attr("transpose_a", false)
                     .Attr("transpose_b", false)
                     .Finalize(node_def()));
    
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<Eigen::half>(TensorShape({M, K}), mat_A);
    AddInputFromArray<Eigen::half>(TensorShape({K, N}), mat_B);

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), DT_HALF, TensorShape({M, N}));
    test::FillValues<Eigen::half>(&expected, mat_C);

    test::ExpectTensorNear<Eigen::half>(expected, *GetOutput(0), 5e-4);
}
};

TEST_F(MatMulfp16Test, Half) {
  int M = 1;
  int N = 1024;
  int K = 3224;

  std::vector<Eigen::half> mat_A;
  std::vector<Eigen::half> mat_B;
  std::vector<Eigen::half> mat_C;

  for (int i = 0; i < M * K; ++i) {
    mat_A.push_back(Eigen::half(rand() / double(RAND_MAX) - 0.5));
  }
  
  for (int i = 0; i < N * K; ++i){
    mat_B.push_back(Eigen::half(rand() / double(RAND_MAX) - 0.5));
  }

  for (int i = 0; i < M * N; ++i) {
    mat_C.push_back(Eigen::half(128));
  }

  RunNaiveMatmulTest(mat_A, mat_B, mat_C, M, N, K);
  RunMatmulTest(mat_A, mat_B, mat_C, M, N, K);
}

}

}  // end namespace tensorflow
