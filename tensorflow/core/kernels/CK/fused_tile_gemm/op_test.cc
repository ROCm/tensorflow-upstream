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
    class FusedTileGemmTest : public OpsTestBase {
    protected:
        void RunUnfusedTest(const std::vector<Eigen::half>& mat_A,
                            const std::vector<Eigen::half>& mat_B,
                            std::vector<Eigen::half>& mat_C,
                            int M,
                            int N,
                            int K) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float psum = 0.f;
                    for (int k = 0; k < K; k++) {
                        float areg = float(mat_A[m * K + k]);
                        float breg = float(mat_B[k * N + n]);
                        psum += areg * breg;
                    }
                    mat_C[m * N + n] = Eigen::half(psum);
                }
            }
        }

        void RunFusedTileGemmTest(const std::vector<Eigen::half>& mat_A,
                          const std::vector<Eigen::half>& mat_B,
                          std::vector<Eigen::half>& mat_C,
                          int M,
                          int N,
                          int K) {
            SetDevice(DEVICE_GPU,
                    std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                        "GPU", {}, "/job:a/replica:0/task:0")));

            TF_EXPECT_OK(NodeDefBuilder("fused_tile_gemm", "FusedTileGemm")
                            .Input(FakeInput(DT_HALF))
                            .Input(FakeInput(DT_HALF))
                            .Attr("head_num", 1)
                            .Finalize(node_def()));

            TF_EXPECT_OK(InitOp());

            AddInputFromArray<Eigen::half>(TensorShape({M, K}), mat_A);  // 0
            AddInputFromArray<Eigen::half>(TensorShape({K, N}), mat_B);  // 1

            // Run the kernel
            TF_ASSERT_OK(RunOpKernel());

            // Allocate and fill the expected output tensor
            Tensor expected(allocator(), DT_HALF, TensorShape({M, N}));
            test::FillValues<Eigen::half>(&expected, mat_C);

            std::cout << "Expected output tensor (first few values):" << std::endl;
            for (int i = 0; i < std::min(M * N, 10); ++i) {
                std::cout << float(expected.flat<Eigen::half>()(i)) << " ";
            }
            std::cout << std::endl;

            // Retrieve and print the actual output tensor
            Tensor output = *GetOutput(0);
            auto output_flat = output.flat<Eigen::half>();

            // Enhanced debugging: Print all values if they do not match
            auto expected_flat = expected.flat<Eigen::half>();
            bool all_close = true;
            float abs_err = 0.0005;

            test::ExpectTensorNear<Eigen::half>(expected, *GetOutput(0), 0.0005);

            std::cout << "Actual output tensor (first few values):" << std::endl;
            for (int i = 0; i < std::min(M * N, 10); ++i) {
                std::cout << float(output_flat(i)) << " ";
            }
            std::cout << std::endl;

            for (int i = 0; i < M * N; ++i) {
                if (abs(float(expected_flat(i)) - float(output_flat(i))) > abs_err) {
                    std::cout << "Mismatch at index " << i << ": expected " 
                            << float(expected_flat(i)) << ", got " 
                            << float(output_flat(i)) << std::endl;
                    all_close = false;
                }
            }

            if (all_close) {
                std::cout << "All values are within tolerance." << std::endl;
            } else {
                std::cout << "There are mismatched values." << std::endl;
            }

            // Assert that all values are within the expected tolerance
            EXPECT_TRUE(all_close) << "Custom operation result does not match reference result";
        }

    };

    TEST_F(FusedTileGemmTest, MediumMatrices) {
        const int M = 256;
        const int N = 256;
        const int K = 256;

        std::vector<Eigen::half> mat_A(M * K);
        std::vector<Eigen::half> mat_B(K * N);
        std::vector<Eigen::half> mat_C(M * N);

        // Initialize matrices A and B with distinct values
        for (int i = 0; i < M * K; ++i) {
            mat_A[i] = Eigen::half(i % 3 + 1); // Example distinct values
        }
        for (int i = 0; i < K * N; ++i) {
            mat_B[i] = Eigen::half((i % 5) + 1); // Example distinct values
        }

        // Print initial values of matrices A and B
        std::cout << "Initial values of matrix A (first few values):" << std::endl;
        for (int i = 0; i < std::min(M * K, 10); ++i) {
            std::cout << float(mat_A[i]) << " ";
        }
        std::cout << std::endl;

        std::cout << "Initial values of matrix B (first few values):" << std::endl;
        for (int i = 0; i < std::min(K * N, 10); ++i) {
            std::cout << float(mat_B[i]) << " ";
        }
        std::cout << std::endl;

        // Run reference computation
        RunUnfusedTest(mat_A, mat_B, mat_C, M, N, K);
        
        // Print reference result (only first few values for brevity)
        std::cout << "Reference result (first few values):" << std::endl;
        for (int i = 0; i < std::min(M * N, 10); ++i) {
            std::cout << float(mat_C[i]) << " ";
        }
        std::cout << std::endl;

        // Run custom operation
        RunFusedTileGemmTest(mat_A, mat_B, mat_C, M, N, K);
    }
}
} // namespace tensorflow
