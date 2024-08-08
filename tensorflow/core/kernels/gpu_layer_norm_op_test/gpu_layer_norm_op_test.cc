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
namespace{
class GpuLayerNormTest : public OpsTestBase {
 protected:
  template <typename T>
  void RunStandardTest(const std::vector<T>& input,
                       const std::vector<T>& gamma,
                       const std::vector<T>& beta,
                       std::vector<T>& res0,
                       int B, 
                       int K, 
                       int N){ 
     for(int b = 0; b < B; b++){
         for(int n = 0; n < N; n++){
             float eps = 1e-12;
             float sum = 0.f;
             for(int k = 0; k < K; k++){
                 float tmp = float(input[b * N * K + n * K  + k]);
                 sum += tmp;
             }
             float mean = sum / float(K);
             float var = 0.f;
             for(int k = 0; k < K; k++){
                 float tmp = float(input[b * N * K +n * K + k]);
                 float tmp_p = (tmp-mean) * (tmp-mean);
                 var += tmp_p;
             }
             var = 1.f / float( sqrt(var / float(K) + eps));
             for(int k = 0; k < K; k++){
                 float tmp = float(input[b * N * K +n * K + k]);
                 float tmp0 = var * float(gamma[k]);
                 float tmp1 = float(beta[k]) - tmp0 * mean;
                 float tmp2 = tmp0 * tmp + tmp1;
                 res0[b * N * K + n * K  + k] = T(tmp2);
             }
         } 
     } 
  }


  template <typename T>
  void RunGpuLayerNormTest3(const std::vector<T>& input,
                           const std::vector<T>& gamma,
                           const std::vector<T>& beta,
                           const std::vector<T>& res0,
                           int b, 
                           int k, 
                           int n){ 
    SetDevice(DEVICE_GPU,
             std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                 "GPU", {}, "/job:a/replica:0/task:0")));
    DataType dt = DataTypeToEnum<T>::v();

    TF_EXPECT_OK(NodeDefBuilder("gpu_layer_norm", "GpuLayerNorm")
                     .Input(FakeInput(dt)) 
                     .Input(FakeInput(dt)) 
                     .Input(FakeInput(dt)) 
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    AddInputFromArray<T>(TensorShape({b, n, k}), input);  
    AddInputFromArray<T>(TensorShape({k}), gamma);  
    AddInputFromArray<T>(TensorShape({k}), beta);   

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected0(allocator(), dt, TensorShape({b, n, k}));
    Tensor expected1(allocator(), dt, TensorShape({b, n}));
    Tensor expected2(allocator(), dt, TensorShape({b, n}));
    test::FillValues<T>(&expected0, res0);

    test::ExpectTensorNear<T>(expected0, *GetOutput(0), 0.0001);
  }



  template <typename T>
  void RunGpuLayerNormTest2(const std::vector<T>& input,
                           const std::vector<T>& gamma,
                           const std::vector<T>& beta,
                           const std::vector<T>& res0,
                           int b, 
                           int k){ 
    SetDevice(DEVICE_GPU,
             std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                 "GPU", {}, "/job:a/replica:0/task:0")));
    DataType dt = DataTypeToEnum<T>::v();

    TF_EXPECT_OK(NodeDefBuilder("gpu_layer_norm", "GpuLayerNorm")
                     .Input(FakeInput(dt)) 
                     .Input(FakeInput(dt)) 
                     .Input(FakeInput(dt)) 
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    AddInputFromArray<T>(TensorShape({b, k}), input);  
    AddInputFromArray<T>(TensorShape({k}), gamma);  
    AddInputFromArray<T>(TensorShape({k}), beta);   

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected0(allocator(), dt, TensorShape({b, k}));
    test::FillValues<T>(&expected0, res0);

    test::ExpectTensorNear<T>(expected0, *GetOutput(0), 0.0001);
  }

};

TEST_F(GpuLayerNormTest, Half2) {
   
    int b = 23;
    int n = 1;
    int k = 1996; 

    srand(9);
    std::vector<Eigen::half> input; 
    std::vector<Eigen::half> gamma; 
    std::vector<Eigen::half> beta;

    std::vector<Eigen::half> res0;
  
    for (int i = 0; i < k * b; ++i) {
        input.push_back(Eigen::half(rand() / double(RAND_MAX)));  
    }
    for (int i = 0; i < k ; ++i) {
        gamma.push_back(Eigen::half(rand() / double(RAND_MAX)));  
    }
    for (int i = 0; i < k ; ++i) {
        beta.push_back(Eigen::half(rand() / double(RAND_MAX)));  
    }
    for (int i = 0; i < b * k ; ++i) {
        res0.push_back(Eigen::half(0));  
    }

    RunStandardTest<Eigen::half>(input,  gamma, beta, res0, b, k, n);
    RunGpuLayerNormTest2<Eigen::half>(input, gamma, beta, res0, b, k);

}

TEST_F(GpuLayerNormTest, Float2) {
   
    int b = 1;
    int k = 1996; 
    int n = 1; 

    srand(21);
    std::vector<float> input; 
    std::vector<float> gamma; 
    std::vector<float> beta;

    std::vector<float> res0;
  
    for (int i = 0; i < n * k * b; ++i) {
        input.push_back(float(rand() / double(RAND_MAX)));  
    }
    for (int i = 0; i < k ; ++i) {
        gamma.push_back(float(rand() / double(RAND_MAX)));  
    }
    for (int i = 0; i < k ; ++i) {
        beta.push_back(0);
    }
    for (int i = 0; i < b * n * k ; ++i) {
        res0.push_back(0);  
    }


    RunStandardTest<float>(input, gamma, beta, res0, b, k, n);
    RunGpuLayerNormTest2<float>(input, gamma, beta, res0, b, k);
}


TEST_F(GpuLayerNormTest, Half3) {
   
    int b = 23;
    int k = 64; 
    int n = 100; 

    srand(9);
    std::vector<Eigen::half> input; 
    std::vector<Eigen::half> gamma; 
    std::vector<Eigen::half> beta;

    std::vector<Eigen::half> res0;
  
    for (int i = 0; i < n * k * b; ++i) {
        input.push_back(Eigen::half(rand() / double(RAND_MAX)));  
    }
    for (int i = 0; i < k ; ++i) {
        gamma.push_back(Eigen::half(rand() / double(RAND_MAX)));  
    }
    for (int i = 0; i < k ; ++i) {
        beta.push_back(Eigen::half(rand() / double(RAND_MAX)));  
    }
    for (int i = 0; i < b * n * k ; ++i) {
        res0.push_back(Eigen::half(0));  
    }

    RunStandardTest<Eigen::half>(input,  gamma, beta, res0, b, k, n);
    RunGpuLayerNormTest3<Eigen::half>(input, gamma, beta, res0, b, k, n);
}

TEST_F(GpuLayerNormTest, Float3) {
   
    int b = 5;
    int k = 32; 
    int n = 2; 

    srand(21);
    std::vector<float> input; 
    std::vector<float> gamma; 
    std::vector<float> beta;

    std::vector<float> res0;
  
    for (int i = 0; i < n * k * b; ++i) {
        input.push_back(float(rand() / double(RAND_MAX)));  
    }
    for (int i = 0; i < k ; ++i) {
        gamma.push_back(float(rand() / double(RAND_MAX)));  
    }
    for (int i = 0; i < k ; ++i) {
        beta.push_back(0);
    }
    for (int i = 0; i < b * n * k ; ++i) {
        res0.push_back(0);  
    }


    RunStandardTest<float>(input, gamma, beta, res0, b, k, n);
    RunGpuLayerNormTest3<float>(input, gamma, beta, res0, b, k, n);
}
}
}  // namespace tensorflow
