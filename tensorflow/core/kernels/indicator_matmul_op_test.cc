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

#include <array>

#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {


struct TestParams {
  int parallel_num, batch_a, batch_b;
  int m, k, n;
  bool adjoint_a, adjoint_b;
};

template <typename T>
struct TensorCreator {
  using IndexType = int64;
  using Params = TestParams;
  using TensorVec = std::array< Tensor, 3 >;

  TensorVec CreateTensors(const Params& p) {
    
    DataType dtype = DataTypeToEnum<T>::v(),
       index_dtype = DataTypeToEnum<IndexType>::v();

    Tensor lhs(dtype, p.adjoint_a ? TensorShape{p.parallel_num, p.batch_a, p.k, p.m}
                                  : TensorShape{p.parallel_num, p.batch_a, p.m, p.k});
    lhs.flat<T>() = lhs.flat<T>().setRandom();

    Tensor rhs(dtype, p.adjoint_b ? TensorShape{p.parallel_num, p.batch_b, p.n, p.k}
                                  : TensorShape{p.parallel_num, p.batch_b, p.k, p.n});
    rhs.flat<T>() = rhs.flat<T>().setRandom();
    rhs.flat<T>() -= rhs.flat<T>().constant(static_cast<T>(0.5f));

    // NOTE indicator indices should be: [0, batch_a)
    IndexType max_idx = p.batch_a; // float range is [0, 1)
    using FT = double;
    Tensor indF(DataTypeToEnum<FT>::v(), {p.batch_b});
    Tensor ind(index_dtype, {p.batch_b});

    indF.flat<FT>() = indF.flat<FT>().setRandom();
    indF.flat<FT>() *= indF.flat<FT>().constant(static_cast<FT>(max_idx));
    ind.flat<IndexType>() = indF.flat<FT>().cast< IndexType >();

    return std::array< Tensor, 3 >{std::move(lhs), std::move(rhs), std::move(ind)};
  }
};  

template <typename T>
class IndicatorMatMulOpTestBase : public OpsTestBase, public TensorCreator<T> {
 protected:

  using Params = typename TensorCreator<T>::Params;
  using IndexType = typename TensorCreator<T>::IndexType;
  using TensorVec = typename TensorCreator<T>::TensorVec;

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor. Optional `fetch_node` parameter
  // allows to define a fetch node directly using a NodeDef for the ops that are
  // not supported by the C++ Api.
  void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                   Tensor* output, bool allow_gpu_device,
                   const NodeDef* fetch_node = nullptr) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    if (fetch_node) {
      *graph.add_node() = *fetch_node;
    }

    // We really want to make sure that graph executed exactly as we passed it
    // to the session, so we disable various optimizations.
    tensorflow::SessionOptions session_options;

    // Disable common runtime constant folding.
    session_options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(OptimizerOptions::L0);

    // Disable Grappler optimizations for tests.
    tensorflow::RewriterConfig* cfg =
        session_options.config.mutable_graph_options()
            ->mutable_rewrite_options();
    cfg->set_constant_folding(tensorflow::RewriterConfig::OFF);
    cfg->set_layout_optimizer(tensorflow::RewriterConfig::OFF);
    cfg->set_remapping(tensorflow::RewriterConfig::OFF);

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(session_options));

    std::vector<DeviceAttributes> available_devices;
    TF_ASSERT_OK(session->ListDevices(&available_devices))
        << "Failed to get available session devices";

    // Check if session has an available GPU device.
    const bool has_gpu_device =
        absl::c_any_of(available_devices, [](const DeviceAttributes& device) {
          return device.device_type() == DEVICE_GPU;
        });

    // If fused computation implemented only for CPU, in this test we don't want
    // to compare GPU vs CPU numbers, so place all nodes on CPU in this case.
    const bool place_all_on_gpu = allow_gpu_device && has_gpu_device;

    const string device = place_all_on_gpu ? "/device:GPU:0" : "/device:CPU:0";
    VLOG(0) << "Running on device " << device;
    for (NodeDef& mutable_node : *graph.mutable_node()) {
      mutable_node.set_device(device);
    }

    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));

    *output = unfused_tensors[0];
  }

  void RunIndicatorMatMulOp(const Params& p, const TensorVec& inputs, 
                            Tensor* output, bool allow_gpu_device) {

    Scope root = tensorflow::Scope::NewRootScope();

    DataType dtype = DataTypeToEnum<T>::v(),
       index_dtype = DataTypeToEnum<IndexType>::v();

    Output 
      lhs = ops::Const(root.WithOpName("lhs"), Input::Initializer(inputs[0])),
      rhs = ops::Const(root.WithOpName("rhs"), Input::Initializer(inputs[1])),
      ind = ops::Const(root.WithOpName("ind"), Input::Initializer(inputs[2]));
#if 0
    NodeDef indicator_matmul;
    TF_EXPECT_OK(NodeDefBuilder("indicator_matmul", "ParallelIndicatorMatMul")
                     .Input({lhs.name(), 0, dtype})
                     .Input({rhs.name(), 0, dtype})
                     .Input({ind.name(), 1, index_dtype})
                     .Attr("adj_x", p.adjoint_a)
                     .Attr("adj_y", p.adjoint_b)
                     .Attr("parallel_num", p.parallel_num)
                     .Finalize(&indicator_matmul));
    RunAndFetch(root, "indicator_matmul", output, allow_gpu_device,
                &indicator_matmul);
#else
    auto indicator_matmul = ops::ParallelIndicatorMatMul(
        root.WithOpName("indicator_matmul"), lhs, rhs, ind, p.parallel_num,
        ops::ParallelIndicatorMatMul::Attrs().
                        AdjX(p.adjoint_a).AdjY(p.adjoint_b));
    RunAndFetch(root, "indicator_matmul", output, allow_gpu_device);
#endif
  }

  void VerifyIndicatorMatMul(const Params& p) {

    VLOG(0) << "----------------------------------- Running " << p.parallel_num 
            << "x" << p.batch_a << "x" << p.batch_b << "; m/n/k: "
            << p.m << "/" << p.n << "/" << p.k << "; adj_a/b: "
            << p.adjoint_a << "/" << p.adjoint_b;

    DataType dtype = DataTypeToEnum<T>::v();
    auto ts = this->CreateTensors(p);

    VLOG(0) << "Indices: " << ts[2].DebugString(16);

    Tensor cpu_result, gpu_result;

    RunIndicatorMatMulOp(p, ts, &cpu_result, /*allow_gpu_device*/false);
    RunIndicatorMatMulOp(p, ts, &gpu_result, /*allow_gpu_device*/true);

    ASSERT_EQ(cpu_result.dtype(), gpu_result.dtype());
    ASSERT_EQ(cpu_result.shape(), gpu_result.shape());

    double atol = dtype == DT_HALF ? 1e-2 : 1e-5,
           rtol = dtype == DT_HALF ? 1e-2 : -1.0;
    test::ExpectClose(cpu_result, gpu_result, atol, rtol);
  }
}; // IndicatorMatMulOpTestBase

template <typename T>
class IndicatorMatMulOpTest : public IndicatorMatMulOpTestBase<T> {};

TYPED_TEST_SUITE_P(IndicatorMatMulOpTest);

#if 0
#define ENUM_PARAMS(parallel_num, batch_a, batch_b, m, k, n)  \
  this->VerifyIndicatorMatMul({parallel_num, batch_a, batch_b, m, k, n, false, false}); \
  this->VerifyIndicatorMatMul({parallel_num, batch_a, batch_b, m, k, n, false, true}); \
  this->VerifyIndicatorMatMul({parallel_num, batch_a, batch_b, m, k, n, true, false}); \
  this->VerifyIndicatorMatMul({parallel_num, batch_a, batch_b, m, k, n, true, true})
#else
#define ENUM_PARAMS(parallel_num, batch_a, batch_b, m, k, n)  \
   this->VerifyIndicatorMatMul({parallel_num, batch_a, batch_b, m, k, n, false, false})
#endif

TYPED_TEST_P(IndicatorMatMulOpTest, ParallelMatMul1) {
  // parallel_num, batch_a, batch_b, m, k, n
  ENUM_PARAMS(40, 100, 75, 36, 128, 120);
}

TYPED_TEST_P(IndicatorMatMulOpTest, ParallelMatMul2) {
  // parallel_num, batch_a, batch_b, m, k, n
  ENUM_PARAMS(100, 50, 35, 136, 256, 120);
}

TYPED_TEST_P(IndicatorMatMulOpTest, ParallelMatMul3) {
  // parallel_num, batch_a, batch_b, m, k, n
  
  //ENUM_PARAMS(1, 1, 1, 12, 16, 8);
  ENUM_PARAMS(1, 10, 128, 75, 40, 100);
  //ENUM_PARAMS(8, 4, 16, 12, 20, 32);
  ENUM_PARAMS(8, 16, 16, 12, 40, 64);

  ENUM_PARAMS(32, 16, 16, 60, 40, 64);
  //ENUM_PARAMS(11, 111, 51, 137, 15, 111);
}
#undef ENUM_PARAMS

REGISTER_TYPED_TEST_SUITE_P(IndicatorMatMulOpTest,
                            ParallelMatMul1,
                            ParallelMatMul2,
                            ParallelMatMul3);

// TODO(ezhulenev): Add support for more data types.
using IndicatorMatMulDataTypes = ::testing::Types< float >;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, IndicatorMatMulOpTest,
                               IndicatorMatMulDataTypes);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* BenchIndicatorMatmul(const TestParams& p) {
  
  Graph* g = new Graph(OpRegistry::Global());
  TensorCreator<T> creator;
  auto inputs = creator.CreateTensors(p);

  Node *indicator_matmul;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "ParallelIndicatorMatMul")
                     .Input(test::graph::Constant(g, inputs[0]))
                     .Input(test::graph::Constant(g, inputs[1]))
                     .Input(test::graph::Constant(g, inputs[2]))
                     .Attr("adj_x", p.adjoint_a)
                     .Attr("adj_y", p.adjoint_b)
                     .Attr("parallel_num", p.parallel_num)
                     .Finalize(g, &indicator_matmul));
  return g;
}

 //parallel_num * batch_b * m * k;
#define BM_IndicatorMatmul(P, BA, BB, M, K, N, T, TNAME)                \
  static void BM_IndicatorMul##_##P##_##BA##_##BB##_##M##_##K##_##N##_##TNAME( \
      int iters) {                                                              \
    testing::UseRealTime();                                                     \
    VLOG(0) << "----------------------------------- Running " << P              \
            << "x" << BA << "x" << BB << "; m/n/k: "                            \
            << M << "/" << N << "/" << K;                                       \
    auto items = static_cast<int64>(iters) * P * BB * M * K * N / 1000;         \
    testing::ItemsProcessed(items);                                             \
    test::Benchmark("gpu", BenchIndicatorMatmul<T>(                 \
                {P, BA, BB, M, K, N, false, false})).Run(iters);                \
  }                                                                             \
  BENCHMARK(BM_IndicatorMul##_##P##_##BA##_##BB##_##M##_##K##_##N##_##TNAME);

BM_IndicatorMatmul(2, 1, 166, 200, 24, 1, float, fp32)
BM_IndicatorMatmul(4, 30, 166, 200, 24, 1, float, fp32)
BM_IndicatorMatmul(4, 30, 166, 24, 200, 1, float, fp32)
BM_IndicatorMatmul(10, 20, 30, 120, 40, 50, float, fp32)

BM_IndicatorMatmul(2, 1, 166, 200, 24, 1, Eigen::half, fp16)
BM_IndicatorMatmul(4, 30, 166, 200, 24, 1, Eigen::half, fp16)
BM_IndicatorMatmul(4, 30, 166, 24, 200, 1, Eigen::half, fp16)
BM_IndicatorMatmul(10, 20, 30, 120, 40, 50, Eigen::half, fp16)


// BM_IndicatorMatmul(10, 20, 30, 120, 80, 50, float)
// BM_IndicatorMatmul(10, 20, 30, 120, 80, 100, float)
// BM_IndicatorMatmul(20, 20, 30, 120, 40, 50, float)
// BM_IndicatorMatmul(30, 20, 30, 120, 40, 50, float)
// BM_IndicatorMatmul(40, 20, 30, 120, 40, 50, float)
// BM_IndicatorMatmul(40, 20, 30, 120, 80, 100, float)
// BM_IndicatorMatmul(40, 50, 30, 120, 40, 50, float)
// BM_IndicatorMatmul(60, 10, 40, 128, 64, 256, float)

// NOTE: run it with: ./bazel-bin/path/to/test --benchmarks=all

}  // end namespace tensorflow
