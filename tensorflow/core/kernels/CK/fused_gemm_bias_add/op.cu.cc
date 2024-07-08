#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
// #include "api.h"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
namespace tensorflow {
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using FP8 = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType = F16;
using B0DataType = F16;
using AccDataType = F32;
using CShuffleDataType = F32;
using D0DataType = F16;
using DsDataType = ck::Tuple<D0DataType>;
using EDataType = F16;

using A0Layout = Row;
using B0Layout = Col;
using D0Layout = Row;
using DsLayout = ck::Tuple<D0Layout>;
using ELayout = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add = ck::tensor_operation::element_wise::Add;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CDEElementOp = Add;

static constexpr auto GemmSpec =
    ck::tensor_operation::device::GemmSpecialization::MNPadding;
using GPUDevice = Eigen::GpuDevice;
using DeviceOpInstance =
    ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
        Row, Col, DsLayout, ELayout, A0DataType, B0DataType, DsDataType,
        EDataType, AccDataType, CShuffleDataType, AElementOp, BElementOp,
        CDEElementOp, GemmSpec, 256, 256, 128, 64, 16, 16, 32, 32, 4, 2,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0, S<4, 64, 1>,
        S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0, 1, 1, S<1, 32, 1, 8>, S<8, 8, 1>,
        ck::BlockGemmPipelineScheduler::Interwave,
        ck::BlockGemmPipelineVersion::v1, F16>;
namespace functor {
template <typename dataTP_>
struct Fused_Gemm_Bias_Add_Functor<GPUDevice, dataTP_> {
 public:
  static Status Compute(const GPUDevice& d, int M, int N, int K, int Batch,
                        const void* a0, const void* b0, const void* d0,
                        void* e) {
    const bool time_kernel = std::getenv("TF_CK_TIME_KERNEL") != nullptr;
    const auto& stream = d.stream();
    auto device_op = DeviceOpInstance{};
    auto invoker = device_op.MakeInvoker();
    constexpr ck::index_t NumDTensor = DsDataType::Size();
    auto argument = device_op.MakeArgument(
        a0, b0, std::array<const void*, NumDTensor>{d0}, e, M, N, K, K, K,
        std::array<ck::index_t, NumDTensor>{ck::Number<0>{}}, N, AElementOp{},
        BElementOp{}, CDEElementOp{});
    if (!device_op.IsSupportedArgument(argument)) {
      return errors::InvalidArgument(
          "wrong! device_gemm with the specified compilation parameters does "
          "not support this GEMM problem");
    }
    float ave_time =
        invoker.Run(argument, StreamConfig{stream, time_kernel, 20, 50});
    if (time_kernel) {
      std::size_t flop = std::size_t(2) * M * N * K;
      std::size_t num_btype = sizeof(A0DataType) * M * K +
                              sizeof(B0DataType) * K * N +
                              sizeof(EDataType) * M * N;

      float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

      float gb_per_sec = num_btype / 1.E6 / ave_time;
      LOG(INFO) << "Running time: " << ave_time << " ms, " << tflops
                << " TFlops, " << gb_per_sec << " GB/s";
    }
    return Status::OK();
  }
};  // struct Fused_Gemm_Bias_Add_Functor
}  // namespace functor
template struct functor::Fused_Gemm_Bias_Add_Functor<GPUDevice, Eigen::half>;
}  // namespace tensorflow
#endif
