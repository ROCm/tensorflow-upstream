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
using GPUDevice = Eigen::GpuDevice;
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
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
    ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceOpInstance =
    ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3
    // clang-format off
///######|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     DsData|     EData|     AccData|         CShuffle|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
///######|         |         |         |        |       Type|       Type|       Type|      Type|        Type|         DataType| Elementwise| Elementwise|  Elementwise| Spacialization|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
///######|         |         |         |        |           |           |           |          |            |                 |   Operation|   Operation|    Operation|               |      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
///######|         |         |         |        |           |           |           |          |            |                 |            |            |             |               |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |    S<C, D0, D1>|
///###### RRR
      ///<      Row,      Row, DsLayout, ELayout, A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   256,   128,    64,  16,   4,  32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<16, 16, 1>,    S<0, 2, 1>,     S<0, 2, 1>,             1,               8,              4,          0,          1,           1,               S<1, 32, 1, 8>,      S<8, 8, 1>,  ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, F16>;
///###### RCR
        <Row, Col, DsLayout, ELayout, A0DataType, B0DataType,
        DsDataType, EDataType, AccDataType, CShuffleDataType,
        AElementOp,  BElementOp, CDEElementOp,       GemmSpec,
        256,
        128, 128,
        64, 8, 8,
        16,   16,
        4,    4,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 0,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 0,
        1, 2, S<1, 32, 1, 8>,      S<8, 8>,  ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, F16>;
// clang-format on
namespace functor {
template <typename dataTP_>
struct FusedGemmBiasAddFunctor<GPUDevice, dataTP_> {
 public:
  static Status Compute(const GPUDevice& d, int M, int N, int K, const void* a0,
                        const void* b0, const void* d0, void* e) {
    const hipStream_t stream = d.stream();
    auto device_op = DeviceOpInstance{};
    auto invoker = device_op.MakeInvoker();
    constexpr ck::index_t NumDTensor = DsDataType::Size();
    auto argument = device_op.MakeArgument(
        a0, b0, std::array<const void*, NumDTensor>{d0}, e, M, N, K, K, K,
        std::array<ck::index_t, NumDTensor>{ck::Number<0>{}}, N, AElementOp{},
        BElementOp{}, CDEElementOp{});
    if (!device_op.IsSupportedArgument(argument)) {
      return errors::InvalidArgument(device_op.GetTypeString(),
                                     " does not support this problem");
    }
    invoker.Run(argument, StreamConfig{stream, false, 0, 20, 50});

    return Status::OK();
  }
};  // struct Fused_Gemm_Bias_Add_Functor
}  // namespace functor
template struct functor::FusedGemmBiasAddFunctor<GPUDevice, Eigen::half>;
}  // namespace tensorflow
#endif
