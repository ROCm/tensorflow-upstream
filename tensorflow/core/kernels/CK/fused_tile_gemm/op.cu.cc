#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"
#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType = ck::half_t;
using BDataType = ck::half_t;
using AccDataType = float;
using CShuffleDataType = ck::half_t;
using CDataType = ck::half_t;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault =
    ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// clang-format off
using DeviceGemmV2Instance = 
    ck::tensor_operation::device::DeviceBatchedGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        128,
        16,  64,  64,
        8,   4,
        16,  16,
        1,   2,
        S<8, 16, 1>,  S<1, 0, 2>,   S<1, 0, 2>,
        2,   8,   8,   0,
        S<16, 8, 1>,   S<0, 2, 1>,   S<0, 2, 1>,
        1,   8,   4,   0,
        1,   1,   S<1, 16, 1, 8>,   4,
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v1>;


namespace functor {
template <typename dataTP_>
struct FusedTileGemmFunctor<GPUDevice, dataTP_> {
 public:
  static Status Compute(const GPUDevice& d, const void* mat_A,
                            const void* mat_B,
                            void* mat_D, int batch,
                            int seq, int head_sz, int head_num) {
    const bool time_kernel = std::getenv("TF_CK_TIME_KERNEL") != nullptr;
    const auto& stream = d.stream();
    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto gemm      = DeviceGemmV2Instance{};
    auto invoker   = gemm.MakeInvoker();

    auto argument = gemm.MakeArgument(static_cast<const ADataType*>(mat_A),
                                      static_cast<const BDataType*>(mat_B),
                                      static_cast<CDataType*>(mat_D),
                                      batch,
                                      head_sz,
                                      seq,
                                      head_num*seq,
                                      head_num*head_sz,
                                      head_num*head_sz,
                                      seq,
                                      head_sz,
                                      head_sz,
                                      head_num,
                                      a_element_op,
                                      b_element_op,
                                      c_element_op);
    if (!gemm.IsSupportedArgument(argument)) {
     return errors::InvalidArgument(
          gemm.GetTypeString(), " does not support this problem");
    }

    float ave_time =
        invoker.Run(argument, StreamConfig{stream, time_kernel, 20, 50});
    if (time_kernel) {
    //   std::size_t flop = std::size_t(2) * M * N * K;
    //   std::size_t num_btype = sizeof(A0DataType) * M * K +
    //                           sizeof(B0DataType) * K * N +
    //                           sizeof(EDataType) * M * N;

    //   float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    //   float gb_per_sec = num_btype / 1.E6 / ave_time;
    //   LOG(INFO) << "Running time: " << ave_time << " ms, " << tflops
    //             << " TFlops, " << gb_per_sec << " GB/s";
    }
    return Status::OK();
  }
};  // struct Fused_Gemm_Bias_Add_Functor
}  // namespace functor
template struct functor::FusedTileGemmFunctor<GPUDevice, Eigen::half>;
}  // namespace tensorflow
#endif
