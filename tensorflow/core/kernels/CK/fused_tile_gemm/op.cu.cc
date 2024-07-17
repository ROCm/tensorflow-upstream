#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "op.h"
#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
template <typename T>
Status ComputeInternal(const GPUDevice& d, const void* mat_A, const void* mat_B,
                       void* mat_D, int batch, int seq, int head_sz,
                       int head_num) {
  using Row = ck::tensor_layout::gemm::RowMajor;
  using Col = ck::tensor_layout::gemm::ColumnMajor;

  using PassThrough = ck::tensor_operation::element_wise::PassThrough;

  using ADataType = T;
  using BDataType = T;
  using AccDataType = float;
  using CShuffleDataType = T;
  using CDataType = T;
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

    invoker.Run(argument, StreamConfig{stream, false, 0, 20, 50});

    return Status::OK();
  }

namespace functor {
template <typename T>
struct FusedTileGemmFunctor<GPUDevice, T> {
 public:
  static Status Compute(const GPUDevice& d, const void* mat_A,
                            const void* mat_B,
                            void* mat_D, int batch,
                            int seq, int head_sz, int head_num) {
    if constexpr(std::is_same_v<T, Eigen::half>){
      return ComputeInternal<ck::half_t>(d, mat_A, mat_B, mat_D, batch, seq, head_sz, head_num);
    }
    return Status::OK();
  }
};  // struct Fused_Gemm_Bias_Add_Functor
}  // namespace functor
template struct functor::FusedTileGemmFunctor<GPUDevice, Eigen::half>;
}  // namespace tensorflow
#endif
