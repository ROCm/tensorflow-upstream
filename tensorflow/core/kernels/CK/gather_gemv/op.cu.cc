#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gather_gemv_xdl_cshuffle_v3.hpp"
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
template <typename T>
Status ComputeInternal(const GPUDevice& d, const void* mat_A, const void* mat_B,
                       const int* indices, void* mat_D, int head_sz, int seq,
                       int B, int index, int head_num) {
  using PassThrough = ck::tensor_operation::element_wise::PassThrough;
  using Scale = ck::tensor_operation::element_wise::Scale;

  using Row = ck::tensor_layout::gemm::RowMajor;
  using Col = ck::tensor_layout::gemm::ColumnMajor;

  using ADataType = T;
  using BDataType = T;
  using AccDataType = float;
  using CShuffleDataType = T;
  using CDataType = T;

  using AElementOp = PassThrough;
  using BElementOp = PassThrough;
  using CElementOp = Scale;

  using BLayout = Col;

  static constexpr auto GemmDefault =
      ck::tensor_operation::device::GemmSpecialization::MNKPadding;

  using DeviceGemmV2Instance =
      ck::tensor_operation::device::DeviceGatherGemv_Xdl_CShuffleV3<
          ck::tensor_operation::device::GatherGemvType::v1, ADataType,
          BDataType, CDataType, AccDataType, CShuffleDataType, AElementOp,
          BElementOp, CElementOp, GemmDefault, 256, 128, 128, 64, 8, 8, 16, 16,
          4, 4, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 32, 1>,
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 2, S<1, 32, 1, 8>, 8,
          ck::BlockGemmPipelineScheduler::Intrawave,
          ck::BlockGemmPipelineVersion::v1>;

  auto get_c_element_op = [](int HeadSz) {
    float mul = 1.0f / sqrtf(HeadSz * 1.f);
    return CElementOp{mul};
  };

  const auto& stream = d.stream();

  auto gemm = DeviceGemmV2Instance{};
  auto invoker = gemm.MakeInvoker();

  auto argument = gemm.MakeArgument(static_cast<const ADataType*>(mat_A),
                                    static_cast<const BDataType*>(mat_B),
                                    indices, static_cast<CDataType*>(mat_D), B,
                                    index, head_num, seq, head_sz, AElementOp{},
                                    BElementOp{}, get_c_element_op(head_sz));

  if (!gemm.IsSupportedArgument(argument)) {
    return errors::InvalidArgument(gemm.GetTypeString(),
                                   " does not support this problem");
  }

  invoker.Run(argument, StreamConfig{stream, false, 0, 20, 50});

  return Status::OK();
}
namespace functor {
template <typename T>
struct GatherGemvFunctor<GPUDevice, T> {
 public:
  static Status Compute(const GPUDevice& d, const void* mat_A,
                        const void* mat_B, const int* indices, void* mat_D,
                        int head_sz, int seq, int B, int index, int head_num) {
    if constexpr (std::is_same_v<T, Eigen::half>) {
      return ComputeInternal<ck::half_t>(d, mat_A, mat_B, indices, mat_D,
                                         head_sz, seq, B, index, head_num);
    }

    return Status::OK();
  }
};  // struct Fused_Gemm_Bias_Add_Functor
}  // namespace functor
template struct functor::GatherGemvFunctor<GPUDevice, Eigen::half>;
}  // namespace tensorflow
#endif
