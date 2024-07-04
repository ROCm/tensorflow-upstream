#pragma once
// macros for cutlass changes required by ali customized logic, must be defined before cutlass includes

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNPadding;

// clang-format off
using DeviceGemmV2Instance =
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType,
        PassThrough, PassThrough, PassThrough, GemmDefault,
        256,
        224, 256,
        64, 8, 2,
        16,   16,
        7,    8,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 0,
        S<8, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>,
        1, 8, 2, 0,
        1, 2, S<1, 32, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>;
// clang-format on

template<int N_TILE>
void FusedGemmBiasAdd(
    int M,
    int N,
    int K,
    int Batch,
    void* A0,
    void* B0,
    void* C0,
    void* D0,
    gpuStream_t stream
){

// #if 0
ck::half_t alpha0 = ck::half_t(1);
ck::half_t beta0 = ck::half_t(1);
ck::gemm::GemmCoord problem_size_0(M, N, K);
typename Gemm0::Arguments arguments_0{
    problem_size_0,
    {reinterpret_cast<cutlass::half_t*>(A0), K}, M * K,
    {reinterpret_cast<cutlass::half_t*>(B0), K}, N * K,
    {reinterpret_cast<cutlass::half_t*>(C0), 0}, N,
    {reinterpret_cast<cutlass::half_t*>(D0), N}, M * N,
    { alpha0, beta0 },
    Batch};

    // do GEMM
    auto gemm      = DeviceGemmV2Instance{};
    auto invoker   = gemm.MakeInvoker();
    float ave_time = 0;

    auto argument = gemm.MakeArgument(
        static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
        static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
        M,
        N,
        K,
        StrideA,
        StrideB,
        StrideC,
        Batch,
        a_element_op,
        b_element_op,
        c_element_op);

    ave_time = invoker.Run(argument, StreamConfig{nullptr, 1});
// #endif
}
