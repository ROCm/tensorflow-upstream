#pragma once
// macros for cutlass changes required by ali customized logic, must be defined before cutlass includes

#include "common.h"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"
#include <iomanip>
// Types.
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

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmV2Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        256,
        128, 128, 
        64, 8, 8,
        16,   16,
        4,    4,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        1, 2, S<1, 32, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>;
// clang-format on

// Helper function to print matrix
template <typename T>
void PrintMatrix(const T* matrix, int rows, int cols, const std::string& name, int num_values = 3) {
    std::cout << "Matrix " << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows; ++i) {
        // Print the first few values
        for (int j = 0; j < std::min(cols, num_values); ++j) {
            std::cout << std::setw(8) << static_cast<float>(matrix[i * cols + j]) << " ";
        }

        // Print ellipsis if there are more values in the row
        if (cols > num_values * 2) {
            std::cout << "... ";
        }

        // Print the last few values
        for (int j = std::max(cols - num_values, num_values); j < cols; ++j) {
            std::cout << std::setw(8) << static_cast<float>(matrix[i * cols + j]) << " ";
        }
        std::cout << std::endl;
    }
}

template<int N_TILE>
void FusedTileGemm(
    int M,
    int N,
    int K,
    int KBatch,
    int StrideA,
    int StrideB,
    int StrideC,
    const ADataType* A,
    const BDataType* B,
    CDataType* C,
    hipStream_t stream
){
    std::cout << "FusedTileGemm called with parameters:" << std::endl;
    std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;
    std::cout << "StrideA: " << StrideA << ", StrideB: " << StrideB << ", StrideC: " << StrideC << std::endl;
    std::cout << "KBatch: " << KBatch << std::endl;

    // Print input pointers
    std::cout << "Pointer A: " << static_cast<const void*>(A) << std::endl;
    std::cout << "Pointer B: " << static_cast<const void*>(B) << std::endl;
    std::cout << "Pointer C: " << static_cast<void*>(C) << std::endl;

    // Print matrices (limited to small sizes for readability)
    PrintMatrix(A, M, K, "A");
    PrintMatrix(B, K, N, "B");
    PrintMatrix(C, M, N, "C (initial)");

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // Print element operations
    std::cout << "Element operations initialized." << std::endl;

    // Configure GEMM parameters
    using DeviceGemmV2Instance = 
        ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
            ALayout, BLayout, CLayout,
            ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,
            PassThrough, PassThrough, PassThrough, GemmDefault,
            256,
            128, 128,
            64, 8, 8,
            16, 16,
            4, 4,
            S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
            2, 8, 8, 0,
            S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
            2, 8, 8, 0,
            1, 2, S<1, 32, 1, 8>, 8,
            ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3>;

    // Create GEMM instance
    auto gemm = DeviceGemmV2Instance{};

    // Print GEMM configuration
    std::cout << "Using GEMM configuration: " << gemm.GetTypeString() << std::endl;

    // Do GEMM
    auto invoker = gemm.MakeInvoker();
    float ave_time = 0;

    // Print before argument creation
    std::cout << "Creating GEMM argument..." << std::endl;

    auto argument = gemm.MakeArgument(
        A,
        B,
        C,
        M,
        N,
        K,
        StrideA,
        StrideB,
        StrideC,
        KBatch,
        a_element_op,
        b_element_op,
        c_element_op);

    // Print argument details
    std::cout << "Argument details: " << std::endl;
    std::cout << "A: " << static_cast<const void*>(A) << ", B: " << static_cast<const void*>(B) << ", C: " << static_cast<void*>(C) << std::endl;
    std::cout << "M: " << M << ", N: " << N << ", K: " << K << ", StrideA: " << StrideA << ", StrideB: " << StrideB << ", StrideC: " << StrideC << std::endl;
    std::cout << "KBatch: " << KBatch << std::endl;

    if (!gemm.IsSupportedArgument(argument)) {
        std::cerr << "GEMM configuration does not support this problem" << std::endl;
        return;
    }

    std::cout << "Running GEMM operation!" << std::endl;
    ave_time = invoker.Run(argument, StreamConfig{stream, 1});
    std::cout << "GEMM operation completed successfully!" << std::endl;
    std::cout << "Average time: " << ave_time << " ms" << std::endl;

    // Print result matrix
    PrintMatrix(C, M, N, "C (result)");
}
