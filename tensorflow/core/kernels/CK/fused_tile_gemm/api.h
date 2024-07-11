// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

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
#include "tensorflow/core/platform/logging.h"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

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

// Helper function to print matrix.
template <typename T>
void PrintMatrix(const T* matrix, int rows, int cols, const std::string& name, int num_values = 3) {
    VLOG(2) << "Matrix " << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows; ++i) {
        // Print the first few values.
        for (int j = 0; j < std::min(cols, num_values); ++j) {
            VLOG(2) << std::setw(8) << static_cast<float>(matrix[i * cols + j]) << " ";
        }

        // Print ellipsis if there are more values in the row.
        if (cols > num_values * 2) {
            VLOG(2) << "... ";
        }

        // Print the last few values.
        for (int j = std::max(cols - num_values, num_values); j < cols; ++j) {
            VLOG(2) << std::setw(8) << static_cast<float>(matrix[i * cols + j]) << " ";
        }
        VLOG(2) << std::endl;
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
    VLOG(2) << "FusedTileGemm called with parameters:" << std::endl;
    VLOG(2) << "M: " << M << ", N: " << N << ", K: " << K << std::endl;
    VLOG(2) << "StrideA: " << StrideA << ", StrideB: " << StrideB << ", StrideC: " << StrideC << std::endl;
    VLOG(2) << "KBatch: " << KBatch << std::endl;

    // Print matrices (limited to small sizes for readability).
    PrintMatrix(A, M, K, "A");
    PrintMatrix(B, K, N, "B");
    PrintMatrix(C, M, N, "C (initial)");

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    VLOG(2) << "Element operations initialized." << std::endl;

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

    // Create GEMM instance.
    auto gemm = DeviceGemmV2Instance{};

    // Print GEMM configuration.
    VLOG(2) << "Using GEMM configuration: " << gemm.GetTypeString() << std::endl;

    // Do GEMM.
    auto invoker = gemm.MakeInvoker();
    float ave_time = 0;

    VLOG(2) << "Creating GEMM argument..." << std::endl;
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

    // Print argument details.
    VLOG(2) << "Argument details: " << std::endl;
    VLOG(2) << "M: " << M << ", N: " << N << ", K: " << K << ", StrideA: " << StrideA << ", StrideB: " << StrideB << ", StrideC: " << StrideC << std::endl;
    VLOG(2) << "KBatch: " << KBatch << std::endl;

    if (!gemm.IsSupportedArgument(argument)) {
        std::cerr << "GEMM configuration does not support this problem" << std::endl;
        return;
    }

    VLOG(2) << "Running GEMM operation!" << std::endl;
    ave_time = invoker.Run(argument, StreamConfig{stream, 1});
    LOG(INFO) << "GEMM operation completed successfully!" << std::endl;
    LOG(INFO) << "Average time: " << ave_time << " ms" << std::endl;

    // Print result matrix
    PrintMatrix(C, M, N, "C (result)");
}
