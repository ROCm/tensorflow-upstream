
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>

#include "ck_tile/05_gemm_row_softmax_gemm/gemm_row_softmax_gemm_dispatch.hpp"

// clang-format off
template void run_gemm_row_softmax_gemm<ck_tile::fp16_t, ck_tile::fp16_t, 16, 16>(const GemmRowSoftmaxGemmParams& param, hipStream_t stream);
// clang-format on
