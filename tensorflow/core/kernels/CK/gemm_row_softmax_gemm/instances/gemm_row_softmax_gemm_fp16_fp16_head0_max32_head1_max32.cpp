
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>

#include "tensorflow/core/kernels/CK/gemm_row_softmax_gemm/gemm_row_softmax_gemm_dispatch.h"

// clang-format off
template void run_gemm_row_softmax_gemm<ck_tile::fp16_t, ck_tile::fp16_t, 32, 32>(const GemmRowSoftmaxGemmParams& param, hipStream_t stream);
// clang-format on
