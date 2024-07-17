
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>

#include "ck_tile/08_gemv_softmax_gemv/gemv_softmax_gemv_dispatch.hpp"

// clang-format off
template void run_gemv_softmax_gemv<ck_tile::fp16_t, ck_tile::fp16_t, 128>(const GemvSoftmaxGemvParams& param, hipStream_t stream);
// clang-format on
