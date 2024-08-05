
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>

#include "ck_tile/06_gemm_ln/gemm_ln_dispatch.hpp"

// clang-format off
template float run_gemm_ln<ck_tile::fp16_t, 16>(const gemm_ln_args& param, ck_tile::stream_config stream);
// clang-format on
