
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>

#include "tensorflow/core/kernels/CK/gather_attention/gather_attention_dispatch.hpp"

// clang-format off
template void run_gather_attention<ck_tile::fp16_t, ck_tile::fp16_t, 64>(const GatherAttentionParams& param, hipStream_t stream);
// clang-format on
