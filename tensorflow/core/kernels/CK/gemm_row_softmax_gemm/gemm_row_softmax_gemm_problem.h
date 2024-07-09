// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "gemm_row_softmax_gemm_tile_shape.h"

namespace ck_tile {

template <typename InOutDataType_,
          typename GemmAccDataType_,
          typename SMComputeDataType_,
          typename MaskDataType_,
          typename GemmRowSoftmaxTileShape_,
          typename GemmTileShape_>
struct GemmRowSoftmaxGemmProblem
{
    using InOutDataType   = remove_cvref_t<InOutDataType_>;
    using GemmAccDataType = remove_cvref_t<GemmAccDataType_>;

    // DataType used when computing Softmax
    using SMComputeDataType = remove_cvref_t<SMComputeDataType_>;
    using MaskDataType      = remove_cvref_t<MaskDataType_>;

    using GemmRowSoftmaxTileShape = remove_cvref_t<GemmRowSoftmaxTileShape_>;
    using GemmTileShape           = remove_cvref_t<GemmTileShape_>;

    // clang-format off
    // Gemm0 is the Gemm of GemmRowSoftmax, in which the Gemm and RowSoftmax are together implemented in a kernel
    // Gemm1 is the second Gemm of GemmRowSoftmaxGemm, which is implemented as a standalone Gemm kernel
    // clang-format on
    static constexpr index_t kGemm0BlockSize =
        GemmRowSoftmaxTileShape::GemmNumWarps * get_warp_size();
    static constexpr index_t kGemm1BlockSize = GemmTileShape::GemmNumWarps * get_warp_size();
};

} // namespace ck_tile
