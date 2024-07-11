// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "gemv_softmax_gemv_tile_shape.hpp"

namespace ck_tile {

template <typename InOutDataType_, typename GemmAccDataType_,
          typename SMPLComputeDataType_, typename MaskDataType_,
          typename GemvSoftmaxGemvTileShape_>
struct GemvSoftmaxGemvProblem {
  using InOutDataType = remove_cvref_t<InOutDataType_>;
  using GemmAccDataType = remove_cvref_t<GemmAccDataType_>;

  // DataType used when computing Softmax
  using SMPLComputeDataType = remove_cvref_t<SMPLComputeDataType_>;
  using MaskDataType = remove_cvref_t<MaskDataType_>;

  using GemvSoftmaxGemvTileShape = remove_cvref_t<GemvSoftmaxGemvTileShape_>;

  static constexpr index_t kBlockSize =
      GemvSoftmaxGemvTileShape::NumWarps * get_warp_size();
};

}  // namespace ck_tile
