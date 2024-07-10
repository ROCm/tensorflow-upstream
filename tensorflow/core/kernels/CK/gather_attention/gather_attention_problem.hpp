// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "gather_attention_tile_shape.hpp"

namespace ck_tile {

template <typename InOutDataType_, typename GemmAccDataType_,
          typename SMPLComputeDataType_, typename MaskDataType_,
          typename GatherAttentionTileShape_>
struct GatherAttentionProblem {
  using InOutDataType = remove_cvref_t<InOutDataType_>;
  using GemmAccDataType = remove_cvref_t<GemmAccDataType_>;

  // DataType used when computing Softmax
  using SMPLComputeDataType = remove_cvref_t<SMPLComputeDataType_>;
  using MaskDataType = remove_cvref_t<MaskDataType_>;

  using GatherAttentionTileShape = remove_cvref_t<GatherAttentionTileShape_>;

  static constexpr index_t kBlockSize =
      GatherAttentionTileShape::NumWarps * get_warp_size();
};

}  // namespace ck_tile
