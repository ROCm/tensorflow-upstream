// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockTile_,  // sequence<...> to indicate the tile sizes for
                                // M, N, K of one Gemm
          typename GemmBlockWarps_, typename GemmWarpTile_>
struct GemmTileSetting {
  using BlockTile = remove_cvref_t<BlockTile_>;

  using GemmBlockWarps = remove_cvref_t<GemmBlockWarps_>;
  using GemmWarpTile = remove_cvref_t<GemmWarpTile_>;

  static constexpr index_t GemmNumWarps =
      reduce_on_sequence(GemmBlockWarps{}, multiplies{}, number<1>{});

  static_assert(GemmNumWarps > 0,
                "Invalid number of warps in the tile occurred!");

  static constexpr index_t kGemmM = BlockTile::at(number<0>{});
  static constexpr index_t kGemmN = BlockTile::at(number<1>{});
  static constexpr index_t kGemmK = BlockTile::at(number<2>{});
  static constexpr index_t kMaxK = BlockTile::at(number<3>{});
};

}  // namespace ck_tile
