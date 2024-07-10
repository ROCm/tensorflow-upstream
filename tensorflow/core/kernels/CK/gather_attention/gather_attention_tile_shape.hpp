// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockTile_,  // sequence<...> to indicate the tile sizes for
                                // M, N, K of one Gemm
          typename Gemm0BlockWarps_, typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_, typename Gemm1WarpTile_>
struct GatherAttentionTileSetting {
  using BlockTile = remove_cvref_t<BlockTile_>;
  using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
  using Gemm0WarpTile = remove_cvref_t<Gemm0WarpTile_>;
  using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
  using Gemm1WarpTile = remove_cvref_t<Gemm1WarpTile_>;

  // GemmK dim of any BlockGemm has only one warp since split-k is expensive to
  // use, so GemmN dim of Gemm0 must also has only one warp since it is the
  // GemmK dim of Gemm1, but GemmN dim of Gemm1 can have more than one warps
  static constexpr index_t NumWarpsGemm0 =
      reduce_on_sequence(Gemm0BlockWarps{}, multiplies{}, number<1>{});
  static constexpr index_t NumWarpsGemm1 =
      reduce_on_sequence(Gemm1BlockWarps{}, multiplies{}, number<1>{});

  static_assert(NumWarpsGemm1 % NumWarpsGemm0 == 0, "Check failed!");

  static constexpr index_t NumWarps = NumWarpsGemm1;

  static constexpr index_t kM = BlockTile::at(number<0>{});
  static constexpr index_t kN0 =
      BlockTile::at(number<1>{});  // tile size along seqlen
  static constexpr index_t kK0 =
      BlockTile::at(number<2>{});  // tile size along AB0Gemm unroll
  static constexpr index_t kN1 =
      BlockTile::at(number<3>{});  // tile size along B1/D head_sz dim
  static constexpr index_t kK1 =
      BlockTile::at(number<4>{});  // tile size along PB1Gemm unroll
  static constexpr index_t kMaxK = BlockTile::at(number<5>{});

  static_assert(kMaxK % kN1 == 0, "Check failed!");
  static_assert(kMaxK % kK0 == 0, "Check failed!");
  static_assert(kN0 % kK1 == 0, "Check failed!");
};

}  // namespace ck_tile
