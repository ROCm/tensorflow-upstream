// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadSeqLen_, bool kPadAB0HeadSz_, bool kPadB1DHeadSz_,
          index_t kBlockPerCu_>
struct GatherAttentionTileTraits {
  static constexpr bool kPadSeqLen = kPadSeqLen_;
  static constexpr bool kPadAB0HeadSz = kPadAB0HeadSz_;
  static constexpr bool kPadB1DHeadSz = kPadB1DHeadSz_;

  static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

}  // namespace ck_tile
