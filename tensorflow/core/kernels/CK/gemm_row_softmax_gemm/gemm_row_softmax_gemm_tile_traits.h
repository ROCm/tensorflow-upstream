// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadGemmM_, bool kPadGemmN_, bool kPadGemmK_,
          index_t kGemmBlockPerCu_>
struct GemmTileTraits {
  static constexpr bool kPadGemmM = kPadGemmM_;
  static constexpr bool kPadGemmN = kPadGemmN_;
  static constexpr bool kPadGemmK = kPadGemmK_;

  static constexpr index_t kGemmBlockPerCu = kGemmBlockPerCu_;
};

}  // namespace ck_tile
