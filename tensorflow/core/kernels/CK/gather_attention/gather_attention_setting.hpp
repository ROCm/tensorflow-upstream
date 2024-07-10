// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "gather_attention_tile_shape.hpp"

// Type configuration
template <typename DataType>
struct GatherAttentionTypeConfig;

template <>
struct GatherAttentionTypeConfig<ck_tile::fp16_t> {
  using GemmAccDataType = float;
  using SMComputeDataType = float;
};

template <>
struct GatherAttentionTypeConfig<ck_tile::bf16_t> {
  using GemmAccDataType = float;
  using SMComputeDataType = float;
};

template <ck_tile::index_t MaxK>
struct GatherAttnBlockTile;

template <>
struct GatherAttnBlockTile<32> {
  using type = ck_tile::sequence<16, 32, 16, 32, 16, 32>;
  using gemm0_warps = ck_tile::sequence<1, 1, 1>;
  using gemm1_warps = ck_tile::sequence<1, 2, 1>;
};

template <>
struct GatherAttnBlockTile<64> {
  using type = ck_tile::sequence<16, 32, 16, 64, 16, 64>;
  using gemm0_warps = ck_tile::sequence<1, 1, 1>;
  using gemm1_warps = ck_tile::sequence<1, 2, 1>;
};

template <>
struct GatherAttnBlockTile<128> {
  using type = ck_tile::sequence<16, 32, 16, 128, 16, 128>;
  using gemm0_warps = ck_tile::sequence<1, 1, 1>;
  using gemm1_warps = ck_tile::sequence<1, 2, 1>;
};

template <>
struct GatherAttnBlockTile<256> {
  using type = ck_tile::sequence<16, 32, 16, 256, 16, 256>;
  using gemm0_warps = ck_tile::sequence<1, 1, 1>;
  using gemm1_warps = ck_tile::sequence<1, 2, 1>;
};

using GatherAttnWarpTile = ck_tile::sequence<16, 16, 16>;

template <ck_tile::index_t MaxK>
struct GatherAttentionTileShape;

template <>
struct GatherAttentionTileShape<32>
    : ck_tile::GatherAttentionTileSetting<
          typename GatherAttnBlockTile<32>::type,
          typename GatherAttnBlockTile<32>::gemm0_warps, GatherAttnWarpTile,
          typename GatherAttnBlockTile<32>::gemm1_warps, GatherAttnWarpTile> {};

template <>
struct GatherAttentionTileShape<64>
    : ck_tile::GatherAttentionTileSetting<
          typename GatherAttnBlockTile<64>::type,
          typename GatherAttnBlockTile<64>::gemm0_warps, GatherAttnWarpTile,
          typename GatherAttnBlockTile<64>::gemm1_warps, GatherAttnWarpTile> {};

template <>
struct GatherAttentionTileShape<128>
    : ck_tile::GatherAttentionTileSetting<
          typename GatherAttnBlockTile<128>::type,
          typename GatherAttnBlockTile<128>::gemm0_warps, GatherAttnWarpTile,
          typename GatherAttnBlockTile<128>::gemm1_warps, GatherAttnWarpTile> {
};

template <>
struct GatherAttentionTileShape<256>
    : ck_tile::GatherAttentionTileSetting<
          typename GatherAttnBlockTile<256>::type,
          typename GatherAttnBlockTile<256>::gemm0_warps, GatherAttnWarpTile,
          typename GatherAttnBlockTile<256>::gemm1_warps, GatherAttnWarpTile> {
};
