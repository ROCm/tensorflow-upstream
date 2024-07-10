// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "gemm_row_softmax_gemm_tile_shape.h"

// Type configuration
template <typename DataType>
struct GemmRowSoftmaxGemmTypeConfig;

template <>
struct GemmRowSoftmaxGemmTypeConfig<ck_tile::fp16_t> {
  using GemmAccDataType = float;
  using SMComputeDataType = float;
};

template <>
struct GemmRowSoftmaxGemmTypeConfig<ck_tile::bf16_t> {
  using GemmAccDataType = float;
  using SMComputeDataType = float;
};

// Tile configuraton for GemmRowSoftmax kernel
template <ck_tile::index_t MaxK>
struct Gemm0RowSoftmaxBlockTile;

template <>
struct Gemm0RowSoftmaxBlockTile<8> {
  using tile_lengths = ck_tile::sequence<64, 64, 8, 8>;
  using gemm_warps = ck_tile::sequence<2, 1, 1>;
};

template <>
struct Gemm0RowSoftmaxBlockTile<16> {
  using tile_lengths = ck_tile::sequence<64, 64, 8, 16>;
  using gemm_warps = ck_tile::sequence<2, 1, 1>;
};

template <>
struct Gemm0RowSoftmaxBlockTile<32> {
  using tile_lengths = ck_tile::sequence<64, 64, 8, 32>;
  using gemm_warps = ck_tile::sequence<2, 1, 1>;
};

// Tile configuraton for Gemm kernel
template <ck_tile::index_t MaxK>
struct Gemm1BlockTile;

template <>
struct Gemm1BlockTile<16> {
  using tile_lengths = ck_tile::sequence<64, 64, 16, 16>;
  using gemm_warps = ck_tile::sequence<2, 1, 1>;
};

template <>
struct Gemm1BlockTile<32> {
  using tile_lengths = ck_tile::sequence<64, 64, 16, 32>;
  using gemm_warps = ck_tile::sequence<2, 1, 1>;
};

template <>
struct Gemm1BlockTile<64> {
  using tile_lengths = ck_tile::sequence<64, 64, 16, 64>;
  using gemm_warps = ck_tile::sequence<2, 1, 1>;
};

using WarpTile1 = ck_tile::sequence<32, 32, 8>;
using WarpTile2 = ck_tile::sequence<32, 32, 16>;

template <ck_tile::index_t MaxK>
struct Gemm0RowSoftmaxTileShape;

template <>
struct Gemm0RowSoftmaxTileShape<8>
    : ck_tile::GemmTileSetting<
          typename Gemm0RowSoftmaxBlockTile<8>::tile_lengths,
          typename Gemm0RowSoftmaxBlockTile<8>::gemm_warps, WarpTile1> {};

template <>
struct Gemm0RowSoftmaxTileShape<16>
    : ck_tile::GemmTileSetting<
          typename Gemm0RowSoftmaxBlockTile<16>::tile_lengths,
          typename Gemm0RowSoftmaxBlockTile<16>::gemm_warps, WarpTile1> {};

template <>
struct Gemm0RowSoftmaxTileShape<32>
    : ck_tile::GemmTileSetting<
          typename Gemm0RowSoftmaxBlockTile<32>::tile_lengths,
          typename Gemm0RowSoftmaxBlockTile<32>::gemm_warps, WarpTile1> {};

template <ck_tile::index_t MaxK>
struct Gemm1TileShape;

template <>
struct Gemm1TileShape<16>
    : ck_tile::GemmTileSetting<typename Gemm1BlockTile<16>::tile_lengths,
                               typename Gemm1BlockTile<16>::gemm_warps,
                               WarpTile2> {};

template <>
struct Gemm1TileShape<32>
    : ck_tile::GemmTileSetting<typename Gemm1BlockTile<32>::tile_lengths,
                               typename Gemm1BlockTile<32>::gemm_warps,
                               WarpTile2> {};

template <>
struct Gemm1TileShape<64>
    : ck_tile::GemmTileSetting<typename Gemm1BlockTile<64>::tile_lengths,
                               typename Gemm1BlockTile<64>::gemm_warps,
                               WarpTile2> {};
